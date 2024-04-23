#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
import random
import shutil
import torchvision

from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, CLIPModel, CLIPTextConfig
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available

from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import prepare_mask_and_masked_image

# custom dataset
from datasets_utils.vitonhd import VitonHDDataset

# custom utils
from utils.parse_args import parse_args
from utils.save_model_card import save_model_card
from utils.log_validation import log_validation

from typing import Callable, List, Optional, Union

if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# DATASET_NAME_MAPPING = {
#     "lambdalabs/pokemon-blip-captions": ("image", "text"),
# }


def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    # noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.set_timesteps(50, device=device)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # text_encoder = CLIPTextModel.from_pretrained(
        #     args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        # )
        # vae = AutoencoderKL.from_pretrained(
        #     args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        # )
        # text_encoder = CLIPTextModel.from_pretrained(
        #     args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        # )

        clip_model = CLIPModel.from_pretrained(args.pretrained_model_path_of_CLIP)
        text_encoder = CLIPTextModel(CLIPTextConfig.from_pretrained(args.pretrained_model_path_of_CLIP))
        text_encoder.text_model = clip_model.text_model
        
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    # )
        
    # unet = torch.hub.load(map_location=torch.device('cpu'), dataset=args.dataset, repo_or_dir='aimagelab/multimodal-garment-designer', source='github',
                        #   model='mgd', pretrained=False)
        
    unet = torch.hub.load(map_location=device, dataset=args.dataset, repo_or_dir='aimagelab/multimodal-garment-designer', source='github',
                          model='mgd', pretrained=True)
    
    # unet.to(accelerator.device, dtype=weight_dtype)

    # unet.save_pretrained(os.path.join(args.output_dir, "unet"))
    # unet = torch.hub.load(map_location=device, dataset=args.dataset, repo_or_dir='/home/tri/Uni/Year4/Thesis/Experiment/results/diffusers/train_output/unet/', source='local',
    #                       model='diffusion_pytorch_model.safetensors', pretrained=False)

    # unet = UNet2DConditionModel.from_pretrained("/home/tri/Uni/Year4/Thesis/Experiment/results/diffusers/train_output/unet/",use_safetensors=True)
    
    # unet = UNet2DConditionModel.from_pretrained("teslayder/mgd",use_safetensors=True)


    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )

        # ema_unet = torch.hub.load(map_location=torch.device('cpu'), dataset=args.dataset, repo_or_dir='aimagelab/multimodal-garment-designer', source='github',
        #                   model='mgd', pretrained=False)
        
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,    #edit weight decay here
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # if args.dataset_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     dataset = load_dataset(
    #         args.dataset_name,
    #         args.dataset_config_name,
    #         cache_dir=args.cache_dir,
    #         data_dir=args.train_data_dir,
    #     )
    # else:
    #     data_files = {}
    #     if args.train_data_dir is not None:
    #         data_files["train"] = os.path.join(args.train_data_dir, "**")
    #     dataset = load_dataset(
    #         "imagefolder",
    #         data_files=data_files,
    #         cache_dir=args.cache_dir,
    #     )
    #     # See more about loading custom images at
    #     # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # # Preprocessing the datasets.
    # # We need to tokenize inputs and targets.
    # column_names = dataset["train"].column_names

    # # 6. Get the column names for input/target.
    # dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    # if args.image_column is None:
    #     image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    # else:
    #     image_column = args.image_column
    #     if image_column not in column_names:
    #         raise ValueError(
    #             f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
    #         )
    # if args.caption_column is None:
    #     caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    # else:
    #     caption_column = args.caption_column
    #     if caption_column not in column_names:
    #         raise ValueError(
    #             f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
    #         )

    # # Preprocessing the datasets.
    # # We need to tokenize input captions and transform the images.
    # def tokenize_captions(examples, is_train=True):
    #     captions = []
    #     for caption in examples[caption_column]:
    #         if isinstance(caption, str):
    #             captions.append(caption)
    #         elif isinstance(caption, (list, np.ndarray)):
    #             # take a random caption if there are multiple
    #             captions.append(random.choice(caption) if is_train else caption[0])
    #         else:
    #             raise ValueError(
    #                 f"Caption column `{caption_column}` should contain either strings or lists of strings."
    #             )
    #     inputs = tokenizer(
    #         captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    #     )
    #     return inputs.input_ids

    # # Preprocessing the datasets.
    # train_transforms = transforms.Compose(
    #     [
    #         transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    #         transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
    #         transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5], [0.5]),
    #     ]
    # )

    # def preprocess_train(examples):
    #     images = [image.convert("RGB") for image in examples[image_column]]
    #     examples["pixel_values"] = [train_transforms(image) for image in images]
    #     examples["input_ids"] = tokenize_captions(examples)
    #     return examples

    # with accelerator.main_process_first():
    #     if args.max_train_samples is not None:
    #         dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    #     # Set the training transforms
    #     train_dataset = dataset["train"].with_transform(preprocess_train)

    # def collate_fn(examples):
    #     pixel_values = torch.stack([example["pixel_values"] for example in examples])
    #     pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    #     input_ids = torch.stack([example["input_ids"] for example in examples])
    #     return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     batch_size=args.train_batch_size,
    #     num_workers=args.dataloader_num_workers,
    # )

    train_dataset = VitonHDDataset(
        dataroot_path=args.dataset_path,
        phase='train',
        sketch_threshold_range=(20, 20),
        radius=5,
        tokenizer=tokenizer,
        size=(512, 384),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers_test,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(prompt: Union[str, List[str]], device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt: Optional[Union[str, List[str]]] = None):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1: -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings
    
     # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * noise_scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
            mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // vae_scale_factor, width //vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        if isinstance(generator, list):
            masked_image_latents = [
                vae.encode(masked_image[i: i + 1]).latent_dist.sample(generator=generator[i])
                for i in range(batch_size)
            ]
            masked_image_latents = torch.cat(masked_image_latents, dim=0)
        else:
            masked_image_latents = vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = 0.18215 * masked_image_latents

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents
    
    # def _execution_device():
    #     r"""
    #     Returns the device on which the pipeline's models will be executed. After calling
    #     `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
    #     hooks.
    #     """
    #     if device != torch.device("meta") or not hasattr(unet, "_hf_hook"):
    #         return device
    #     for module in unet.modules():
    #         if (
    #                 hasattr(module, "_hf_hook")
    #                 and hasattr(module._hf_hook, "execution_device")
    #                 and module._hf_hook.execution_device is not None
    #         ):
    #             return torch.device(module._hf_hook.execution_device)
    #     return device
    

    # Train!
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                # latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = vae.encode(batch["image"].to(weight_dtype)).latent_dist.sample()
                # latents = latents * vae.config.scaling_factor

                model_img = batch["image"]
                mask_img = batch["inpaint_mask"]
                mask_img = mask_img.type(torch.float32)
                prompt = batch["original_captions"]  # prompts is a list of length N, where N=batch size.
                pose_map = batch["pose_map"]
                sketch = batch["im_sketch"]
                # ext = ".jpg"

                mask_image = mask_img
                image = model_img
                num_channels_latents = vae.config.latent_channels
                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
                height=512
                width=384
                height = height or accelerator.unwrap_model(unet).config.sample_size * vae_scale_factor
                width = width or accelerator.unwrap_model(unet).config.sample_size * vae_scale_factor

                batch_size = 1 if isinstance(prompt, str) else len(prompt)
                # device = _execution_device()
                # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
                # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
                # corresponds to doing no classifier free guidance.
                guidance_scale = 7.5
                do_classifier_free_guidance = guidance_scale > 1.0
                num_images_per_prompt = 1
                negative_prompt = None
                text_embeddings = _encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)

                seed = 1234
                generator = torch.Generator(torch.device("cuda" if torch.cuda.is_available() else "cpu")).manual_seed(seed)


                latents = prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    text_embeddings.dtype,
                    device,
                    generator,
                    latents,
                )

                # 4. Preprocess mask, image and posemap
                mask, masked_image = prepare_mask_and_masked_image(image, mask_image, height, width)

                # 7. Prepare mask latent variables
                mask, masked_image_latents = prepare_mask_latents(
                    mask,
                    masked_image,
                    batch_size * num_images_per_prompt,
                    height,
                    width,
                    text_embeddings.dtype,
                    device,
                    generator,
                    do_classifier_free_guidance,
                )


                num_channels_masked_image = masked_image_latents.shape[1]

                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                # timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                # rand i time step from 0 to 1000
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,) , device=latents.device)
                # timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                latents = noisy_latents

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # Get the text embedding for conditioning
                # encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
                
                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # 5a. Compute the number of steps to run sketch conditioning
                # sketch_conditioning_steps = (1 - sketch_cond_rate) * num_inference_steps
                num_inference_steps = 1
                start_cond_rate = 0.0
                sketch_cond_rate = 0.2
                start_cond_step = int(num_inference_steps * start_cond_rate)
                sketch_start = start_cond_step
                sketch_end = sketch_cond_rate * num_inference_steps + start_cond_step

                # 4. Preprocess mask, image and posemap
                pose_map = torch.nn.functional.interpolate(
                    pose_map, size=(pose_map.shape[2] // 8, pose_map.shape[3] // 8), mode="bilinear"
                )
                no_pose = False
                if no_pose:
                    pose_map = torch.zeros_like(pose_map)
                    
                sketch = torchvision.transforms.functional.resize(
                sketch, size=(sketch.shape[2] // 8, sketch.shape[3] // 8),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                antialias=True)
                sketch = sketch
                # 7a. Prepare pose map latent variables
                pose_map = torch.cat([torch.zeros_like(pose_map), pose_map]) if do_classifier_free_guidance else pose_map
                sketch = torch.cat([torch.zeros_like(sketch), sketch]) if do_classifier_free_guidance else sketch
                
                # # 10a. Sketch conditioning
                # if i < sketch_start or i > sketch_end:
                #     local_sketch = torch.zeros_like(sketch)
                # else:
                local_sketch = sketch
                
                # 8. Check that sizes of mask, masked image and latents match
                num_channels_mask = mask.shape[1]
                num_channels_masked_image = masked_image_latents.shape[1]
                num_channels_pose_map = pose_map.shape[1]
                num_channels_sketch = sketch.shape[1]

                if num_channels_latents + num_channels_mask + num_channels_masked_image + num_channels_pose_map + num_channels_sketch != accelerator.unwrap_model(unet).config.in_channels:
                    raise ValueError(
                        f"Incorrect configuration settings! The config of `pipeline.unet`: {accelerator.unwrap_model(unet).config} expects"
                        f" {accelerator.unwrap_model(unet).config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                        f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                        f" = {num_channels_latents + num_channels_masked_image + num_channels_mask}. Please verify the config of"
                        " `pipeline.unet` or your `mask_image` or `image` input."
                    )
                    
                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timesteps)
                latent_model_input = torch.cat(
                    [latent_model_input, mask, masked_image_latents, pose_map.to(mask.dtype), local_sketch.to(mask.dtype)],
                    dim=1)
                
                # print("2. latent_model_input type:", type(latent_model_input), " ", latent_model_input.size())

                # predict the noise residual
                # model_pred = unet(latent_model_input, timesteps, encoder_hidden_states=text_embeddings, return_dict=False)[0]
                model_pred = unet(latent_model_input, timesteps, encoder_hidden_states=text_embeddings).sample

                # Predict the noise residual and compute loss
                # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings, return_dict=False)[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
                    model_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # take from below accelerator.sync_gradients
            accelerator.log({"train_loss": train_loss}, step=global_step)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    unet,
                    args,
                    accelerator,
                    weight_dtype,
                    global_step,
                )
                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        # pipeline = StableDiffusionPipeline.from_pretrained(
        #     args.pretrained_model_name_or_path,
        #     text_encoder=text_encoder,
        #     vae=vae,
        #     unet=unet,
        #     revision=args.revision,
        #     variant=args.variant,
        # )
        # pipeline.save_pretrained(args.output_dir)

        # torch.save(unet.state_dict(), os.path.join(args.output_dir, "unet_viton.pth"))
            
        unet.to(accelerator.device, dtype=weight_dtype)

        # torch.save(unet.state_dict(), os.path.join(args.output_dir, "unet_viton.pth"))

        unet.save_pretrained(os.path.join(args.output_dir, "save_pretrained_unet"))

        torch.save({
            'epoch': args.num_train_epochs,
            'num_training_steps': args.max_train_steps * accelerator.num_processes,
            'model_state_dict': unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.detach().item(),
            }, os.path.join(args.output_dir, "torch_save_unet"))


        # Run a final round of inference.
        # images = []
        # if args.validation_prompts is not None:
        #     logger.info("Running inference for collecting generated images...")
        #     pipeline = pipeline.to(accelerator.device)
        #     pipeline.torch_dtype = weight_dtype
        #     pipeline.set_progress_bar_config(disable=True)

        #     if args.enable_xformers_memory_efficient_attention:
        #         pipeline.enable_xformers_memory_efficient_attention()

        #     if args.seed is None:
        #         generator = None
        #     else:
        #         generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

        #     for i in range(len(args.validation_prompts)):
        #         with torch.autocast("cuda"):
        #             image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]
        #         images.append(image)

        # if args.push_to_hub:
        #     save_model_card(args, repo_id, images, repo_folder=args.output_dir)
        #     upload_folder(
        #         repo_id=repo_id,
        #         folder_path=args.output_dir,
        #         commit_message="End of training",
        #         ignore_patterns=["step_*", "epoch_*"],
        #     )

    accelerator.end_training()


if __name__ == "__main__":
    main()
