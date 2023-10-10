import logging
import os
import os.path as osp
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Union

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor, AttnAddedKVProcessor2_0, LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor, LoRAAttnProcessor2_0, SlicedAttnAddedKVProcessor)
from PIL import Image
from rich.logging import RichHandler
from transformers import (AutoImageProcessor, CLIPImageProcessor,
                          CLIPTextModel, CLIPTokenizer,
                          UperNetForSemanticSegmentation)
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin

from diffusers.pipelines import StableDiffusionPipeline
from animatediff.pipelines import AnimationPipeline, load_text_embeddings
from animatediff.models.clip import CLIPSkipTextModel
from animatediff import console, get_dir
from animatediff.generate import create_pipeline
from animatediff.models.unet import UNet3DConditionModel
from animatediff.settings import (ModelConfig, get_infer_config,
                                  get_model_config)
from animatediff.utils.model import checkpoint_to_pipeline, get_base_model
from animatediff.utils.pipeline import send_to_device
from animatediff.utils.util import is_v2_motion_module, save_video

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(console=console, rich_tracebacks=True),
    ],
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

data_dir = get_dir("data")
checkpoint_dir = data_dir.joinpath("models/sd")
pipeline_dir = data_dir.joinpath("models/huggingface")


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='config for base model, lora, textual inversion and motion module.'
    )
    parser.add_argument('--img-path', type=str)

    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save-interval', type=int, default=50)
    parser.add_argument('--with-mm',
                        action='store_true',
                        help='whether train lora with motion module.')
    parser.add_argument('--disable-half',
                        action='store_true',
                        help='whether disable fp16 in lora training')
    parser.add_argument(
        '--n-frames',
        type=int,
        default=8,
        help='use how many frames to train the LoRA with motion module.')
    parser.add_argument(
        '--loss-on-first-frame',
        action='store_true',
        help='If true, only use the first frame to train LoRA.')

    parser.add_argument('--max-img-size', type=int, default=256)
    parser.add_argument('--force-half-vae', action='store_true')
    parser.add_argument('--save-dir', type=str)

    return parser.parse_args()


def main():
    args = get_args()
    config_path = args.config

    config: ModelConfig = get_model_config(config_path)

    motion_module_path = config.motion_module
    is_v2 = is_v2_motion_module(motion_module_path)
    infer_config = get_infer_config(is_v2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    base_model_path: Path = get_base_model(
        'runwayml/stable-diffusion-v1-5',
        local_dir=get_dir("data/models/huggingface"))
    pipeline = create_pipeline(base_model=base_model_path,
                               model_config=config,
                               infer_config=infer_config,
                               use_xformers=False,
                               is_2d=not args.with_mm)
    unet = pipeline.unet
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    noise_scheduler = pipeline.scheduler

    if pipeline.device == device:
        logger.info(
            "Pipeline already on the correct device, skipping device transfer")
    else:
        pipeline = send_to_device(pipeline,
                                  device,
                                  freeze=True,
                                  force_half=args.force_half_vae,
                                  compile=False)

    train_lora(unet, vae, text_encoder, tokenizer, noise_scheduler, pipeline,
               args, config, args.disable_half)


@torch.no_grad()
def img2latent(img_path: str, vae: AutoencoderKL, max_img_size: int = 256):
    img_pil = Image.open(img_path).convert('RGB')
    # NOTE: resize long edge to 768
    long_edge = max(img_pil.size)
    if long_edge > max_img_size:
        scale_factor = max_img_size / long_edge
    else:
        scale_factor = 1
    w, h = img_pil.size
    img_pil = img_pil.resize((int(w * scale_factor), int(h * scale_factor)))
    logging.info(f'Resize image to {img_pil.size}, scale factor is {scale_factor}.')

    img = np.array(Image.open(img_path).convert('RGB'))
    img_ten = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
    img_ten = img_ten / 127.5 - 1  # [-1, 1]

    img_ten = img_ten.to(vae.device, vae.dtype)
    latent = vae.encode(img_ten).latent_dist.sample()
    latent = latent * vae.config.scaling_factor
    return latent


@torch.no_grad()
def latent2image(latents, vae: AutoencoderKL, return_type='np'):
    # latents = 1 / 0.18215 * latents.detach()
    latents = 1 / vae.config['scaling_factor'] * latents.detach(
    )
    image = vae.decode(latents, return_dict=False)[0]
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
    return image


@torch.no_grad()
def text2embedding(prompt: str, pipeline: Union[AnimationPipeline,
                                                StableDiffusionPipeline],
                   config: ModelConfig):
    # NOTE: this function current only work for animate diff
    if isinstance(pipeline, AnimationPipeline):
        prompt_embeds = pipeline._encode_prompt(
            [prompt],
            pipeline.device,
            1,
            do_classifier_free_guidance=False,
            clip_skip=config.clip_skip)
    else:
        tokenizer = pipeline.tokenizer
        text_encoder = pipeline.text_encoder
        max_length = tokenizer.model_max_length
        input_ids = tokenizer(prompt,
                              truncation=True,
                              padding="max_length",
                              max_length=max_length,
                              return_tensors="pt")
        prompt_embeds = text_encoder(input_ids.to(text_encoder.device),
                                     attention_mask=None)
        prompt_embeds = prompt_embeds[0]

        raise NotImplementedError('Only support AnimationPipeline.')

    return prompt_embeds


def get_prompt(model_config: ModelConfig):

    for k in model_config.prompt_map.keys():
        pr = model_config.prompt_map[k]
        if model_config.head_prompt:
            pr = model_config.head_prompt + "," + pr
        if model_config.tail_prompt:
            pr = pr + "," + model_config.tail_prompt

        # NOTE: return the first prompt to train LoRA
        return pr


def add_lora(unet: Union[UNet2DConditionModel, UNet3DConditionModel],
             rank: int):
    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct lora layers
    unet_lora_attn_procs = {}
    unet_lora_parameters = []
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith(
            "attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(
                unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if isinstance(attn_processor,
                      (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor,
                       AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = (LoRAAttnProcessor2_0 if hasattr(
                F, "scaled_dot_product_attention") else LoRAAttnProcessor)

        module = lora_attn_processor_class(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank).to(unet.device, unet.dtype)
        unet_lora_attn_procs[name] = module
        unet_lora_parameters.extend(module.parameters())

    unet.set_attn_processor(unet_lora_attn_procs)

    logging.info(f'Add {len(list(unet_lora_attn_procs.keys()))} LoRAs.')
    return unet, unet_lora_attn_procs, unet_lora_parameters


@torch.no_grad()
def run_visualize_and_save(unet: Union[UNet3DConditionModel,
                                       UNet2DConditionModel],
                           vae: AutoencoderKL, text_encoder: CLIPSkipTextModel,
                           tokenizer: CLIPTokenizer, noise_scheduler,
                           prompt: str, height: int, width: int, with_mm: bool,
                           img_save_prefix: str, lora_save_path: str,
                           config: ModelConfig):
    unet_dtype = unet.dtype
    unet = unet.half()
    # build pipeline
    if with_mm:
        pipeline = AnimationPipeline(vae=vae,
                                     text_encoder=text_encoder,
                                     tokenizer=tokenizer,
                                     unet=unet,
                                     scheduler=noise_scheduler,
                                     feature_extractor=None,
                                     controlnet_map=None)
        prompt_mapping = {0: prompt}
        result = pipeline('dummy',
                          height,
                          width,
                          config.steps,
                          config.guidance_scale,
                          video_length=16,
                          context_frames=16,
                          prompt_map=prompt_mapping,
                          negative_prompt=config.n_prompt[0],
                          clip_skip=config.clip_skip)['videos']
        save_video(result, Path(img_save_prefix + '.gif'))
        logging.info(f'Image saved to {img_save_prefix + ".gif"}')
    else:
        pipeline = StableDiffusionPipeline(vae=vae,
                                           text_encoder=text_encoder,
                                           tokenizer=tokenizer,
                                           unet=unet,
                                           scheduler=noise_scheduler,
                                           safety_checker=None,
                                           feature_extractor=None,
                                           requires_safety_checker=False)
        result = pipeline(
            prompt,
            height,
            width,
            config.steps,
            config.guidance_scale,
        )

    unet = unet.to(dtype=unet_dtype)

    # save lora
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)
    LoraLoaderMixin.save_lora_weights(
        save_directory=lora_save_path,
        unet_lora_layers=unet_lora_layers,
        text_encoder_lora_layers=None,
    )
    logging.info(f'Save LoRA to {lora_save_path}')


def train_lora(unet: Union[UNet3DConditionModel,
                           UNet2DConditionModel], vae: AutoencoderKL,
               text_encoder: CLIPSkipTextModel, tokenizer: CLIPTokenizer,
               noise_scheduler, pipeline: Union[AnimationPipeline,
                                                StableDiffusionPipeline], args,
               config: ModelConfig, disable_half: bool):

    if disable_half:
        unet = unet.float()
        logging.info('Train UNet and LoRA in fp32.')

    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if not args.save_dir:
        save_dir = osp.join('data', 'loras', f'{time_str}-{config.save_name}')
    else:
        save_dir = args.save_dir
    img_vis_dir = osp.join(save_dir, 'vis')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(img_vis_dir, exist_ok=True)

    logger.info(f'Will save outputs to {save_dir}')
    logger.info(f'Visualization results will be saved to {img_vis_dir}.')

    # 0. add lora for UNet
    unet.requires_grad_(False)
    unet, attn_procs, lora_params = add_lora(unet, args.rank)

    # 1. prepare image latent and prompt
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    model_input = img2latent(args.img_path, vae, args.max_img_size)
    model_input = model_input.to(unet.device, unet.dtype)
    prompt = get_prompt(config)
    # encoder_hidden_states = text2embedding(prompt, tokenizer, text_encoder)
    encoder_hidden_states = text2embedding(prompt, pipeline, config)

    # 2. prepare optimizer
    optimizer = torch.optim.Adam(lora_params,
                                 lr=args.lr,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 weight_decay=1e-2)

    with_mm = args.with_mm
    if with_mm:
        # [1, C, H, W] -> [1, C, n_frames, H, W]
        model_input = model_input.unsqueeze(2).repeat(1, 1, args.n_frames, 1,
                                                      1)
        logging.info(
            f'Train LoRA with motion module. n_frames: {args.n_frames}')

    h, w = model_input.shape[-2:]
    h, w = h * 8, w * 8

    # 3. loop
    for idx in tqdm(range(args.max_steps)):
        if (idx + 1) % args.save_interval == 0 or idx == 0:
            # NOTE: do vis and save
            img_vis_prefix = osp.join(img_vis_dir, f'vis_{idx+1}')
            lora_save_path = osp.join(save_dir, f'checkpoint_{idx+1}')

            run_visualize_and_save(unet, vae, text_encoder, tokenizer,
                                   noise_scheduler, prompt, h, w, with_mm,
                                   img_vis_prefix, lora_save_path, config)

        noise = torch.randn_like(model_input)
        # bsz, channels, height, width = model_input.shape
        bsz = model_input.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0,
                                  noise_scheduler.config.num_train_timesteps,
                                  (bsz, ),
                                  device=model_input.device)
        timesteps = timesteps.long()
        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = noise_scheduler.add_noise(model_input, noise,
                                                      timesteps)
        # Predict the noise residual
        model_pred = unet(noisy_model_input.to(unet.device, unet.dtype),
                          timesteps,
                          encoder_hidden_states=encoder_hidden_states.to(
                              unet.device, unet.dtype)).sample
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(model_input, noise,
                                                  timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )

        if args.loss_on_first_frame and args.with_mm:
            # only use the first frame to train LoRA
            model_pred = model_pred[:, 0]
            target = target[:, 0]

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        print(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    img_vis_prefix = osp.join(img_vis_dir, f'vis_{idx+1}')
    lora_save_path = osp.join(save_dir, f'checkpoint_{idx+1}')
    run_visualize_and_save(unet, vae, text_encoder, tokenizer, noise_scheduler,
                           prompt, h, w, with_mm, img_vis_prefix,
                           lora_save_path, config)


if __name__ == '__main__':
    main()
