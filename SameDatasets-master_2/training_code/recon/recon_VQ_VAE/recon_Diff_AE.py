from PIL import Image
import torch
from tqdm import tqdm
from diffusers import StableDiffusionXLImg2ImgPipeline, DiffusionPipeline, DDIMScheduler, AutoPipelineForText2Image, \
    AutoencoderKL, AutoPipelineForImage2Image
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from diffusers import StableDiffusion3Pipeline
import torchvision.transforms as transforms
from diffusers import AutoencoderKL  # 仍然使用 AutoencoderKL，但换 Diffusion 专用模型
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def reconstruct_simple(x, ae, seed, steps=None, tools=None):
    generator = torch.Generator().manual_seed(seed)
    x = x.to(dtype=ae.dtype) * 2.0 - 1.0  # [-1, 1] 归一化
    latents = ae.encode(x).latent_dist.sample(generator=generator)  # Diffusion 潜变量
    reconstructions = ae.decode(latents).sample  # 解码
    reconstructions = (reconstructions / 2 + 0.5).clamp(0, 1)  # [0, 1] 范围
    return reconstructions


def get_vae(repo_id: str):
    print(f"Loading VAE from {repo_id}...")

    # 如果是本地路径
    if os.path.isdir(repo_id):
        vae = AutoencoderKL.from_pretrained(
            repo_id,
            local_files_only=True
        )
    else:
        vae = AutoencoderKL.from_pretrained(repo_id)

    return vae



@torch.no_grad()
def ddim_inversion(unet, cond, latent, scheduler, steps=None):
    timesteps = reversed(scheduler.timesteps)
    if steps is not None:
        timesteps = timesteps[:steps]
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        for i, t in enumerate(tqdm(timesteps)):
            cond_batch = cond.repeat(latent.shape[0], 1, 1)

            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[timesteps[i - 1]]
                if i > 0 else scheduler.final_alpha_cumprod
            )

            mu = alpha_prod_t ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

            eps = unet(latent, t, encoder_hidden_states=cond_batch).sample

            pred_x0 = (latent - sigma_prev * eps) / mu_prev
            latent = mu * pred_x0 + sigma * eps
    return latent


@torch.no_grad()
def ddim_sample(x, cond, unet, scheduler, steps=None):
    timesteps = scheduler.timesteps
    if steps is not None:
        timesteps = timesteps[-steps:]
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        for i, t in enumerate(tqdm(timesteps)):
            cond_batch = cond.repeat(x.shape[0], 1, 1)
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else scheduler.final_alpha_cumprod
            )
            mu = alpha_prod_t ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

            eps = unet(x, t, encoder_hidden_states=cond_batch).sample

            pred_x0 = (x - sigma * eps) / mu
            x = mu_prev * pred_x0 + sigma_prev * eps

    return x
