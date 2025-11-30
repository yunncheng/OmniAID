from PIL import Image
import torch
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline, StableDiffusionXLImg2ImgPipeline, DiffusionPipeline, StableDiffusionPipeline, \
    DDIMScheduler, AutoPipelineForText2Image, AutoencoderKL, AutoPipelineForImage2Image, AutoencoderTiny
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
import torchvision.transforms as transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a custom HybridVAE class that combines the standard VAE and the ConsistencyDecoder
class HybridVAE:
    def __init__(self, encoder_vae, decoder_vae):
        self.encoder = encoder_vae
        self.decoder = decoder_vae
        self.dtype = encoder_vae.dtype
    
    def encode(self, x):
        return self.encoder.encode(x).latent_dist.mean
    
    def decode(self, latents, return_dict=False):
        return self.decoder(latents)
    
    def to(self, device):
        self.encoder = self.encoder.to(device)
        return self

def reconstruct_simple(x, ae, seed, steps=None, tools=None):
    decode_dtype = ae.dtype
    generator = torch.Generator().manual_seed(seed)
    x = x.to(dtype=ae.dtype) * 2.0 - 1.0
    if isinstance(ae, HybridVAE):
        latents = ae.encode(x)
    else:
        latents = retrieve_latents(ae.encode(x), generator=generator)
    reconstructions = ae.decode(
                        latents.to(decode_dtype), return_dict=False
                    )[0]
    reconstructions = (reconstructions / 2 + 0.5).clamp(0, 1)
    return reconstructions


def get_vae(repo_id, return_full=False):
    local_path = '/mnt/shared-storage-user/guoyuncheng/Project/AlignedForensics/training_code/recon/weights'

    if 'stable-diffusion-3' in repo_id or 'sd3' in repo_id.lower():
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", 
            torch_dtype=torch.float32, 
            cache_dir='weights')
        # pipe = StableDiffusion3Pipeline.from_pretrained(f'{local_path}/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671')
        return pipe.vae

    elif 'taesdxl' in repo_id.lower():
        vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxl",
            torch_dtype=torch.float32,
            cache_dir='weights'
        )
        # vae = AutoencoderTiny.from_pretrained(f'{local_path}/models--madebyollin--taesdxl/snapshots/b20258aaef75ef61e659c1e0f14f251cf0ad153e')
        return vae

    elif 'stable-diffusion-xl' in repo_id or ('sdxl' in repo_id.lower() and 'taesdxl' not in repo_id.lower()):
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,
            cache_dir='weights'
        )
        # pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(f'{local_path}/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b')
        return pipe.vae
        
    elif 'stable-diffusion-v2' in repo_id or 'sd21' in repo_id.lower() or 'ft-mse' in repo_id_lower:
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float32,
            cache_dir='weights'
        )
        # pipe = DiffusionPipeline.from_pretrained(f'{local_path}/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6')
        return pipe.vae

    elif 'stable-diffusion-v1' in repo_id or 'sd15' in repo_id.lower() or 'ldm' in repo_id.lower():
        pipe = StableDiffusionPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            cache_dir='weights'
        )
        # pipe = StableDiffusionPipeline.from_pretrained(f'{local_path}/models--sd-legacy--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14')
        return pipe.vae


    elif 'ft-ema' in repo_id.lower():
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-ema",
            torch_dtype=torch.float32,
            cache_dir='weights'
        )
        # vae = AutoencoderKL.from_pretrained(f'{local_path}/models--stabilityai--sd-vae-ft-ema/snapshots/f04b2c4b98319346dad8c65879f680b1997b204a')
        return vae

    elif 'taesd' in repo_id.lower():
        vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd",
            torch_dtype=torch.float32,
            cache_dir='weights'
        )
        # vae = AutoencoderTiny.from_pretrained(f'{local_path}/models--madebyollin--taesd/snapshots/614f76814bbe30edbe2e627ace1c2234c81a2c0e')
        return vae

    else:
        return None


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
