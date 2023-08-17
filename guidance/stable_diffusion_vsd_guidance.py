from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline, DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from os.path import isfile
from pathlib import Path
import random
from contextlib import contextmanager

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.cuda.amp import custom_bwd, custom_fwd

# from threestudio
class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x):
        return self.module(x).to(self.dtype)

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusionVSDGuidance(nn.Module):
    def __init__(self, device, fp16, vram_O, sd_version='2.1', hf_key=None, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')
        
        # from threestudio
        lora_model_key = "stabilityai/stable-diffusion-2-1"

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t).to(device)
        # from threestudio
        pipe_lora = StableDiffusionPipeline.from_pretrained(lora_model_key, torch_dtype=self.precision_t).to(device)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)

            pipe_lora.enable_sequential_cpu_offload()
            pipe_lora.enable_vae_slicing()
            pipe_lora.unet.to(memory_format=torch.channels_last)
            pipe_lora.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
            pipe_lora.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
    
        # from threestudio
        del pipe_lora.vae
        pipe_lora.vae = pipe.vae

        self.vae_lora = pipe_lora.vae
        self.unet_lora = pipe_lora.unet
        # from threestudio
        del pipe_lora.text_encoder

        torch.cuda.empty_cache()

        # from threestudio
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.unet_lora.parameters():
            p.requires_grad_(False)

        # from threestudio
        # FIXME: hard-coded dims
        self.camera_embedding = ToWeightsDType(
            TimestepEmbedding(16, 1280).to(device), self.precision_t
        ).to(device)
        self.unet_lora.class_embedding = self.camera_embedding
    
        # set up LoRA layers
        lora_attn_procs = {}
        for name in self.unet_lora.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet_lora.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet_lora.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet_lora.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet_lora.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )

        self.unet_lora.set_attn_processor(lora_attn_procs)

        self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors)
        self.lora_layers._load_state_dict_pre_hooks.clear()
        self.lora_layers._state_dict_hooks.clear()

        # NOTE: threestudio uses DDPM here!
        self.scheduler = DDPMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)
        self.scheduler_lora = DDPMScheduler.from_pretrained(
            lora_model_key,
            subfolder="scheduler",
            torch_dtype=self.precision_t,
        )

        pipe.scheduler = self.scheduler
        pipe_lora.scheduler = self.scheduler_lora

        # del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        if not vram_O:
            self.unet_lora.to(device)

        print(f'[INFO] loaded stable diffusion!')
    
    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]

        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings
    
    def compute_grad_vsd(self,
                        text_embeddings, # ["BB 77 768"]
                        pred_rgb, # [B 4 64 64]
                        camera_condition, # [B, 4, 4]
                        guidance_scale=7.5,
                        guidance_scale_lora=1.0,
                        save_guidance_path:Path=None,
                        as_latent=False
                        ):
        
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        B = latents.shape[0]
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (B,), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            # noise prediction from SD
            with self.disable_unet_class_embedding(self.unet) as unet:
                noise_pred_pretrain = unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

            # noise prediction from LoRA
            noise_pred_est = self.unet_lora(
                latent_model_input,
                tt,
                encoder_hidden_states=text_embeddings,
                class_labels=torch.cat(
                    [
                        camera_condition.view(B, -1),
                        torch.zeros_like(camera_condition.view(B, -1)),
                    ],
                    dim=0,
                ),
                cross_attention_kwargs={"scale": 1.0},
                ).sample

            # NOTE: this was used previously
            # perform guidance (high scale from paper!)
            # noise_pred_uncond, noise_pred_pos = noise_pred_pretrain.chunk(2)
            # noise_pred_pretrain = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        # NOTE: order of text and unconditional is reversed compared to what was used previously
        (
            noise_pred_pretrain_text,
            noise_pred_pretrain_uncond,
        ) = noise_pred_pretrain.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        # NOTE: to
        noise_pred_pretrain = noise_pred_pretrain_uncond + guidance_scale * (
            noise_pred_pretrain_text - noise_pred_pretrain_uncond
        )

        assert self.scheduler.config.prediction_type == "epsilon"

        (
            noise_pred_est_text,
            noise_pred_est_uncond,
        ) = noise_pred_est.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_est = noise_pred_est_uncond + guidance_scale_lora * (
            noise_pred_est_text - noise_pred_est_uncond
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        grad = w * (noise_pred_pretrain - noise_pred_est)
        grad = torch.nan_to_num(grad)

        if save_guidance_path:
            with torch.no_grad():
                if as_latent:
                    pred_rgb_512 = self.decode_latents(latents)

                # visualize predicted denoised image
                # The following block of code is equivalent to `predict_start_from_noise`...
                # see zero123_utils.py's version for a simpler implementation.
                alphas = self.scheduler.alphas.to(latents)
                total_timesteps = self.max_step - self.min_step + 1
                index = total_timesteps - t.to(latents.device) - 1 
                b = len(noise_pred_pretrain)
                a_t = alphas[index].reshape(b,1,1,1).to(self.device)
                sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
                sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(self.device)                
                pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred_pretrain) / a_t.sqrt() # current prediction for x_0
                result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))

                # visualize noisier image
                result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))

                # TODO: also denoise all-the-way

                # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image],dim=0)
                save_image(viz_images, save_guidance_path)

        return grad, latents

    # grad_scale is not used here
    def train_step(self, text_embeddings, pred_rgb, camera_extrinsics, guidance_scale=7.5, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None):
        batch_size = pred_rgb.shape[0]
        camera_condition = camera_extrinsics

        # TODO: add camera condition
        grad, latents = self.compute_grad_vsd(
            text_embeddings,
            pred_rgb,
            guidance_scale=guidance_scale,
            camera_condition=camera_condition,
            save_guidance_path=save_guidance_path,
            as_latent=as_latent,
        )
        
        target = (latents - grad).detach()
        loss_vsd = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        loss_lora = self.train_lora(latents, text_embeddings, camera_condition)

        return {
            "loss_vsd": loss_vsd,
            "loss_lora": loss_lora,
        }
    

    # from threestudio
    def train_lora(
        self,
        latents, # [B 4 64 64]
        text_embeddings, # [BB 77 768]
        camera_condition, # [B 4 4]
        lora_n_timestamp_samples: int = 1,
        lora_cfg_training: bool = True,
    ):
        B = latents.shape[0]
        latents = latents.detach().repeat(lora_n_timestamp_samples, 1, 1, 1)

        t = torch.randint(
            int(self.num_train_timesteps * 0.0),
            int(self.num_train_timesteps * 1.0),
            [B * lora_n_timestamp_samples],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler_lora.add_noise(latents, noise, t)
        if self.scheduler_lora.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler_lora.config.prediction_type == "v_prediction":
            target = self.scheduler_lora.get_velocity(latents, noise, t)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler_lora.config.prediction_type}"
            )
        # use view-independent *conditioning* text embeddings in LoRA
        text_embeddings, _ = text_embeddings.chunk(2)
        if lora_cfg_training and random.random() < 0.1:
            camera_condition = torch.zeros_like(camera_condition)
        noise_pred = self.unet_lora(
                noisy_latents,
                t,
                encoder_hidden_states=text_embeddings.repeat(
                    lora_n_timestamp_samples, 1, 1
                ),
                class_labels=camera_condition.view(B, -1).repeat(
                    lora_n_timestamp_samples, 1
                ),
                cross_attention_kwargs={"scale": 1.0},
            ).sample

        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")


    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts) # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()




