import torch.nn as nn
from typing import Any, Dict, Optional
from compel import Compel
import torch
import numpy as np
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm import tqdm


__SOLVER__ = {}

def register_solver(name: str):
    def wrapper(cls):
        if __SOLVER__.get(name, None) is not None:
            raise ValueError(f"Solver {name} already registered.")
        __SOLVER__[name] = cls
        return cls
    return wrapper

def get_solver(name: str, **kwargs):
    if name not in __SOLVER__:
        raise ValueError(f"Solver {name} does not exist.")
    return __SOLVER__[name](**kwargs)

class StableDiffusion():
    def __init__(self,
                 solver_config: Dict,
                 model_key:str="runwayml/stable-diffusion-v1-5",
                 pipeline_path="runwayml/stable-diffusion-v1-5",
                 device: Optional[torch.device]=None,
                 **kwargs):
        self.device = device

        self.dtype = kwargs.get("pipe_dtype", torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.dtype).to(device)
        pipe.load_lora_weights(pipeline_path)
        self.vae = pipe.vae
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        total_timesteps = len(self.scheduler.alphas)
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        self.skip = total_timesteps // solver_config.num_sampling

        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.sample(*args, **kwargs)

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Solver must implement sample() method.")

    def sample_forward_backward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Solver must implement sample_forward_backward() method.")

    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

    @torch.no_grad()
    def get_text_embed(self, null_prompt, prompt):
        """
        Get text embedding.
        args:
            null_prompt (str): null text
            prompt (str): guidance text
        """
        # null text embedding (negation)
        null_text_input = self.tokenizer(null_prompt,
                                         padding='max_length',
                                         max_length=self.tokenizer.model_max_length,
                                         return_tensors="pt",)
        null_text_embed = self.text_encoder(null_text_input.input_ids.to(self.device))[0]

        # text embedding (guidance)
        if prompt == "":
            text_input = self.tokenizer(prompt,
                                        padding='max_length',
                                        max_length=self.tokenizer.model_max_length,
                                        return_tensors="pt",
                                        truncation=True)
            text_embed = self.text_encoder(text_input.input_ids.to(self.device))[0]
        else:
            text_embed = self.compel(prompt)

        return null_text_embed, text_embed

    def encode(self, x):
        """
        xt -> zt
        """
        return self.vae.encode(x).latent_dist.sample() * 0.18215

    def decode(self, zt):
        """
        zt -> xt
        """
        zt = 1/0.18215 * zt
        img = self.vae.decode(zt).sample.float()
        return img

    def predict_noise(self,
                      zt: torch.Tensor,
                      t: torch.Tensor,
                      uc: torch.Tensor,
                      c: torch.Tensor):
        """
        compuate epsilon_theta for null and condition
        args:
            zt (torch.Tensor): latent features
            t (torch.Tensor): timestep
            uc (torch.Tensor): null-text embedding
            c (torch.Tensor): text embedding
        """
        t_in = t.unsqueeze(0)
        if uc is None:
            noise_c = self.unet(zt, t_in, encoder_hidden_states=c)['sample']
            noise_uc = noise_c
        elif c is None:
            noise_uc = self.unet(zt, t_in, encoder_hidden_states=uc)['sample']
            noise_c = noise_uc
        else:
            c_embed = torch.cat([uc, c], dim=0)
            z_in = torch.cat([zt] * 2)
            noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed)['sample']
            noise_uc, noise_c = noise_pred.chunk(2)

        return noise_uc, noise_c

    def initialize_latent(self,
                          method: str='random',
                          src_img: Optional[torch.Tensor]=None,
                          target_size=None,
                          start_lambda=1000,
                          **kwargs):
        if method == 'ddim':
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('uc'),
                               kwargs.get('c'),
                               cfg_guidance=kwargs.get('cfg_guidance', 0.0))
        elif method == 'npi':
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('c'),
                               kwargs.get('c'),
                               cfg_guidance=1.0)
        elif method == 'random':
            size = kwargs.get('latent_dim', (
                1, 4, target_size[0] // self.vae_scale_factor, target_size[0] // self.vae_scale_factor))
            z = torch.randn(size).to(self.device)
            if src_img is not None:
                src_latent = self.encode(src_img)
                start_lambda = torch.randint(start_lambda, start_lambda+1, (1,), device=self.device)
                z = self.scheduler.add_noise(src_latent, z, start_lambda)
        else:
            raise NotImplementedError

        return z.requires_grad_()

@register_solver("ddim")
class Sampler(StableDiffusion):
    """
    Basic DDIM solver for SD.
    Useful for text-to-image generation
    """

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample_forward_backward(self,
                                gt,
                                cfg_guidance=7.5,
                                prompt=["",""],
                                target_size=(256, 256),
                                start_lambda=1000,
                                **kwargs):
        """
                Main function that defines each solver.
                This will generate samples without considering measurements.
                """
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        # Initialize zT
        zt = self.initialize_latent(target_size=target_size, src_img=gt, start_lambda=start_lambda)
        zt = zt.requires_grad_()

        # get the index of the start_lambda
        neighbor_index = np.argmin(np.absolute(self.scheduler.timesteps.cpu().numpy() - start_lambda))
        start_lambda = self.scheduler.timesteps[neighbor_index]
        start_lambda_idx = (self.scheduler.timesteps == start_lambda).nonzero(as_tuple=True)[0].item()
        pbar = tqdm(self.scheduler.timesteps[start_lambda_idx:], desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)
            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
                # tweedie
            z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()
            zt = at_prev.sqrt() * z0t + (1 - at_prev).sqrt() * noise_pred
            zt.detach_()

        # for the last step, do not add noise
        gt_encode = self.encode(gt.to(self.dtype).to(self.device))
        loss = nn.MSELoss(reduction="none")(gt_encode, z0t)
        loss = loss.mean(dim=(1, 2, 3))
        img = self.decode(z0t)
        img = img.clamp(-1, 1)

        return img, loss


    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               cfg_guidance=7.5,
               prompt=["",""],
               target_size=(512, 512),
               **kwargs):
        """
        Main function that defines each solver.
        This will generate samples without considering measurements.
        """

        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # Initialize zT
        zt = self.initialize_latent(target_size=target_size)
        zt = zt.requires_grad_()

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()


if __name__ == "__main__":
    # print all list of solvers
    print(f"Possble solvers: {[x for x in __SOLVER__.keys()]}")