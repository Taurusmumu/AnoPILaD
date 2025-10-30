from typing import Any, Callable, Dict, Optional
import torch
import torch.nn.functional as F
import os
import numpy as np
from diffusers import DDIMScheduler, UNet2DModel
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


class DDPM():
    def __init__(self,
                 solver_config: Dict,
                 model_key:str="google/ddpm-celebahq-256",
                 pipeline_path="google/ddpm-celebahq-256",
                 device: Optional[torch.device]=None,
                 **kwargs):
        self.device = device

        self.dtype = kwargs.get("pipe_dtype", torch.float16)
        self.unet = UNet2DModel.from_pretrained(os.path.join(pipeline_path, 'unet'))
        self.scheduler = DDIMScheduler.from_pretrained(model_key)
        total_timesteps = len(self.scheduler.alphas)
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        self.skip = total_timesteps // solver_config.num_sampling

        self.final_alpha_cumprod = torch.tensor([1.0]).to(device)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.sample(*args, **kwargs)

    # def sample(self, *args: Any, **kwargs: Any) -> Any:
    #     raise NotImplementedError("Solver must implement sample() method.")

    def sample_forward_backward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Solver must implement sample_forward_backward() method.")

    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

    def predict_noise(self,
                      zt: torch.Tensor,
                      t: torch.Tensor):
        """
        compuate epsilon_theta for null and condition
        args:
            zt (torch.Tensor): latent features
            t (torch.Tensor): timestep
        """
        t_in = t.unsqueeze(0)
        noise_pred = self.unet(zt, t_in)['sample']
        # print(noise_pred.size())
        return noise_pred

    def initialize_latent(self,
                          method: str='random',
                          src_img: Optional[torch.Tensor]=None,
                          target_size=None,
                          start_lambda=1000,
                          **kwargs):
        size = kwargs.get('latent_dim', (
            1, 3, target_size[0], target_size[0]))
        z = torch.randn(size).to(self.device)
        if src_img is not None:
            start_lambda = torch.randint(start_lambda, start_lambda + 1, (1,), device=self.device)
            noisy = self.scheduler.add_noise(src_img, z, start_lambda)
        return noisy.requires_grad_(), z


@register_solver("ddim")
class BaseDDIM(DDPM):
    """
    Basic DDIM solver for SD.
    Useful for text-to-image generation
    """

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample_forward_backward(self,
                                gt,
                                target_size=(512, 512),
                                callback_fn=None,
                                start_lambda=1000,
                                **kwargs):
        """
                Main function that defines each solver.
                This will generate samples without considering measurements.
                """
        # Initialize zT
        gt = gt.to(self.dtype).to(self.device)
        zt, noise_p = self.initialize_latent(target_size=target_size, src_img=gt, start_lambda=start_lambda)
        zt = zt.requires_grad_()

        # get the index of the start_lambda
        neighbor_index = np.argmin(np.absolute(self.scheduler.timesteps.cpu().numpy() - start_lambda))
        start_lambda = self.scheduler.timesteps[neighbor_index]
        start_lambda_idx = (self.scheduler.timesteps == start_lambda).nonzero(as_tuple=True)[0].item()
        pbar = tqdm(self.scheduler.timesteps[start_lambda_idx:], desc="AnoDDPM")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)
            with torch.no_grad():
                noise_pred = self.predict_noise(zt, t)

            z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()
            zt = at_prev.sqrt() * z0t + (1 - at_prev).sqrt() * noise_pred

        img = z0t
        img = (img).clamp(-1, 1)
        return img


if __name__ == "__main__":
    # print all list of solvers
    print(f"Possble solvers: {[x for x in __SOLVER__.keys()]}")