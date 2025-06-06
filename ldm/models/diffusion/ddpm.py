"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
import warnings

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config, make_contiguous, cat_interleave
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import IdentityFirstStage, AutoencoderKL
from ldm.models.lidar_diffusion import VQModelInterface
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.data.utils import get_camera_vis, get_lidar_vis, inverse_depth_normalization
from torchvision.transforms import Resize
from ldm.data.lidar_converter import pool_resize

import random
import torch.nn.functional as F

from omegaconf.listconfig import ListConfig

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 u_cond_percent=0,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size 
        self.channels = channels
        self.u_cond_percent=u_cond_percent
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        if k == "inpaint":
            image_data = batch.get("image")
            lidar_data = batch.get("lidar")
        else:
            raise NotImplementedError(f"key {k} not implemented")

        image_data = make_contiguous(image_data)
        lidar_data = make_contiguous(lidar_data)

        return image_data, lidar_data

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 cond_stage_config,
                 first_stage_config=None,
                 lidar_stage_config=None,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 lidar_scale_factor=1.0,
                 scale_by_std=False,
                 use_camera=True,
                 use_lidar=False,
                 range_object_norm=False,
                 range_object_norm_scale=0.75,
                 range_int_norm=False,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        self.range_object_norm = range_object_norm
        self.range_object_norm_scale = range_object_norm_scale
        self.range_int_norm = range_int_norm

        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
    
        # CFG
        self.learnable_vector = nn.Parameter(torch.randn((1,1,768)), requires_grad=False)
        self.bbox_uncond_vector = nn.Parameter(torch.randn((1,1,768)), requires_grad=True)

        self.proj_out=nn.Linear(1024, 768).requires_grad_(False)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
            self.lidar_scale_factor = lidar_scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        self.use_camera = use_camera
        self.use_lidar = use_lidar
        if self.use_camera is False and first_stage_config is not None:
            warnings.warn("No camera input, but first_stage_config is not None. Setting first_stage_config to None.")
            first_stage_config = None
        if self.use_lidar is False and lidar_stage_config is not None:
            warnings.warn("No lidar input, but lidar_stage_config is not None. Setting lidar_stage_config to None.")
            lidar_stage_config = None

        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.instantiate_lidar_stage(lidar_stage_config)

        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None  

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config): # this
        if config is None:
            self.first_stage_model = None
        else:
            model = instantiate_from_config(config)
            self.first_stage_model = model.eval()
            self.first_stage_model.train = disabled_train
            self.first_stage_model.requires_grad_(False)

    def instantiate_lidar_stage(self, config):
        if config is None:
            self.lidar_stage_model = None
        else:
            model = instantiate_from_config(config)
            self.lidar_stage_model = model.eval()
            self.lidar_stage_model.train = disabled_train
            self.lidar_stage_model.requires_grad_(False)

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                self.cond_stage_model.requires_grad_(False)
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model
            self.cond_stage_model.requires_grad_(False)
            if hasattr(self.cond_stage_model, "bbox_embedder"):
                self.cond_stage_model.bbox_embedder.requires_grad_(True)
                # self.cond_stage_model.bbox_embedder.class_embedder.requires_grad_(False)


    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                            force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior, scale_factor=1):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)

        c["ref_image_token"] = self.proj_out(c["ref_image_token"])
        cond = []
        if "ref_image" in self.cond_stage_key:
            cond.append(c["ref_image_token"])
        if "ref_bbox" in self.cond_stage_key:
            cond.append(c["ref_bbox_token"])
        cond = torch.cat(cond, dim=1)

        return cond


    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting
    
    @torch.no_grad()
    def process_conditioning(self, cond, cond_key=None, force_c_encode=False, bs=None, batch=None):
        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if isinstance(cond_key, ListConfig):
                    xc = {k: cond[k].to(self.device) for k in cond_key}
                elif cond_key == 'image':
                    xc = {"ref_image": cond['ref_image'].to(self.device)}
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            if not self.cond_stage_trainable or force_c_encode:
                c = self.get_learned_conditioning(xc)
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}

        return c, xc

    @torch.no_grad()
    def get_input(self, batch, k, force_c_encode=False, bs=None, return_vae_rec=False):
        
        image_data, lidar_data = super().get_input(batch, k)

        def select_first_samples(x, num_samples):
            if isinstance(x, dict):
                return {k: select_first_samples(v, num_samples) for k, v in x.items()}
            elif isinstance(x, torch.Tensor):
                return x[:num_samples]
            return None

        if bs is not None:
            image_data = select_first_samples(image_data, bs)
            lidar_data = select_first_samples(lidar_data, bs)

        z_image, z_lidar = self.encode_all_stages(
            image_gt=image_data.get("GT"),
            image_inpaint=image_data.get("inpaint_image"),
            image_mask=image_data.get("inpaint_mask"),
            range_gt=lidar_data.get("range_data"),
            range_inpaint=lidar_data.get("range_data_inpaint"),
            range_mask=lidar_data.get("range_mask"),
        )

        out = {
            "z": [],
            "cond": [],
        }
        if self.use_camera:
            out["z"].append(z_image)
            c, _ = self.process_conditioning(image_data["cond"], force_c_encode=force_c_encode)
            out["cond"].append(c)

            if return_vae_rec:
                out["image_rec"] = self.decode_first_stage(
                    z_image[:, :4, ...]
                )
                out["image_rec"] = torch.clamp(out["image_rec"], -1., 1.)

        if self.use_lidar:
            # Align lidar feature map with image latent
            if z_lidar.shape[-1] != self.image_size:
                warnings.warn("Cropping lidar feature map to match image latent size.")

            W = z_lidar.shape[-1]
            left, right = W // 2 - self.image_size // 2, W // 2 + self.image_size // 2
            pad = (self.image_size - z_lidar.shape[-2]) // 2
            # out["z"].append(
            #     F.interpolate(z_lidar[..., left:right], size=self.image_size, mode="nearest")
            # )
            # pad top and bottom if lidar feature map is smaller than image latent
            out["z"].append(
                F.pad(z_lidar[..., left:right], (0, 0, pad, pad), mode="constant", value=0)
            )

            # Align bbox with cropped lidar feature map
            lidar_data["cond"]["ref_bbox"][..., 0] = (lidar_data["cond"]["ref_bbox"][..., 0] * W - left) / self.image_size
            lidar_data["cond"]["ref_bbox"][..., 1] += pad / self.image_size
            c, _ = self.process_conditioning(lidar_data["cond"], force_c_encode=force_c_encode)
            out["cond"].append(c)
            out["z_lidar"] = z_lidar[:, :4, ...]

            if return_vae_rec:
                out["lidar_rec"] = self.decode_first_stage(
                    z_lidar[:, :4, ...], module_name="lidar_stage_model"
                )
                out["lidar_rec"] = torch.clamp(out["lidar_rec"], -1., 1.)

        out["z"] = cat_interleave(out["z"])
        if force_c_encode:
            out["cond"] = cat_interleave(out["cond"])
        else:
            out["cond"] = {
                k : cat_interleave([d[k] for d in out["cond"]]) for k in self.cond_stage_key
            }

        return out
        
    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False, module_name="first_stage_model"):
        assert module_name in ["first_stage_model", "lidar_stage_model"]
        module = getattr(self, module_name)
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = module.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        if module_name == "first_stage_model":
            z = 1. / self.scale_factor * z
        else:
            z = 1. / self.lidar_scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(module, VQModelInterface):
                    output_list = [module.decode(z[:, :, :, :, i], force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [module.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(module, VQModelInterface):
                    return module.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return module.decode(z)

        else:
            if isinstance(module, VQModelInterface):
                return module.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                if self.first_stage_key=='inpaint':
                    return module.decode(z[:,:4,:,:])
                else:
                    return module.decode(z)


    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False, module_name="first_stage_model"):
        assert module_name in ["first_stage_model", "lidar_stage_model"]
        module = getattr(self, module_name)
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = module.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        if module_name == "first_stage_model":
            z = 1. / self.scale_factor * z
        else:
            z = 1. / self.lidar_scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(module, VQModelInterface):  
                    output_list = [module.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [module.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(module, VQModelInterface):
                    return module.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return module.decode(z)

        else:
            if isinstance(module, VQModelInterface):
                return module.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return module.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x, module_name="first_stage_model"):
        module = getattr(self, module_name)
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [module.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return module.encode(x)
        else:
            return module.encode(x)
        
    def encode_all_stages(self, image_gt, image_inpaint, image_mask, range_gt, range_inpaint, range_mask):
        z_image, z_lidar = None, None

        if self.use_camera:
            encoder_posterior = self.encode_first_stage(image_gt)
            z = self.get_first_stage_encoding(encoder_posterior, scale_factor=self.scale_factor).detach()

            encoder_posterior_inpaint = self.encode_first_stage(image_inpaint)
            z_inpaint = self.get_first_stage_encoding(encoder_posterior_inpaint, scale_factor=self.scale_factor).detach()

            mask_resized = F.interpolate(image_mask, size=z.shape[-1], mode="nearest")
            z_image = torch.cat((z, z_inpaint, mask_resized), dim=1)

        if self.use_lidar:
            encoder_posterior = self.encode_first_stage(range_gt, module_name="lidar_stage_model")
            z = self.get_first_stage_encoding(encoder_posterior, scale_factor=self.lidar_scale_factor).detach()

            encoder_posterior_inpaint = self.encode_first_stage(range_inpaint, module_name="lidar_stage_model")
            z_inpaint = self.get_first_stage_encoding(encoder_posterior_inpaint, scale_factor=self.lidar_scale_factor).detach()

            mask_resized = F.interpolate(range_mask, size=z.shape[-1], mode="nearest")
            z_lidar = torch.cat((z, z_inpaint, mask_resized), dim=1)

        return z_image, z_lidar

    def shared_step(self, batch, **kwargs):
        data = self.get_input(batch, self.first_stage_key)
        loss = self(data["z"], data["cond"])
        return loss

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        self.u_cond_prop=random.uniform(0, 1)
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
                    
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))

        if self.u_cond_prop<self.u_cond_percent:
            c = [self.learnable_vector.repeat(x.shape[0],1,1)]
            if "ref_bbox" in self.cond_stage_key:
                c.append(self.bbox_uncond_vector.repeat(x.shape[0],1,1))
            c = torch.cat(c, dim=1)

        return self.p_losses(x, c, t, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        if hasattr(self, "split_input_params"):
            raise NotImplementedError
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids  
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if self.cond_stage_key in ["image", "LR_image", "segmentation",
                                       'bbox_img'] and self.model.conditioning_key:  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert (len(c) == 1)  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == 'coordinates_bbox':
                assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params['original_image_size']
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left positions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
                                        for patch_nr in range(z.shape[-1])]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [(x_tl, y_tl,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device)
                                      for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
                print(patch_limits_tknzd[0].shape)
                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device)

                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])

                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

            else:
                cond_list = [cond for _ in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0],
                                  tuple)  # todo cant deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, x_start, cond, t, noise=None, ):
        if self.first_stage_key == 'inpaint':
            # x_start=x_start[:,:4,:,:]
            noise = default(noise, lambda: torch.randn_like(x_start[:,:4,:,:]))
            x_noisy = self.q_sample(x_start=x_start[:,:4,:,:], t=t, noise=noise)
            x_noisy = torch.cat((x_noisy,x_start[:,4:,:,:]),dim=1)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})


        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self,cond,batch_size,ddim, ddim_steps,**kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,**kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates
    
    @torch.no_grad()
    def decode_sample(self, sample, z_lidar=None):
        h_camera, h_lidar = None, None

        if self.use_camera and self.use_lidar:
            h_camera = sample[::2]
            # h_lidar = F.interpolate(sample[1::2], size=(z_lidar.shape[-2], self.image_size), mode='nearest')
            bottom = (sample[1::2].shape[-2] - z_lidar.shape[-2]) // 2
            top = bottom + z_lidar.shape[-2]
            h_lidar = sample[1::2][:, :, bottom:top, :]

            if self.image_size != z_lidar.shape[-1]:
                # Undo lidar latent crop
                z_lidar[..., z_lidar.shape[-1]//2 - self.image_size//2 :z_lidar.shape[-1]//2 + self.image_size//2] = h_lidar
                h_lidar = z_lidar
        elif self.use_camera:
            h_camera = sample
        else:
            # h_lidar = F.interpolate(sample, size=(z_lidar.shape[-2], self.image_size), mode='nearest')
            bottom = (sample[1::2].shape[-2] - z_lidar.shape[-2]) // 2
            top = bottom + z_lidar.shape[-2]
            h_lidar = sample[:, :, bottom:top, :]

            if self.image_size != z_lidar.shape[-1]:
                # Undo lidar latent crop
                z_lidar[..., z_lidar.shape[-1]//2 - self.image_size//2 :z_lidar.shape[-1]//2 + self.image_size//2] = h_lidar
                h_lidar = z_lidar

        return h_camera, h_lidar


    @torch.no_grad()
    def log_images(self, batch, log_metrics=True, ddim_steps=50, ddim_eta=1., split="train", **kwargs):
        data = self.get_input(
            batch,
            self.first_stage_key,
            force_c_encode=True,
            return_vae_rec=True,
        )

        with self.ema_scope("Plotting"):
            if self.first_stage_key=='inpaint':
                samples, _ = self.sample_log(
                    cond=data["cond"],batch_size=data['z'].shape[0],ddim=(ddim_steps is not None),
                    ddim_steps=ddim_steps,eta=ddim_eta,rest=data['z'][:,4:,:,:]
                )   
                h_camera, h_lidar = self.decode_sample(samples, data.get("z_lidar"))

        log, _ = self.log_data(batch, data, h_camera, h_lidar, log_metrics=log_metrics, split=split)
        return log
    
    @torch.no_grad()
    def log_data(self, batch, data, h_camera, h_lidar, log_metrics=True, return_sample=False, split="train"):
        log = dict()
        lidar_metrics = None
        if self.use_camera:
            image_sample = self.decode_first_stage(h_camera)
            image_sample = torch.clamp(image_sample, -1., 1.)

            sample, input, inpaint_input, reference, rec = get_camera_vis(
                sample=image_sample,
                input=batch["image"]["GT"],
                inpaint_input=batch["image"]["inpaint_image"],
                reference=batch["image"]["cond"]["ref_image"],
                rec=data["image_rec"],
                ref_bboxes=batch["image"]["cond"]["ref_bbox"],
            )

            sample_no_box, input_no_box, inpaint_input_no_box, reference_no_box, rec_no_box = get_camera_vis(
                sample=image_sample,
                input=batch["image"]["GT"],
                inpaint_input=batch["image"]["inpaint_image"],
                reference=batch["image"]["cond"]["ref_image"],
                rec=data["image_rec"],
            )

            log["image_preds"] = torch.cat([input, inpaint_input, reference, sample], dim=-2)
            log["image_preds_no_box"] = torch.cat([input_no_box, inpaint_input_no_box, reference_no_box, sample_no_box], dim=-2)
            log["image_input-rec"] = torch.cat([input, rec], dim=-2)

            if return_sample:
                log["image_sample"] = image_sample

        if self.use_lidar:
            lidar_sample = self.decode_first_stage(h_lidar, module_name="lidar_stage_model")
            lidar_sample = torch.clamp(lidar_sample, -1., 1.)
            
            inpaint_depth = batch["lidar"]["range_data_inpaint"][:, [0]]
            sample_depth = lidar_sample[:, [0]]
            input_depth = batch["lidar"]["range_data"][:, [0]]
            rec_depth = data["lidar_rec"][:, [0]]

            inpaint_int = batch["lidar"]["range_data_inpaint"][:, [1]]
            sample_int = lidar_sample[:, [1]]
            input_int = batch["lidar"]["range_data"][:, [1]]
            rec_int = data["lidar_rec"][:, [1]]

            mask = 1 - batch["lidar"]["range_mask"][:, [0]]
            instance_mask = batch["lidar"]["range_instance_mask"]

            log["range_depth_pred"] = torch.cat([input_depth, inpaint_depth, instance_mask, sample_depth, rec_depth], dim=-2)
            log["range_int_pred"] = torch.cat([input_int, inpaint_int, instance_mask, sample_int, rec_int], dim=-2)

            if return_sample:
                log["range_sample_depth"] = sample_depth
                log["range_sample_int"] = sample_int
                log["range_bbox_mask"] = mask

            if self.range_object_norm:
                # center_depth = batch["lidar"]["center_depth"].view(-1, 1, 1, 1)
                # sample_depth = torch.clamp(torch.atanh(sample_depth) / self.range_object_norm_scale + center_depth, -1, 1)
                # input_depth = torch.clamp(torch.atanh(input_depth) / self.range_object_norm_scale + center_depth, -1, 1)
                # rec_depth = torch.clamp(torch.atanh(rec_depth) / self.range_object_norm_scale + center_depth, -1, 1)

                for i in range(sample_depth.shape[0]):
                    min_depth_obj = batch["lidar"]["min_depth_obj"][i]
                    max_depth_obj = batch["lidar"]["max_depth_obj"][i]
                    sample_depth[i] = inverse_depth_normalization(sample_depth[i], min_depth_obj, max_depth_obj, alpha=self.range_object_norm_scale)
                    input_depth[i] = inverse_depth_normalization(input_depth[i], min_depth_obj, max_depth_obj, alpha=self.range_object_norm_scale)
                    rec_depth[i] = inverse_depth_normalization(rec_depth[i], min_depth_obj, max_depth_obj, alpha=self.range_object_norm_scale)

            if self.range_int_norm:
                sample_int = torch.clamp(-0.5 * torch.log(1 - (sample_int + 1) / 2) - 1, -1, 1)
                input_int = torch.clamp(-0.5 * torch.log(1 - (input_int + 1) / 2) - 1, -1, 1)
                rec_int = torch.clamp(-0.5 * torch.log(1 - (rec_int + 1) / 2) - 1, -1, 1)

            # Compute metrics
            lidar_metrics = {}
            for pred_name,(pred, gt) in {
                "pred_depth": (sample_depth, input_depth),
                "rec_depth": (rec_depth, input_depth),
                "pred_int": (sample_int, input_int),
                "rec_int": (rec_int, input_int),
                }.items():
                for score_name in ["mse", "median_error"]:
                    B = pred.shape[0]
                    object_scores, mask_scores, full_scores = [], [], []
                    for i in range(B):
                        new_size = (32, batch["lidar"]["width_crop"][i].item())
                        pred_ = pool_resize(pred[[i]], new_size)
                        instance_mask_ = pool_resize(instance_mask[[i]], new_size, mode='max_pool')
                        mask_ = pool_resize(mask[[i]], new_size, mode="max_pool")
                        gt_ = pool_resize(gt[[i]], new_size)


                        if score_name == "median_error":
                            object_scores.append(
                                torch.abs(pred_[instance_mask_ == 1] - gt_[instance_mask_ == 1]).median().item()
                            )
                            mask_scores.append(
                                torch.abs(pred_[mask_ == 1] - gt_[mask_ == 1]).median().item()
                            )
                            full_scores.append(
                                torch.abs(pred_ - gt_).median().item()
                            )
                        else:
                            object_scores.append(
                                (((pred_[instance_mask_ == 1] - gt_[instance_mask_ == 1]) ** 2).mean().item() ** 0.5)
                            )
                            mask_scores.append(
                                (((pred_[mask_ == 1] - gt_[mask_ == 1]) ** 2).mean().item() ** 0.5)
                            )
                            full_scores.append(
                                (((pred_ - gt_) ** 2).mean().item() ** 0.5)
                            )

                        if np.isnan(object_scores[-1]):
                            del object_scores[-1]

                    lidar_metrics.update({
                        f"{score_name}/object_{pred_name}" : np.mean(object_scores),
                        f"{score_name}/mask_{pred_name}" : np.mean(mask_scores),
                        # f"{reduction}/full_{pred_name}_L1" : np.mean(full_scores),
                    })

            # Make metrics more interpretable by scaling them to the original range
            lidar_metrics = {f"{split}/{k}": v * ((54 - 1.4) / 2) if "depth" in k else v * 128 for k, v in lidar_metrics.items()}
            if log_metrics:
                self.log_dict(lidar_metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            # Point cloud visualisations
            sample_vis, input_vis, rec_vis = get_lidar_vis(
                sample=sample_depth,
                input=input_depth,
                rec=rec_depth,
                bboxes=batch["bbox_3d"],
                range_depth_orig=batch["lidar"]["range_depth_orig"],
                range_shift_left=batch["lidar"]["range_shift_left"],
                range_pitch=batch["lidar"]["range_pitch"],
                range_yaw=batch["lidar"]["range_yaw"],
                width_crop=batch["lidar"]["width_crop"],
            )

            log["lidar_input-pred-rec"] = torch.cat([input_vis, sample_vis, rec_vis], dim=-2)

        return log, lidar_metrics

    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        param_names = []

        for name, param in self.model.named_parameters():
            if (
                "cond_adapter" in name or
                "lidar" in name or 
                "cross_modal" in name
                # "attn1" in name or
                # "attn2" in name or
                # "norm1" in name or
                # "norm2" in name
            ):
                params.append(param)
                param_names.append(name)
                assert param.requires_grad, f"{name} requires grad is False"

        if self.cond_stage_trainable:
            if "ref_bbox" in self.cond_stage_key:
                print(f"{self.__class__.__name__}: optimizing bbox conditioning params!")
                for name, param in self.cond_stage_model.bbox_embedder.named_parameters():
                    if "class_embedder" not in name:
                        params.append(param)
                        param_names.append(name)
                        assert param.requires_grad, f"{name} requires grad is False"

                params.append(self.bbox_uncond_vector)
                param_names.append("bbox_uncond_vector")
                assert self.bbox_uncond_vector.requires_grad, f"bbox_uncond_vector requires grad is False"
            
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
            param_names.append("logvar")
            assert self.logvar.requires_grad, f"logvar requires grad is False"

        print(f"Optimizing parameters: {param_names}")
        opt = torch.optim.AdamW(params, lr=lr)

        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config).eval()

        for name, param in self.diffusion_model.named_parameters():
            if (
                "cond_adapter" in name or
                "lidar" in name or 
                "cross_modal" in name
                # "attn1" in name or
                # "attn2" in name or
                # "norm1" in name or
                # "norm2" in name
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out

