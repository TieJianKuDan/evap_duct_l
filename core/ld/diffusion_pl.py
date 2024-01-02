from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from ..mod.cuboid_transformer.cuboid_transformer_unet import \
    CuboidTransformerUNet
from ..utils.layout import layout_to_in_out_slice
from ..utils.optim import warmup_lambda
from ..vae.vae_pl import VAEPLM
from .latent_diffusion import LatentDiffusion
from .utils import extract_into_tensor, make_beta_schedule


class CTUnetPL(pl.LightningModule):

    def __init__(self, model_config, optim_config, total_num_steps) -> None:
        super(CTUnetPL, self).__init__()
        self.time_steps = model_config.time_steps
        num_blocks = len(model_config.depth)
        if isinstance(model_config.self_pattern, str):
            block_attn_patterns = [model_config.self_pattern] * num_blocks
        self.model = CuboidTransformerUNet(
            input_shape=model_config["input_shape"],
            target_shape=model_config["target_shape"],
            base_units=model_config["base_units"],
            scale_alpha=model_config["scale_alpha"],
            num_heads=model_config["num_heads"],
            attn_drop=model_config["attn_drop"],
            proj_drop=model_config["proj_drop"],
            ffn_drop=model_config["ffn_drop"],
            # inter-attn downsample/upsample
            downsample=model_config["downsample"],
            downsample_type=model_config["downsample_type"],
            upsample_type=model_config["upsample_type"],
            upsample_kernel_size=model_config["upsample_kernel_size"],
            # attention
            depth=model_config["depth"],
            block_attn_patterns=block_attn_patterns,
            # global vectors
            num_global_vectors=model_config["num_global_vectors"],
            use_global_vector_ffn=model_config["use_global_vector_ffn"],
            use_global_self_attn=model_config["use_global_self_attn"],
            separate_global_qkv=model_config["separate_global_qkv"],
            global_dim_ratio=model_config["global_dim_ratio"],
            # misc
            ffn_activation=model_config["ffn_activation"],
            gated_ffn=model_config["gated_ffn"],
            norm_layer=model_config["norm_layer"],
            padding_type=model_config["padding_type"],
            checkpoint_level=model_config["checkpoint_level"],
            pos_embed_type=model_config["pos_embed_type"],
            use_relative_pos=model_config["use_relative_pos"],
            self_attn_use_final_proj=model_config["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=model_config["attn_linear_init_mode"],
            ffn_linear_init_mode=model_config["ffn_linear_init_mode"],
            ffn2_linear_init_mode=model_config["ffn2_linear_init_mode"],
            attn_proj_linear_init_mode=model_config["attn_proj_linear_init_mode"],
            conv_init_mode=model_config["conv_init_mode"],
            down_linear_init_mode=model_config["down_up_linear_init_mode"],
            up_linear_init_mode=model_config["down_up_linear_init_mode"],
            global_proj_linear_init_mode=model_config["global_proj_linear_init_mode"],
            norm_init_mode=model_config["norm_init_mode"],
            # timestep embedding for diffusion
            time_embed_channels_mult=model_config["time_embed_channels_mult"],
            time_embed_use_scale_shift_norm=model_config["time_embed_use_scale_shift_norm"],
            time_embed_dropout=model_config["time_embed_dropout"],
            unet_res_connect=model_config["unet_res_connect"],
        )

        self.register_schedule(
            model_config.given_betas,
            beta_schedule=model_config.beta_schedule,
            timesteps=model_config.time_steps,
            linear_start=model_config.linear_start,
            linear_end=model_config.linear_end,
            cosine_s=model_config.cosine_s,
        )

        self.batch_axis = 0
        self.optim_config = optim_config
        self.total_num_steps = total_num_steps

        self.save_hyperparameters()

    def register_schedule(
            self,
            given_betas=None, 
            beta_schedule="linear", 
            timesteps=1000,
            linear_start=1e-4, 
            linear_end=2e-2, 
            cosine_s=8e-3
    ):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule, 
                timesteps, 
                linear_start=linear_start, 
                linear_end=linear_end,
                cosine_s=cosine_s
            )
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_cumprod', 
            to_torch(np.sqrt(alphas_cumprod))
        )
        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod', 
            to_torch(np.sqrt(1. - alphas_cumprod))
        )

    def extract_into_tensor(self, a, t, x_shape):
        return extract_into_tensor(a=a, t=t, x_shape=x_shape,
                                   batch_axis=self.batch_axis)

    def q_sample(self, x_start, t, noise):
        return (
            self.extract_into_tensor(
                self.sqrt_alphas_cumprod, 
                t, 
                x_start.shape
            ) * 
            x_start +
            self.extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, 
                t, 
                x_start.shape
            ) * 
            noise
        )

    def i_q_sample(self, xt, t, noise):
        return (
            (
                xt - self.extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, 
                t, 
                xt.shape
                ) * 
                noise
            ) / 
            self.extract_into_tensor(
                self.sqrt_alphas_cumprod, 
                t, 
                xt.shape
            )
        )

    def forward(self, x, t):
        return self.model(x, t, None)

    def training_step(self, batch, batch_idx):
        x0 = batch.permute(0, 2, 3, 1).unsqueeze(1)
        batch_size = x0.shape[self.batch_axis]
        t = torch.randint(
            0, 
            self.num_timesteps, 
            (batch_size,), 
            device=self.device
        ).long()
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        noise_hat = self(xt, t)
        loss = mse_loss(noise_hat, noise, reduction="mean")
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0 = batch.permute(0, 2, 3, 1).unsqueeze(1)
        batch_size = x0.shape[self.batch_axis]
        t = torch.randint(
            0, 
            self.num_timesteps, 
            (batch_size,), 
            device=self.device
        ).long()
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        noise_hat = self(xt, t)
        loss = mse_loss(noise_hat, noise, reduction="mean")
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x0 = batch.permute(0, 2, 3, 1).unsqueeze(1)
        batch_size = x0.shape[self.batch_axis]
        t = torch.randint(
            0, 
            self.num_timesteps, 
            (batch_size,), 
            device=self.device
        ).long()
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        noise_hat = self(xt, t)
        loss = mse_loss(noise_hat, noise, reduction="mean")
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        lr = self.optim_config.lr
        betas = self.optim_config.betas
        opt = torch.optim.Adam(
            params=self.model.parameters(),
            lr=lr, 
            betas=betas
        )

        warmup_iter = int(
            np.round(
                self.optim_config.warmup_percentage * self.total_num_steps)
        )
        if self.optim_config.lr_scheduler_mode == 'none':
            return opt
        else:
            if self.optim_config.lr_scheduler_mode == 'cosine':
                # generator
                warmup_scheduler = LambdaLR(
                    opt,
                    lr_lambda=warmup_lambda(
                        warmup_steps=warmup_iter,
                        min_lr_ratio=self.optim_config.warmup_min_lr_ratio
                    )
                )
                cosine_scheduler = CosineAnnealingLR(
                    opt,
                    T_max=(self.total_num_steps - warmup_iter),
                    eta_min=self.optim_config.min_lr_ratio * self.optim_config.lr
                )
                lr_scheduler = SequentialLR(
                    opt,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_iter]
                )
                lr_scheduler_config = {
                    'scheduler': lr_scheduler,
                    'interval': 'step',
                    'frequency': 1, 
                }
            else:
                raise NotImplementedError
            return {
                    "optimizer": opt, 
                    "lr_scheduler": lr_scheduler_config
            }

class EDHLatentDiffusion(LatentDiffusion):
    
    def __init__(
            self, model_config, optim_config, 
            total_num_steps,
    ):
        self.total_num_steps = total_num_steps

        self.vae = model_config.vae
        pretrained_ckpt_path = self.vae["pretrained_ckpt_path"]
        first_stage_model = VAEPLM.load_from_checkpoint(pretrained_ckpt_path).model

        self.latent = model_config.latent_model
        num_blocks = len(self.latent.depth)
        assert type(self.latent.self_pattern) == str
        block_attn_patterns = [self.latent["self_pattern"]] * num_blocks
        latent_model = CuboidTransformerUNet(
            input_shape=self.latent["input_shape"],
            target_shape=self.latent["target_shape"],
            base_units=self.latent["base_units"],
            scale_alpha=self.latent["scale_alpha"],
            num_heads=self.latent["num_heads"],
            attn_drop=self.latent["attn_drop"],
            proj_drop=self.latent["proj_drop"],
            ffn_drop=self.latent["ffn_drop"],
            # inter-attn downsample/upsample
            downsample=self.latent["downsample"],
            downsample_type=self.latent["downsample_type"],
            upsample_type=self.latent["upsample_type"],
            upsample_kernel_size=self.latent["upsample_kernel_size"],
            # attention
            depth=self.latent["depth"],
            block_attn_patterns=block_attn_patterns,
            # global vectors
            num_global_vectors=self.latent["num_global_vectors"],
            use_global_vector_ffn=self.latent["use_global_vector_ffn"],
            use_global_self_attn=self.latent["use_global_self_attn"],
            separate_global_qkv=self.latent["separate_global_qkv"],
            global_dim_ratio=self.latent["global_dim_ratio"],
            # misc
            ffn_activation=self.latent["ffn_activation"],
            gated_ffn=self.latent["gated_ffn"],
            norm_layer=self.latent["norm_layer"],
            padding_type=self.latent["padding_type"],
            checkpoint_level=self.latent["checkpoint_level"],
            pos_embed_type=self.latent["pos_embed_type"],
            use_relative_pos=self.latent["use_relative_pos"],
            self_attn_use_final_proj=self.latent["self_attn_use_final_proj"],
            # initialization
            attn_linear_init_mode=self.latent["attn_linear_init_mode"],
            ffn_linear_init_mode=self.latent["ffn_linear_init_mode"],
            ffn2_linear_init_mode=self.latent["ffn2_linear_init_mode"],
            attn_proj_linear_init_mode=self.latent["attn_proj_linear_init_mode"],
            conv_init_mode=self.latent["conv_init_mode"],
            down_linear_init_mode=self.latent["down_up_linear_init_mode"],
            up_linear_init_mode=self.latent["down_up_linear_init_mode"],
            global_proj_linear_init_mode=self.latent["global_proj_linear_init_mode"],
            norm_init_mode=self.latent["norm_init_mode"],
            # timestep embedding for diffusion
            time_embed_channels_mult=self.latent["time_embed_channels_mult"],
            time_embed_use_scale_shift_norm=self.latent["time_embed_use_scale_shift_norm"],
            time_embed_dropout=self.latent["time_embed_dropout"],
            unet_res_connect=self.latent["unet_res_connect"], 
        )

        self.diffusion = model_config.diffusion
        self.layout = self.diffusion.layout
        self.in_len = self.diffusion.in_len
        self.out_len = self.diffusion.out_len
        self.batch_axis = 0
        self.optim_config = optim_config
        self.total_num_steps = total_num_steps
        super(EDHLatentDiffusion, self).__init__(
            torch_nn_module=latent_model,
            layout=self.diffusion.layout,
            data_shape=self.diffusion["data_shape"],
            timesteps=self.diffusion["timesteps"],
            beta_schedule=self.diffusion["beta_schedule"],
            loss_type=self.optim_config.loss_type,
            use_ema=self.diffusion["use_ema"],
            log_every_t=self.diffusion["log_every_t"],
            clip_denoised=self.diffusion["clip_denoised"],
            linear_start=self.diffusion["linear_start"],
            linear_end=self.diffusion["linear_end"],
            cosine_s=self.diffusion["cosine_s"],
            given_betas=self.diffusion["given_betas"],
            original_elbo_weight=self.diffusion["original_elbo_weight"],
            v_posterior=self.diffusion["v_posterior"],
            l_simple_weight=self.diffusion["l_simple_weight"],
            parameterization=self.diffusion["parameterization"],
            learn_logvar=self.diffusion["learn_logvar"],
            logvar_init=self.diffusion["logvar_init"],
            # latent diffusion
            latent_shape=self.diffusion["latent_shape"],
            first_stage_model=first_stage_model,
            cond_stage_model=self.diffusion["cond_stage_model"],
            num_timesteps_cond=self.diffusion["num_timesteps_cond"],
            cond_stage_trainable=self.diffusion["cond_stage_trainable"],
            cond_stage_forward=self.diffusion["cond_stage_forward"],
            scale_by_std=self.diffusion["scale_by_std"],
            scale_factor=self.diffusion["scale_factor"], 
        )

        self.save_hyperparameters()

    @property
    def in_slice(self):
        if not hasattr(self, "_in_slice"):
            in_slice, out_slice = layout_to_in_out_slice(
                layout=self.layout,
                in_len=self.in_len,
                out_len=self.out_len)
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._in_slice

    @property
    def out_slice(self):
        if not hasattr(self, "_out_slice"):
            in_slice, out_slice = layout_to_in_out_slice(
                layout=self.oc.layout.layout,
                in_len=self.oc.layout.in_len,
                out_len=self.oc.layout.out_len)
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._out_slice

    @torch.no_grad()
    def get_input(self, batch, **kwargs):
        r"""
        dataset dependent
        re-implement it for each specific dataset

        Parameters
        ----------
        batch:  Any
            raw data batch from specific dataloader

        Returns
        -------
        out:    Sequence[torch.Tensor, Dict[str, Any]]
            out[0] should be a torch.Tensor which is the target to generate
            out[1] should be a dict consists of several key-value pairs for conditioning
        """
        return self._get_input_edh(
            batch=batch, 
            return_verbose=kwargs.get("return_verbose", False)
        )

    @torch.no_grad()
    def _get_input_edh(self, batch, return_verbose=False):
        seq = batch
        in_seq = seq[self.in_slice]
        out_seq = seq[self.out_slice].contiguous()
        if return_verbose:
            return out_seq, {"y": in_seq}, in_seq
        else:
            return out_seq, {"y": in_seq}
    
    def configure_optimizers(self):
        lr = self.optim_config.lr
        betas = self.optim_config.betas
        opt = torch.optim.Adam(
            params=self.torch_nn_module.parameters(),
            lr=lr, 
            betas=betas
        )

        warmup_iter = int(
            np.round(
                self.optim_config.warmup_percentage * self.total_num_steps)
        )
        if self.optim_config.lr_scheduler_mode == 'none':
            return opt
        else:
            if self.optim_config.lr_scheduler_mode == 'cosine':
                # generator
                warmup_scheduler = LambdaLR(
                    opt,
                    lr_lambda=warmup_lambda(
                        warmup_steps=warmup_iter,
                        min_lr_ratio=self.optim_config.warmup_min_lr_ratio
                    )
                )
                cosine_scheduler = CosineAnnealingLR(
                    opt,
                    T_max=(self.total_num_steps - warmup_iter),
                    eta_min=self.optim_config.min_lr_ratio * self.optim_config.lr
                )
                lr_scheduler = SequentialLR(
                    opt,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_iter]
                )
                lr_scheduler_config = {
                    'scheduler': lr_scheduler,
                    'interval': 'step',
                    'frequency': 1, 
                }
            else:
                raise NotImplementedError
            return {
                    "optimizer": opt, 
                    "lr_scheduler": lr_scheduler_config
            }