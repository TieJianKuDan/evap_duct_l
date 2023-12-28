from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from ..mod.cuboid_transformer.cuboid_transformer_unet import \
    CuboidTransformerUNet
from ..utils.optim import warmup_lambda
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
