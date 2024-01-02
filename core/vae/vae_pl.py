import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from core.utils.optim import warmup_lambda

from . import AutoencoderKL, LPIPSWithDiscriminator


class VAEPLM(pl.LightningModule):
    
    def __init__(
            self, 
            model_config, 
            loss_config, 
            opt_config, 
            total_num_steps
        ) -> None:
        super(VAEPLM, self).__init__()

        self.model = AutoencoderKL(
            in_channels=model_config.in_channels,                
            out_channels=model_config.out_channels,
            block_out_channels=model_config.block_out_channels,
            layers_per_block=model_config.layers_per_block,
            down_block_types=model_config.down_block_types,
            up_block_types=model_config.up_block_types,
            act_fn=model_config.act_fu,
            latent_channels=model_config.latent_channels,
            norm_num_groups=model_config.norm_num_groups
        )

        self.loss = LPIPSWithDiscriminator(
            disc_start=loss_config.disc_start,
            kl_weight=loss_config.kl_weight,
            disc_weight=loss_config.disc_weight,
            perceptual_weight=loss_config.perceptual_weight,
            disc_in_channels=loss_config.disc_in_channels)

        self.opt_config = opt_config
        self.total_num_steps = total_num_steps
        self.automatic_optimization = False
        self.save_hyperparameters()

    def forward(self, x, sample_posterior=True):
        return self.model(
            x, 
            sample_posterior, 
            return_posterior=True
        )

    def get_last_layer(self):
        return self.model.decoder.conv_out.weight

    def training_step(self, batch, batch_idx):
        opt_ae, opt_disc = self.optimizers()
        lr_ae, lr_disc = self.lr_schedulers()

        batch = rearrange(batch, "b t h w c -> (b t) c h w")
        rec, posterior = self(batch)
        gen_loss, log = self.loss(
            batch, 
            rec, 
            posterior, 
            0, 
            self.global_step, 
            last_layer=self.get_last_layer()
        )
        opt_ae.zero_grad()
        self.manual_backward(gen_loss)
        opt_ae.step()
        lr_ae.step()
        self.log(
            "train_gen", 
            gen_loss, 
            prog_bar=True, 
            logger=True,
            on_step=True, 
            on_epoch=False, 
            sync_dist=False
        )
        self.log_dict(
            log, 
            prog_bar=False, 
            logger=True,
            on_step=True, 
            on_epoch=False, 
            sync_dist=False
        )

        dis_loss, log = self.loss(
            batch, 
            rec, 
            posterior, 
            1, 
            self.global_step, 
            last_layer=self.get_last_layer()
        )
        opt_disc.zero_grad()
        self.manual_backward(dis_loss)
        opt_disc.step()
        lr_disc.step()
        self.log(
            "train_dis", 
            dis_loss, 
            prog_bar=True, 
            logger=True,
            on_step=True, 
            on_epoch=False, 
            sync_dist=False
        )
        self.log_dict(
            log, 
            prog_bar=False, 
            logger=True,
            on_step=True, 
            on_epoch=False, 
            sync_dist=False
        )

    def validation_step(self, batch, batch_idx):
        batch = rearrange(batch, "b t h w c -> (b t) c h w")
        rec, posterior = self(batch)
        gen_loss, log = self.loss(
            batch, 
            rec, 
            posterior, 
            0, 
            self.global_step, 
            last_layer=self.get_last_layer(),
            split="val"
        )
        self.log(
            "val_gen", 
            gen_loss, 
            prog_bar=True, 
            logger=True,
            on_step=False, 
            on_epoch=True, 
            sync_dist=False
        )
        self.log_dict(
            log, 
            prog_bar=False, 
            logger=True,
            on_step=False, 
            on_epoch=True, 
            sync_dist=False
        )

        dis_loss, log = self.loss(
            batch, 
            rec, 
            posterior, 
            1, 
            self.global_step, 
            last_layer=self.get_last_layer(),
            split="val"
        )
        self.log(
            "val_dis", 
            dis_loss, 
            prog_bar=True, 
            logger=True,
            on_step=False, 
            on_epoch=True, 
            sync_dist=False
        )
        self.log_dict(
            log, 
            prog_bar=False, 
            logger=True,
            on_step=False, 
            on_epoch=True, 
            sync_dist=False
        )

    def test_step(self, batch, batch_idx):
        batch = rearrange(batch, "b t h w c -> (b t) c h w")
        rec, posterior = self(batch)
        gen_loss, log = self.loss(
            batch, 
            rec, 
            posterior, 
            0, 
            self.global_step, 
            last_layer=self.get_last_layer(),
            split="val"
        )
        self.log(
            "gen_loss", 
            gen_loss, 
            prog_bar=True, 
            logger=True,
            on_step=False, 
            on_epoch=True, 
            sync_dist=False
        )
        self.log_dict(
            log, 
            prog_bar=False, 
            logger=True,
            on_step=False, 
            on_epoch=True, 
            sync_dist=False
        )

        dis_loss, log = self.loss(
            batch, 
            rec, 
            posterior, 
            1, 
            self.global_step, 
            last_layer=self.get_last_layer(),
            split="val"
        )
        self.log(
            "dis_loss", 
            dis_loss, 
            prog_bar=True, 
            logger=True,
            on_step=False, 
            on_epoch=True, 
            sync_dist=False
        )
        self.log_dict(
            log, 
            prog_bar=False, 
            logger=True,
            on_step=False, 
            on_epoch=True, 
            sync_dist=False
        )

    def configure_optimizers(self):
        optim_cfg = self.opt_config
        lr = optim_cfg.lr
        betas = optim_cfg.betas
        opt_ae = torch.optim.Adam(
            list(self.model.encoder.parameters()) +
                list(self.model.decoder.parameters()) +
                list(self.model.quant_conv.parameters()) +
                list(self.model.post_quant_conv.parameters()),
            lr=lr, 
            betas=betas
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr, 
            betas=betas
        )

        warmup_iter = int(
            np.round(optim_cfg.warmup_percentage * self.total_num_steps)
        )
        if optim_cfg.lr_scheduler_mode == 'none':
            return [{"optimizer": opt_ae}, {"optimizer": opt_disc}]
        else:
            if optim_cfg.lr_scheduler_mode == 'cosine':
                # generator
                warmup_scheduler_ae = LambdaLR(
                    opt_ae,
                    lr_lambda=warmup_lambda(
                        warmup_steps=warmup_iter,
                        min_lr_ratio=optim_cfg.warmup_min_lr_ratio
                    )
                )
                cosine_scheduler_ae = CosineAnnealingLR(
                    opt_ae,
                    T_max=(self.total_num_steps - warmup_iter),
                    eta_min=optim_cfg.min_lr_ratio * optim_cfg.lr)
                lr_scheduler_ae = SequentialLR(
                    opt_ae,
                    schedulers=[warmup_scheduler_ae, cosine_scheduler_ae],
                    milestones=[warmup_iter])
                lr_scheduler_config_ae = {
                    'scheduler': lr_scheduler_ae,
                    'interval': 'step',
                    'frequency': 1, }
                # discriminator
                warmup_scheduler_disc = LambdaLR(
                    opt_disc,
                    lr_lambda=warmup_lambda(
                        warmup_steps=warmup_iter,
                        min_lr_ratio=optim_cfg.warmup_min_lr_ratio
                    )
                )
                cosine_scheduler_disc = CosineAnnealingLR(
                    opt_disc,
                    T_max=(self.total_num_steps - warmup_iter),
                    eta_min=optim_cfg.min_lr_ratio * optim_cfg.lr
                )
                lr_scheduler_disc = SequentialLR(
                    opt_disc,
                    schedulers=[warmup_scheduler_disc, cosine_scheduler_disc],
                    milestones=[warmup_iter]
                )
                lr_scheduler_config_disc = {
                    'scheduler': lr_scheduler_disc,
                    'interval': 'step',
                    'frequency': 1, 
                }
            else:
                raise NotImplementedError
            return [
                {
                    "optimizer": opt_ae, 
                    "lr_scheduler": lr_scheduler_config_ae
                },
                {
                    "optimizer": opt_disc, 
                    "lr_scheduler": lr_scheduler_config_disc
                },
            ]
