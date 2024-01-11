import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from ..utils.optim import warmup_lambda
from .unet_model import UNet


class UnetPL(pl.LightningModule):

    def __init__(
            self, 
            model_config,
            optim_config, 
            total_num_steps
        ) -> None:
        super(UnetPL, self).__init__()

        self.optim_config = optim_config
        self.total_num_steps = total_num_steps
        self.model = UNet(
            model_config.in_channels, 
            model_config.out_channels, 
            model_config.bilinear
        )
        self.loss = MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        era5, edh = batch
        edh_hat = self(era5)
        train_loss = self.loss(edh, edh_hat)
        self.log(
            "train_loss", train_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True    
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        era5, edh = batch
        edh_hat = self(era5)
        train_loss = self.loss(edh, edh_hat)
        self.log(
            "val_loss", train_loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True    
        )

    def test_step(self, batch, batch_idx):
        era5, edh = batch
        edh_hat = self(era5)
        train_loss = self.loss(edh, edh_hat)
        self.log(
            "test_loss", train_loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True    
        )

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