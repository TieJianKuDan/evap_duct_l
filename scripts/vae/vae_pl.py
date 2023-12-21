import os

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from core.utils.optim import warmup_lambda
from core.vae import AutoencoderKL, LPIPSWithDiscriminator


class CelebADataset(Dataset):

    def __init__(self, root, img_shape=(64, 64)) -> None:
        super().__init__()
        self.root = root
        self.img_shape = img_shape
        self.filenames = sorted(os.listdir(root))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        path = os.path.join(self.root, self.filenames[index])
        img = Image.open(path).convert('RGB')
        pipeline = transforms.Compose([
            transforms.CenterCrop(168),
            transforms.Resize(self.img_shape),
            transforms.ToTensor()
        ])
        return pipeline(img)


class CelebaPLDM(pl.LightningDataModule):

    def __init__(self, data_config, batch_size):
        self.data_dir = \
            R"D:\Projects\atmosphere\evap_duct_l\datasets\img_align_celeba"
        self.val_ratio = data_config.val_ratio
        self.test_ratio = data_config.test_ratio
        self.batch_size = batch_size
        self.seed = data_config.seed

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        datasets = CelebADataset(self.data_dir)
        self.train_set, self.val_set, self.test_set = \
            random_split(
                datasets, 
                [
                    1 - self.val_ratio - self.test_ratio, 
                    self.val_ratio, 
                    self.test_ratio
                ], 
                generator=torch.Generator().manual_seed(self.seed)
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, 
            batch_size=self.batch_size, 
            shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            shuffle=True
        )

    @property
    def train_sample_num(self):
        return len(self.train_set)
    
    @property
    def val_sample_num(self):
        return len(self.val_set)
    
    @property
    def test_sample_num(self):
        return len(self.test_set)

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
            "gen_loss", 
            gen_loss, 
            prog_bar=True, 
            logger=True,
            on_step=True, 
            on_epoch=True, 
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
            "dis_loss", 
            dis_loss, 
            prog_bar=True, 
            logger=True,
            on_step=True, 
            on_epoch=True, 
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
