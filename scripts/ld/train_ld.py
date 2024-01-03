import sys

sys.path.append("./")

import torch
from lightning_fabric import seed_everything
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, callbacks, loggers

from core.ld.diffusion_pl import EDHLatentDiffusion
from scripts.data.dm import EDHPLDM


def main():
    config = OmegaConf.load(open("scripts/ld/config.yaml", "r"))
    seed_everything(
        seed=config.seed,
        workers=True
    )
    torch.set_float32_matmul_precision(
        config.optim.float32_matmul_precision
    )

    # data
    dm = EDHPLDM(
        config.dataset, 
        config.optim.batch_size,
        config.seed
    )
    dm.setup()

    total_num_steps = \
        dm.train_sample_num * config.optim.max_epochs / config.optim.batch_size

    # model
    net = EDHLatentDiffusion(config.model, config.optim, total_num_steps)

    # trainer
    trainer = Trainer(
        max_epochs=config.optim.max_epochs,
        accelerator=config.optim.accelerator,
        logger=[
            loggers.TensorBoardLogger(
                "./logs/ld/",
                name="edh",
            )
        ],
        precision=config.optim.precision,
        enable_checkpointing=True,
        callbacks=[
            callbacks.ModelCheckpoint(
                dirpath="ckp/edh/ld",
                monitor=config.optim.monitor
            ),
            callbacks.EarlyStopping(
                monitor=config.optim.monitor,
                patience=config.optim.patience,
            )
        ],
        deterministic="warn",
    )

    trainer.fit(
        net, dm,
        ckpt_path="ck/edh/ld/best.ckpt"
    )

if __name__ == "__main__":
    main()