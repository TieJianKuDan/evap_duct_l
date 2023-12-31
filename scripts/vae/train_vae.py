import sys
sys.path.append("./")

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, callbacks, loggers, seed_everything

from core.vae.vae_pl import VAEPLM
from scripts.data.dm import EDHPLDM


def main():
    config = OmegaConf.load(open("scripts/vae/config.yaml", "r"))
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
    net = VAEPLM(config.model, config.loss, config.optim, total_num_steps)

    # trainer
    trainer = Trainer(
        max_epochs=config.optim.max_epochs,
        accelerator=config.optim.accelerator,
        logger=[
            loggers.TensorBoardLogger(
                "./logs/vae/",
                name="edh",
            )
        ],
        precision=config.optim.precision,
        enable_checkpointing=True,
        callbacks=[
            callbacks.ModelCheckpoint(
                dirpath="ckp/edh/vae",
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
        ckpt_path=R"ckp\edh\vae\best.ckpt"
    )

if __name__ == "__main__":
    main()