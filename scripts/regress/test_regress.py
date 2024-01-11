import sys

sys.path.append("./")

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from omegaconf import OmegaConf

from core.unet.unet_pl import UnetPL
from scripts.data.dm import ERA5Dataset
from scripts.data.view import edh_subplot


def main():
    config = OmegaConf.load(open("scripts/unet/config.yaml", "r"))
    torch.set_float32_matmul_precision(
        config.optim.float32_matmul_precision
    )

    # data
    test_set = ERA5Dataset(R"data\edh\2021", R"data\era5\2021")
    test_loader = DataLoader(test_set, 8, True)

    # model
    net = UnetPL.load_from_checkpoint(R"ckp\edh\regress\best.ckpt")

    era5, edh = next(iter(test_loader))
    era5 = era5.to(net.device)
    edh = edh.to(net.device)
    edh_hat = net(era5)
    loss = mse_loss(edh_hat, edh, reduction="mean")
    print(loss)
    
    fig = edh_subplot(
        test_set.lon,
        test_set.lat,
        torch.cat(
            (
                edh.detach().cpu().squeeze(1), 
                edh_hat.detach().cpu().squeeze(1)
            ),
            0
        ),
        2,
        8
    )
    fig.savefig("imgs/regress.jpg")

if __name__ == "__main__":
    main()