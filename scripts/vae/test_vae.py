import sys

sys.path.append("./")

import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from omegaconf import OmegaConf

from core.vae.vae_pl import VAEPLM
from scripts.data.dm import CelebADataset, EDHDataset
from scripts.data import view


def main():
    config = OmegaConf.load(open("scripts/vae/config.yaml", "r"))
    torch.set_float32_matmul_precision(
        config.optim.float32_matmul_precision
    )

    # data
    test_set = EDHDataset("data\edh")
    test_loader = DataLoader(test_set, 8, True)

    # model
    net = VAEPLM.load_from_checkpoint(
        R"ckp\edh\vae\best.ckpt"
    )

    org = next(iter(test_loader)).cuda()
    rec, _ = net(org)

    print(org.shape)
    print(rec.shape)
    fig = view.edh_subplot(
        test_set.lon, test_set.lat, 
        torch.cat((org.detach().cpu(), rec.detach().cpu()), 0).squeeze(1)
        , 2, 8
    )
    fig.savefig("imgs/tmp.jpg")

if __name__ == "__main__":
    main()