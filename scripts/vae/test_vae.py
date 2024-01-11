import sys

sys.path.append("./")

import torch
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from core.vae.vae_pl import VAEPLM
from scripts.data import view
from scripts.data.dm import CelebADataset, EDHDataset


def main():
    config = OmegaConf.load(open("scripts/vae/config.yaml", "r"))
    torch.set_float32_matmul_precision(
        config.optim.float32_matmul_precision
    )

    # data
    test_set = EDHDataset("data\edh", 4)
    test_loader = DataLoader(test_set, 1, True)

    # model
    net = VAEPLM.load_from_checkpoint(
        R"ckp\edh\vae\best.ckpt"
    )

    org = next(iter(test_loader)).to(net.device)
    org = rearrange(org, "b t h w c -> (b t) c h w")
    rec, _ = net(org)

    print(org.shape)
    print(rec.shape)
    fig = view.edh_subplot(
        test_set.lon, test_set.lat, 
        torch.cat((org.detach().cpu(), rec.detach().cpu()), 0).squeeze(1)
        , 2, 4
    )
    fig.savefig("imgs/vae.jpg")

if __name__ == "__main__":
    main()