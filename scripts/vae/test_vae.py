import sys

sys.path.append("./")

import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from omegaconf import OmegaConf

from core.vae.vae_pl import VAEPLM
from scripts.data.dm import CelebADataset


def main():
    config = OmegaConf.load(open("scripts/vae/config.yaml", "r"))
    torch.set_float32_matmul_precision(
        config.optim.float32_matmul_precision
    )

    # data
    test_set = CelebADataset("data\img_align_celeba")
    test_loader = DataLoader(test_set, 8, True)

    # model
    net = VAEPLM.load_from_checkpoint(R"ckp\vae\best.ckpt")

    org = next(iter(test_loader)).cuda()
    rec, _ = net(org)

    decoder = net.model.decoder
    z = torch.randn((8, 64, 8, 8)).cuda()*10
    gen = decoder(z)

    fig, axs = plt.subplots(3, 8)

    for i in range(8):
        axs[0, i].imshow(
            org[i].detach().permute(1, 2, 0).cpu())
        axs[0, i].axis("off")
        axs[1, i].imshow(
            rec[i].detach().permute(1, 2, 0).cpu())
        axs[1, i].axis("off")
        axs[2, i].imshow(
            gen[i].detach().permute(1, 2, 0).cpu())
        axs[2, i].axis("off")
    
    fig.tight_layout()
    fig.savefig("test.jpg")

if __name__ == "__main__":
    main()