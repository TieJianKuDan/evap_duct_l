import sys

sys.path.append("./")

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from omegaconf import OmegaConf

from core.ld.diffusion_pl import CTUnetPL
from scripts.data.dm import CelebADataset


def main():
    config = OmegaConf.load(open("scripts/ld/config.yaml", "r"))
    torch.set_float32_matmul_precision(
        config.optim.float32_matmul_precision
    )

    # data
    test_set = CelebADataset("data/img_align_celeba")
    test_loader = DataLoader(test_set, 8, True)

    # model
    net = CTUnetPL.load_from_checkpoint("ckp/ld/best.ckpt")

    x0 = next(iter(test_loader)).to(net.device)
    x0 = x0.permute(0, 2, 3, 1).unsqueeze(1)
    batch_size = x0.shape[net.batch_axis]
    t = torch.randint(
        0, 
        net.num_timesteps, 
        (batch_size,), 
        device=net.device
    ).long()
    noise = torch.randn_like(x0)
    xt = net.q_sample(x0, t, noise)
    noise_hat = net(xt, t)
    loss = mse_loss(noise_hat, noise, reduction="mean")
    print(loss)
    x0_hat = net.i_q_sample(xt, t, noise_hat)
    x0.squeeze_(1)
    xt.squeeze_(1)
    x0_hat.squeeze_(1)

    fig, axs = plt.subplots(3, 8)

    for i in range(8):
        axs[0, i].imshow(
            x0[i].detach().cpu())
        axs[0, i].axis("off")
        axs[1, i].imshow(
            xt[i].detach().cpu())
        axs[1, i].axis("off")
        axs[2, i].imshow(
            x0_hat[i].detach().cpu())
        axs[2, i].axis("off")
    
    fig.tight_layout()
    fig.savefig("test.jpg")

if __name__ == "__main__":
    main()