import sys

sys.path.append("./")

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from core.ld.diffusion_pl import EDHLatentDiffusion
from scripts.data.dm import EDHDataset
from scripts.data import view


def main():
    config = OmegaConf.load(open("scripts/vae/config.yaml", "r"))
    torch.set_float32_matmul_precision(
        config.optim.float32_matmul_precision
    )

    # data
    test_set = EDHDataset("data\edh", 15)
    test_loader = DataLoader(test_set, 1, True)

    # model
    net = EDHLatentDiffusion.load_from_checkpoint(
        R"ckp\edh\ld\best.ckpt"
    )

    batch = next(iter(test_loader)).to(net.device)
    out_seq, in_seq = net.get_input(batch)
    print((out_seq.shape, in_seq["y"].shape))
    pred_seq = net.sample(in_seq, 1)
    print(pred_seq.shape)
    fig = view.edh_subplot(
        test_set.lon, test_set.lat,
        torch.cat((
            in_seq["y"][:, 1:],
            out_seq, 
            pred_seq
            ), 1
        ).squeeze().cpu(),
        3,
        6
    )
    fig.savefig("imgs/ld.jpg")
    

if __name__ == "__main__":
    main()