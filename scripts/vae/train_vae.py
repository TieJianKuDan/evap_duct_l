from matplotlib import pyplot as plt
from omegaconf import OmegaConf, dictconfig
from pytorch_lightning import Trainer

from scripts.vae.vae_pl import VAEPLM, CelebaPLDM

oc_from_file = OmegaConf.load(open("config.yaml", "r"))
data_config = oc_from_file.data
model_config = oc_from_file.model
loss_config = oc_from_file.loss
opt_config = oc_from_file.opt

dm = CelebaPLDM(data_config, opt_config.batch_size)
dm.setup()

total_num_steps = \
    dm.train_sample_num * opt_config.max_epochs / opt_config.batch_size

net = VAEPLM(model_config, loss_config, opt_config, total_num_steps)

trainer = Trainer(max_epochs=opt_config.max_epochs)

trainer.fit(net, dm)