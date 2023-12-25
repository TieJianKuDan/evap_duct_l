import os

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


root_dir = os.getcwd()

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

    def __init__(self, data_config, batch_size, seed):
        super(CelebaPLDM, self).__init__()
        self.data_dir = root_dir + "\datasets\img_align_celeba"
        self.val_ratio = data_config.val_ratio
        self.test_ratio = data_config.test_ratio
        self.num_workers = data_config.num_workers
        self.persistent_workers = data_config.persistent_workers
        self.batch_size = batch_size
        self.seed = seed

    @property
    def train_sample_num(self):
        return len(self.train_set)
    
    @property
    def val_sample_num(self):
        return len(self.val_set)
    
    @property
    def test_sample_num(self):
        return len(self.test_set)

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
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )