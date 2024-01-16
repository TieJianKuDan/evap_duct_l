import gc
import os

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.transforms import Normalize

root_dir = os.getcwd()

def normlize(data:np.ndarray):
    # handle nan
    data_ = data[~np.isnan(data)]
    mean = data_.mean()
    std = data_.std()
    data = np.nan_to_num(data, nan=mean)
    return (data - mean) / std

class CelebADataset(Dataset):

    def __init__(self, root, img_shape=(32, 32)) -> None:
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
        self.data_dir = root_dir + "\data\img_align_celeba"
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
    
class EDHDataset(Dataset):

    def __init__(self, root, seq_len) -> None:
        super().__init__()
        _edh = []
        for root, _, files in os.walk(root):  
            for filename in files:
                _edh.append(
                    xr.open_dataset(os.path.join(root, filename))
                )
        _edh = sorted(
            _edh, key=lambda _edh: _edh.time.data[0]
        )
        self.lon = _edh[0].longitude.data[:-1]
        self.lat = _edh[0].latitude.data[:-1]
        edh = None
        time = None
        for i in range(len(_edh)):
            if edh is None:
                edh = _edh[i].EDH.data[:, :-1, :-1]
            else:
                edh = np.concatenate(
                    (edh, _edh[i].EDH.data[:, :-1, :-1]),
                    axis=0
                )
            if time is None:
                time = _edh[i].time.data
            else:
                time = np.concatenate(
                    (time, _edh[i].time.data),
                    axis=0
                )
        edh = np.expand_dims(edh, -1)

        self.edh_seq = []
        self.time_seq = []
        self.seq_len = seq_len
        for i in range(edh.shape[0] - self.seq_len + 1):
            self.edh_seq.append(edh[i : i + self.seq_len])
            self.time_seq.append(time[i : i + self.seq_len])

    def __len__(self) -> int:
        return len(self.time_seq)

    def __getitem__(self, index: int):
        return self.edh_seq[index]
    
class EDHPLDM(pl.LightningDataModule):

    def __init__(self, data_config, batch_size, seed):
        super(EDHPLDM, self).__init__()
        self.data_dir = root_dir + "\data\edh"
        self.val_ratio = data_config.val_ratio
        self.test_ratio = data_config.test_ratio
        self.num_workers = data_config.num_workers
        self.persistent_workers = data_config.persistent_workers
        self.seq_len = data_config.seq_len
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
        datasets = EDHDataset(self.data_dir, self.seq_len)
        self.lon = datasets.lon
        self.lat = datasets.lat
        self.time = datasets.time_seq
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
            # num_workers=self.num_workers,
            # persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, 
            batch_size=self.batch_size, 
            shuffle=False,
            # num_workers=self.num_workers,
            # persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            shuffle=False,
            # num_workers=self.num_workers,
            # persistent_workers=self.persistent_workers
        )
    
class ERA5Dataset(Dataset):

    def __init__(self, edh, era5) -> None:
        super().__init__()
        _edhs = []
        for root, _, files in os.walk(edh):  
            for filename in files:
                _edhs.append(
                    xr.open_dataset(os.path.join(root, filename))
                )
        _edhs = sorted(
            _edhs, key=lambda _edh: _edh.time.data[0]
        )
        self.lon = _edhs[0].longitude.data[:-1]
        self.lat = _edhs[0].latitude.data[:-1]
        self.edh = None
        self.time = None
        for _edh in _edhs:
            if self.edh is None:
                self.edh = _edh.EDH.data[:, :-1, :-1]
            else:
                self.edh = np.concatenate((
                    self.edh, _edh.EDH.data[:, :-1, :-1]), 0)
            if self.time is None:
                self.time = _edh.time.data
            else:
                self.time = np.concatenate((
                    self.time, _edh.time.data), 0)

        _era5s = []
        for root, _, files in os.walk(era5):  
            for filename in files:
                _era5s.append(
                    xr.open_dataset(os.path.join(root, filename))
                )
        _era5s = sorted(
            _era5s, key=lambda _era5: _era5.time.data[0]
        )
        self.u10 = None
        self.v10 = None
        self.t2m = None
        self.msl = None
        self.sst = None
        self.q2m = None
        for _era5 in _era5s:
            if self.u10 is None:
                self.u10 = _era5.u10.data[:, :-1, :-1]
            else:
                self.u10 = np.concatenate((
                    self.u10, _era5.u10.data[:, :-1, :-1]), 0)
            
            if self.v10 is None:
                self.v10 = _era5.v10.data[:, :-1, :-1]
            else:
                self.v10 = np.concatenate((
                    self.v10, _era5.v10.data[:, :-1, :-1]), 0) 
                
            if self.t2m is None:
                self.t2m = _era5.t2m.data[:, :-1, :-1]
            else:
                self.t2m = np.concatenate((
                    self.t2m, _era5.t2m.data[:, :-1, :-1]), 0) 
                
            if self.msl is None:
                self.msl = _era5.msl.data[:, :-1, :-1]
            else:
                self.msl = np.concatenate((
                    self.msl, _era5.msl.data[:, :-1, :-1]), 0) 

            if self.sst is None:
                self.sst = _era5.sst.data[:, :-1, :-1]
            else:
                self.sst = np.concatenate((
                    self.sst, _era5.sst.data[:, :-1, :-1]), 0) 

            if self.q2m is None:
                self.q2m = _era5.q2m.data[:, :-1, :-1]
            else:
                self.q2m = np.concatenate((
                    self.q2m, _era5.q2m.data[:, :-1, :-1]), 0) 
        self.u10 = normlize(self.u10)
        self.v10 = normlize(self.v10)
        self.t2m = normlize(self.t2m)
        self.msl = normlize(self.msl)
        self.sst = normlize(self.sst)
        self.q2m = normlize(self.q2m)

    def __len__(self) -> int:
        return len(self.time)

    def __getitem__(self, index: int):
        return (
            np.array(
                [
                    self.u10[index], self.v10[index], self.t2m[index],
                    self.msl[index], self.sst[index], self.q2m[index]
                ]
            ),
            np.expand_dims(self.edh[index], axis=0)
        )
    
class ERA5PLDM(pl.LightningDataModule):

    def __init__(self, data_config, batch_size, seed):
        super(ERA5PLDM, self).__init__()
        self.edh_dir = root_dir + "/data/train/edh/"
        self.era5_dir = root_dir + "/data/train/era5/"
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
        datasets = ERA5Dataset(edh=self.edh_dir, era5=self.era5_dir)
        self.lon = datasets.lon
        self.lat = datasets.lat
        self.time = datasets.time
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