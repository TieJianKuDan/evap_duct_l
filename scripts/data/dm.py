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
        # load data
        _edh = []
        for root, _, files in os.walk(root):  
            for filename in files:
                _edh.append(
                    xr.open_dataset(os.path.join(root, filename)).EDH
                )
        _edh = sorted(
            _edh, key=lambda _edh: _edh.time.data[0]
        )
        self.lon = _edh[0].longitude.data[:-1]
        self.lat = _edh[0].latitude.data[:-1]
        edh = [None] * len(_edh)
        time = [None] * len(_edh)
        for i in range(len(edh)):
            edh[i] = _edh[i].data[:, :-1, :-1]
            time[i] = _edh[i].time.data

        edh = np.concatenate(edh, axis=0)
        edh = np.expand_dims(edh, -1)
        time = np.concatenate(time, axis=0)

        # slide window
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
        # load edh(time, lat, lon)
        _edh = []
        for root, _, files in os.walk(edh):  
            for filename in files:
                _edh.append(
                    xr.load_dataset(os.path.join(root, filename)).EDH
                )
        _edh = sorted(
            _edh, key=lambda _edh: _edh.time.data[0]
        )
        self.lon = _edh[0].longitude.data[:-1]
        self.lat = _edh[0].latitude.data[:-1]
        self.edh, self.time = self.__handle_edh(_edh)
        gc.collect()

        # load era5=[time, lat, lon]
        _era5 = []
        for root, _, files in os.walk(era5):  
            for filename in files:
                _era5.append(
                    xr.load_dataset(os.path.join(root, filename))
                )
        _era5 = sorted(
            _era5, key=lambda _era5: _era5.time.data[0]
        )
        self.u10 = self.__handle_u10(_era5)
        gc.collect()
        self.v10 = self.__handle_v10(_era5)
        gc.collect()
        self.t2m = self.__handle_t2m(_era5)
        gc.collect()
        self.msl = self.__handle_msl(_era5)
        gc.collect()
        self.sst = self.__handle_sst(_era5)
        gc.collect()
        self.q2m = self.__handle_q2m(_era5)
        gc.collect()

    def __len__(self) -> int:
        return len(self.time)

    def __handle_edh(self, _edh:list):
        edh = [None] * len(_edh)
        time = [None] * len(_edh)
        for i in range(len(_edh)):
            edh[i] = _edh[i].data[:, :-1, :-1]
            time[i] = _edh[i].time.data
        edh = np.concatenate(edh, axis=0)
        time = np.concatenate(time, axis=0)
        return (edh, time)

    def __handle_u10(self, _era5:list):
        u10 = [None] * len(_era5)
        for i in range(len(_era5)):
            u10[i] = _era5[i].u10.data[:, :-1, :-1]
        u10 = np.concatenate(u10, axis=0)
        # delete NaN and Normalize
        u10 = normlize(u10)
        return u10

    def __handle_v10(self, _era5:list):
        v10 = [None] * len(_era5)
        for i in range(len(_era5)):
            v10[i] = _era5[i].v10.data[:, :-1, :-1]
        v10 = np.concatenate(v10, axis=0)
        # delete NaN and Normalize
        v10 = normlize(v10)
        return v10

    def __handle_t2m(self, _era5:list):
        t2m = [None] * len(_era5)
        for i in range(len(_era5)):
            t2m[i] = _era5[i].t2m.data[:, :-1, :-1]
        t2m = np.concatenate(t2m, axis=0)
        # delete NaN and Normalize
        t2m = normlize(t2m)
        return t2m

    def __handle_msl(self, _era5:list):
        msl = [None] * len(_era5)
        for i in range(len(_era5)):
            msl[i] = _era5[i].msl.data[:, :-1, :-1]
        msl = np.concatenate(msl, axis=0)
        # delete NaN and Normalize
        msl = normlize(msl)
        return msl

    def __handle_sst(self, _era5:list):
        sst = [None] * len(_era5)
        for i in range(len(_era5)):
            sst[i] = _era5[i].sst.data[:, :-1, :-1]
        sst = np.concatenate(sst, axis=0)
        # delete NaN and Normalize
        sst = normlize(sst)
        return sst

    def __handle_q2m(self, _era5:list):
        q2m = [None] * len(_era5)
        for i in range(len(_era5)):
            q2m[i] = _era5[i].q2m.data[:, :-1, :-1]
        q2m = np.concatenate(q2m, axis=0)
        # delete NaN and Normalize
        q2m = normlize(q2m)
        return q2m

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