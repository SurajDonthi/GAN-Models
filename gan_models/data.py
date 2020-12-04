from argparse import ArgumentParser
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):

    def __init__(self):
        super(Dataset, self).__init__()

    def __len__(self):
        return len(None)

    def __getitem__(self, index):
        sample = None
        return sample


class CustomDataLoader(pl.LightningDataModule):

    def __init__(self, data_dir,
                 train_batchsize=32,
                 val_batchsize=32,
                 test_batchsize=32,
                 train_transforms=None,
                 val_transforms=None,
                 test_transforms=None):

        super().__init__()

        self.data_dir = data_dir

        self.train_batchsize = train_batchsize
        self.test_batchsize = test_batchsize
        self.val_batchsize = val_batchsize

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

    def prepare_data(self, *args, **kwargs):
        return super().prepare_data(*args, **kwargs)

    def setup(self, stage=None):
        return super().setup(stage=stage)

    def train_dataloader(self):
        dataset = Dataset()
        # ToDo: Remove hardcoding
        return DataLoader(dataset, batch_size=self.train_batchsize,
                          shuffle=True, num_workers=4)

    def val_dataloader(self):
        dataset = Dataset()
        # ToDo: Remove hardcoding
        return DataLoader(dataset, batch_size=self.train_batchsize,
                          shuffle=True, num_workers=4)

    def test_dataloader(self):
        dataset = Dataset()
        # ToDo: Remove hardcoding
        return DataLoader(dataset, batch_size=self.train_batchsize,
                          shuffle=True, num_workers=4)
