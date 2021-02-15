
from typing import Literal

from pathlib2 import Path
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import (CIFAR10, MNIST, Caltech101, CelebA,
                                  FashionMNIST)
from torchvision.transforms import transforms

from gan_models.base import BaseDataModule

# from gan_models.utils import filtered_kwargs


class SampleDataset(Dataset):

    def __init__(self):
        super(SampleDataset, self).__init__()

    def __len__(self):
        return len(None)

    def __getitem__(self, index):
        sample = None
        return sample


DATASETS = {
    'mnist': MNIST,
    'fashion-mnist': FashionMNIST,
    'caltech101': Caltech101,
    'cifar10': CIFAR10,
    'celeba': CelebA
}
MEAN_N_STD = {
    'mnist': {'mean': (0.1307,), 'std': (0.3081,)},
    'fasion-mnist': {'mean': (0.5,), 'std': (0.5,)},
    'caltech101': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
    'cifar10': {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)}
}


class TorchDataLoader(BaseDataModule):

    def __init__(self, data_dir: str,
                 dataset: Literal[tuple(DATASETS.keys())] = 'mnist',
                 train: bool = True,
                 download: bool = True,
                 train_batchsize: int = 32,
                 num_workers: int = 4,
                 ):

        super().__init__()

        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise Exception(
                f"'Path '{str(self.data_dir)}' does not exist!")
        if not self.data_dir.is_dir():
            raise Exception(
                f"Path '{str(self.data_dir)}' is not a directory!")

        self.dataset = dataset
        self.train = train
        self.download = download

        self.train_batchsize = train_batchsize

        self.num_workers = num_workers
        self.prepare_data()

    def prepare_data(self):

        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(**MEAN_N_STD[self.dataset])
        ])

        if self.dataset == 'caltech101':
            self.train_data = DATASETS[self.dataset](
                root=self.data_dir,
                target_type='category',
                transform=self.train_transforms,
                download=self.download,
            )
        else:
            self.train_data = DATASETS[self.dataset](
                root=self.data_dir,
                train=self.train,
                transform=self.train_transforms,
                download=self.download,
            )

    def train_dataloader(self):

        return DataLoader(self.train_data, batch_size=self.train_batchsize,
                          shuffle=True, num_workers=self.num_workers)
