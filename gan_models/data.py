import inspect
import json
from argparse import ArgumentParser
from typing import Literal

import pytorch_lightning as pl
from pathlib2 import Path
from pytorch_lightning.utilities import parsing
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, MNIST, Caltech101, FashionMNIST, CelebA
from torchvision.transforms import transforms

# from utils import filtered_kwargs


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


class TorchDataLoader(pl.LightningDataModule):

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

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        r"""Extends existing argparse by default `LightningDataModule` attributes.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        added_args = [x.dest for x in parser._actions]

        blacklist = ["kwargs"]
        depr_arg_names = blacklist + added_args
        depr_arg_names = set(depr_arg_names)

        allowed_types = (str, int, float, bool, dict)

        for arg, arg_types, arg_default in (
            at
            for at in cls.get_init_arguments_and_types()
            if at[0] not in depr_arg_names
        ):
            arg_types = [at for at in arg_types if (
                at in allowed_types) or isinstance(at, str)]
            if not arg_types:
                # skip argument with not supported type
                continue

            arg_kwargs = {}
            arg_choices = None
            if dict in arg_types:
                use_type = json.loads
            elif bool in arg_types:
                arg_kwargs.update(nargs="?", const=True)
                # if the only arg type is bool
                if len(arg_types) == 1:
                    use_type = parsing.str_to_bool
                # if only two args (str, bool)
                elif len(arg_types) == 2 and set(arg_types) == {str, bool}:
                    use_type = parsing.str_to_bool_or_str
                else:
                    # filter out the bool as we need to use more general
                    use_type = [at for at in arg_types if at is not bool][0]
            elif all(isinstance(at, str) for at in arg_types):
                use_type = str
                arg_choices = arg_types
            else:
                use_type = arg_types[0]

            if arg_default == inspect._empty:
                arg_default = None

            parser.add_argument(
                f"--{arg}",
                dest=arg,
                default=arg_default,
                # required=True if not arg_default else False,
                type=use_type,
                choices=arg_choices,
                help=f"autogenerated by plb.{cls.__name__}",
                **arg_kwargs,
            )

        return parser
