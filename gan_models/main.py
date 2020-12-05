import os
from argparse import ArgumentParser

import torch
from pathlib2 import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.test_tube import TestTubeLogger
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

from data import CustomDataLoader
from engine import Engine
from utils import save_args


def main(args):
    tt_logger = TestTubeLogger(save_dir=args.log_path, name="",
                               debug=args.debug,
                               description=args.description,
                               create_git_tag=args.git_tag
                               )
    tt_logger.experiment

    log_dir = Path(tt_logger.save_dir) / f"version_{tt_logger.version}"

    checkpoint_dir = log_dir / "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    chkpt_callback = ModelCheckpoint(checkpoint_dir,
                                     #  monitor='val_loss',
                                     #  save_last=True,
                                     #  mode='min',
                                     #  save_top_k=1
                                     )

    # data_loader = CustomDataLoader.from_argparse_args(args)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = MNIST(args.data_dir, download=True, transform=transform)
    test = MNIST(args.data_dir, train=False,
                 download=True, transform=transform)
    train = ConcatDataset([train, test])
    data_loader = DataLoader(train, args.train_batchsize,
                             shuffle=True, num_workers=args.num_workers,
                             pin_memory=True)
    # out_dim -> torch.prod(torch.tensor(transforms.ToTensor()(train[0][0]).shape))
    model = Engine(learning_rate=args.learning_rate)

    save_args(args, log_dir)

    trainer = Trainer.from_argparse_args(args, logger=tt_logger,
                                         checkpoint_callback=chkpt_callback
                                         )

    trainer.fit(model, data_loader)
    # trainer.test(model)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser = CustomDataLoader.add_argparse_args(parser)
    parser = Engine.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    print(f'\nArguments: \n{args}\n')

    main(args)
