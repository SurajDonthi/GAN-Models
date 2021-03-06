import os
from argparse import ArgumentParser

import torch as th
from pathlib2 import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

from gan_models.data import TorchDataLoader
from gan_models.engine import Engine
from gan_models.utils import save_args


def main(args):
    tt_logger = TestTubeLogger(save_dir=args.log_path, name="",
                               debug=args.debug,
                               description=args.description,
                               create_git_tag=args.git_tag,
                               log_graph=True
                               )
    tt_logger.experiment

    log_dir = Path(tt_logger.save_dir) / f"version_{tt_logger.version}"

    checkpoint_dir = log_dir / "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    chkpt_callback = ModelCheckpoint(checkpoint_dir,
                                     monitor='Loss/d_loss_epoch',
                                     save_last=True,
                                     mode='auto',
                                     save_top_k=5,
                                     period=5
                                     )

    data_loader = TorchDataLoader.from_argparse_args(args)

    img_shape = data_loader.train_data[0][0].shape

    model = Engine.from_argparse_args(args, img_shape=img_shape)

    save_args(args, log_dir)

    trainer = Trainer.from_argparse_args(args, logger=tt_logger,
                                         checkpoint_callback=chkpt_callback
                                         )

    trainer.fit(model, data_loader)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser = TorchDataLoader.add_argparse_args(parser)
    parser = Engine.add_argparse_args(parser)
    parser = Engine.add_additional_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    print(f'\nArguments: \n{args}\n')

    main(args)
