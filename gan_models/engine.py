from argparse import ArgumentParser
import pytorch_lightning as pl

from .model import Model


class Engine(Model, pl.LightningModule):

    def __init__(self):
        super().__init__()

    @staticmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument()

        return parser

    def configure_optimizers(self):
        optim = None
        scheduler = None
        return optim, scheduler

    def training_step(self, X):

        loss = None
        self.log(loss, progress_bar=True)

    def validation_step(self, X):

        loss = None
        self.log(loss, progress_bar=True)

    def test_step(self, X):

        loss = None
        self.log(loss, progress_bar=True)
