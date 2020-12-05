import os
from argparse import ArgumentParser

import torch
from torch._C import device, dtype
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy
from torch.optim import Adam
from torchvision.utils import make_grid

# import all GAN models here.
from models import GAN
from utils import str2bool

# Use this in main.py!
MODELS = {
    'gan': GAN,
    'vanilla_gan': GAN
}

LOSSES = {'bce': F.binary_cross_entropy,
          'bce_logits': F.binary_cross_entropy_with_logits,
          'cross_entropy': F.cross_entropy, 'nll_loss': F.nll_loss,
          'kl_div': F.kl_div, 'mse': F.mse_loss,
          'l1_loss': F.l1_loss}

# model = 'gan'
# model = os.getenv('MODEL')

# ToDo: Implement a Factory for parenting different models.


class Engine(GAN):

    def __init__(self, out_dim=784, latent_dim=10, learning_rate=0.0001,
                 loss_func='bce'):
        super().__init__(latent_dim, out_dim)

        self.save_hyperparameters()
        self.loss_func = LOSSES[loss_func]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('-des', '--description', required=False, type=str)
        parser.add_argument('-lp', '--log_path', type=str,
                            default='./lightning_logs')
        parser.add_argument('-lr', '--learning_rate',
                            type=float, default=0.0001)
        parser.add_argument('-c', '--criterion', type=str,
                            choices=LOSSES.keys(),
                            default='bce')
        parser.add_argument('-m', '--model', default='gan',
                            required=False, type=str)
        parser.add_argument('-gt', '--git_tag', required=False, const=False,
                            type=str2bool, nargs='?')
        parser.add_argument('--debug', required=False, const=False, nargs='?',
                            type=str2bool)
        return parser

    def configure_optimizers(self):
        d_optim = Adam(self.discriminator.parameters(),
                       lr=self.hparams.learning_rate)
        g_optim = Adam(self.parameters(),
                       lr=self.hparams.learning_rate)
        return [d_optim, g_optim]

    def discriminator_loss(self, X):
        # Real Loss
        # Here y_hat -> Prediction whether real or fake
        y_hat = self.discriminator(X)
        y = torch.ones_like(y_hat)
        real_loss = self.loss_func(y_hat, y)

        real_acc = accuracy(torch.round(y_hat), y)

        # Fake_loss
        latent_vector = torch.randn(
            X.shape[0], self.hparams.latent_dim, device=self.device)
        self.X_hat = self(latent_vector)
        y_hat = self.discriminator(self.X_hat)
        y = torch.zeros_like(y_hat)
        fake_loss = self.loss_func(y_hat, y)
        fake_acc = accuracy(torch.round(y_hat), y, num_classes=2)

        d_loss = real_loss + fake_loss

        acc = real_acc + fake_acc

        return d_loss, acc

    def generator_loss(self, X):
        # Fake Loss
        latent_vector = torch.randn(
            X.shape[0], self.hparams.latent_dim, device=self.device)
        self.X_hat = self(latent_vector)
        y_hat = self.discriminator(self.X_hat)
        y = torch.ones_like(y_hat)
        g_loss = self.loss_func(y_hat, y)
        acc = accuracy(torch.round(y_hat), y, num_classes=2)

        return g_loss, acc

    def training_step(self, batch, batch_idx, optimizer_idx):
        X, _ = batch
        X = torch.flatten(X, start_dim=1)
        if optimizer_idx == 0:
            loss, acc = self.discriminator_loss(X)
            logs = {"Loss/d_loss": loss, "Accuracy/d_acc": acc}
        if optimizer_idx == 1:
            loss, acc = self.generator_loss(X)
            logs = {"Loss/g_loss": loss, "Accuracy/g_acc": acc}

        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def on_train_epoch_end(self, outputs) -> None:

        self.logger.experiment.add_image(
            'Generated Images', make_grid(self.X_hat), self.current_epoch
        )
