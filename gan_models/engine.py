from argparse import ArgumentParser
from typing import Literal, Optional

import torch as th
# import torch.nn.functional as F
# from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.utilities import parsing
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import make_grid, save_image

from gan_models.base import BaseModule
# import all GAN models here.
from gan_models.models import MODELS

# from gan_models.utils import filtered_kwargs

OPTIMIZERS = {
    'sgd': SGD,
    'adam': Adam,
    'rmsprop': RMSprop
}

SCHEDULERS = {
    'steplr': StepLR
}


class Engine(BaseModule):

    def __init__(self, img_shape: int, latent_dim: int = 100,
                 model: Literal[tuple(MODELS.keys())] = 'gan',
                 learning_rate: float = 0.0002,
                 d_skip_batch: int = 1, g_skip_batch: int = 1,
                 optimizer_options: Optional[dict] = {
                     'optim': 'adam', 'args': {'betas': (0.5, 0.999)}},
                 scheduler_options: Optional[dict] = None,
                 model_args: Optional[dict] = None,
                 save_gen_imgs: bool = True
                 ):

        super().__init__()

        self.save_hyperparameters()
        self.lr = learning_rate
        self.d_skip_batch, self.g_skip_batch = d_skip_batch, g_skip_batch
        self._optimizer_options = optimizer_options
        self._scheduler_options = scheduler_options

        model_args = self.generalize_args(model_args)
        self.model = MODELS[model](
            latent_dim=latent_dim, img_shape=img_shape, **model_args)
        self.G, self.D = self.model.G, self.model.D
        self.G_loss, self.D_loss = self.model.G_loss, self.model.D_loss

    @staticmethod
    def add_additional_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('-lp', '--log_path', type=str,
                            default='./logs')
        parser.add_argument('-des', '--description', required=False, type=str)
        parser.add_argument('-gt', '--git_tag', required=False, const=False,
                            type=parsing.str_to_bool, nargs='?')
        parser.add_argument('--debug', required=False, const=False, nargs='?',
                            type=parsing.str_to_bool)
        return parser

    def configure_optimizers(self):
        Optim = OPTIMIZERS[self._optimizer_options['optim']]
        g_optim = Optim(self.G.parameters(),
                        lr=self.lr,
                        **self._optimizer_options['args'])
        d_optim = Optim(self.D.parameters(),
                        lr=self.lr,
                        **self._optimizer_options['args'])

        if self._scheduler_options:
            scheduler = SCHEDULERS[self._scheduler_options['method']]
            g_scheduler = scheduler(g_optim, **self._scheduler_options['args'])
            d_scheduler = scheduler(d_optim, **self._scheduler_options['args'])

            return [g_optim, d_optim], [g_scheduler, d_scheduler]

        return [g_optim, d_optim]

    def forward(self, X):
        return self.G(X)

    def on_fit_start(self):
        self.model._init_device(self.device)

    def training_step(self, batch, batch_idx, optimizer_idx):

        if optimizer_idx == 0 and batch_idx % self.hparams.g_skip_batch == 0:
            loss = self.G_loss(batch)        # , acc
            logs = {"Loss/g_loss": loss,
                    # "Accuracy/g_acc": acc
                    }
            self.log_dict(logs, prog_bar=True,
                          on_step=True, on_epoch=True)

            return loss

        if optimizer_idx == 1 and batch_idx % self.hparams.d_skip_batch == 0:
            loss = self.D_loss(batch)        # , acc
            logs = {"Loss/d_loss": loss,
                    # "Accuracy/d_acc": acc
                    }
            self.log_dict(logs, prog_bar=True,
                          on_step=True, on_epoch=True)

            return loss

    def on_train_epoch_end(self, outputs) -> None:

        z = th.randn(64, self.hparams.latent_dim, device=self.device)
        if any(arg in self.G.forward.__code__.co_varnames
               for arg in ['labels', 'y', 'label', 'Y']):
            gen_labels = th.randint(10, (64,), device=self.device)
            gen_imgs = self.G(z, gen_labels)
        else:
            gen_imgs = self.G(z)

        self.logger.experiment.add_image(
            'Generated Images', make_grid(gen_imgs),
            self.current_epoch)

        if self.hparams.save_gen_imgs:
            save_image(gen_imgs,
                       self.trainer.logger.save_dir +
                       f'/version_{self.trainer.logger.version}/media' +
                       f'/epoch_{self.current_epoch}.png'
                       )
