import inspect
from argparse import ArgumentParser, Namespace
from typing import Any, List, Optional, Tuple, Union, Literal
import json

import pytorch_lightning as pl
import torch as th
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.utilities import parsing
from torch.optim import RMSprop, SGD, Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import make_grid

# import all GAN models here.
from models import VanillaGAN1D, FMGAN2D, DCGAN, CGAN, WassersteinGAN, CramerGAN
from utils import filtered_kwargs

MODELS = {
    'gan': VanillaGAN1D,
    'vanilla_gan': VanillaGAN1D,
    'dcgan': DCGAN,
    'feature_matching': FMGAN2D,
    'cgan': CGAN,
    'wgan': WassersteinGAN,
    'cramer_gan': CramerGAN
}

OPTIMIZERS = {
    'sgd': SGD,
    'adam': Adam,
    'rmsprop': RMSprop
}

SCHEDULERS = {
    'steplr': StepLR
}


class Engine(pl.LightningModule):

    def __init__(self, out_dim: int, latent_dim: int = 100,
                 model: Literal[tuple(MODELS.keys())] = 'gan',
                 learning_rate: float = 0.0002,
                 d_skip_batch: int = 1, g_skip_batch: int = 1,
                 optimizer_options: Optional[dict] = {
                     'optim': 'adam', 'args': {'betas': (0.5, 0.999)}},
                 scheduler_options: Optional[dict] = None,
                 model_args: Optional[dict] = None
                 ):

        super().__init__()

        self.save_hyperparameters()
        self.lr = learning_rate
        self.d_skip_batch, self.g_skip_batch = d_skip_batch, g_skip_batch
        self._optimizer_options = optimizer_options
        self._scheduler_options = scheduler_options

        model_args = self.generalize_args(model_args)
        self.model = MODELS[model](
            latent_dim, out_dim, **model_args)
        self.G, self.D = self.model.G, self.model.D
        self.G_loss, self.D_loss = self.model.G_loss, self.model.D_loss

    def generalize_args(self, kwargs):
        if (kwargs is None):
            kwargs = {'generator': {}, 'discriminator': {}}
        # <= --> Check whether subset
        if not ({'generator', 'discriminator'} <= kwargs.keys()):
            kwargs = {**kwargs, 'generator': kwargs, 'discriminator': kwargs}
        return kwargs

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

    @classmethod
    def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs):
        """
        Create an instance from CLI arguments.

        Args:
            args: The parser or namespace to take arguments from. Only known arguments will be
             parsed and passed to the :class:`LightningDataModule`.
            **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
             These must be valid DataModule arguments.

        Example::

            parser = ArgumentParser(add_help=False)
            parser = LightningModule.add_argparse_args(parser)
            module = LightningModule.from_argparse_args(args)

        """
        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        # we only want to pass in valid DataModule args, the rest may be user specific
        valid_kwargs = inspect.signature(cls.__init__).parameters
        datamodule_kwargs = dict(
            (name, params[name]) for name in valid_kwargs if name in params
        )
        datamodule_kwargs.update(**kwargs)

        return cls(**datamodule_kwargs)

    @classmethod
    def get_init_arguments_and_types(cls) -> List[Tuple[str, Tuple, Any]]:
        r"""Scans the DataModule signature and returns argument names, types and default values.
        Returns:
            List with tuples of 3 values:
            (argument name, set with argument types, argument default value).
        """
        datamodule_default_params = inspect.signature(cls.__init__).parameters
        name_type_default = []
        for arg in datamodule_default_params:
            arg_type = datamodule_default_params[arg].annotation
            arg_default = datamodule_default_params[arg].default
            try:
                arg_types = tuple(arg_type.__args__)
            except AttributeError:
                arg_types = (arg_type,)

            name_type_default.append((arg, arg_types, arg_default))

        return name_type_default

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

        self.logger.experiment.add_image(
            'Generated Images', make_grid(self.G(z)),
            self.current_epoch
        )

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
