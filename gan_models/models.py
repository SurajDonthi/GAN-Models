import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils import filtered_kwargs


class Generator(nn.Module):
    def __init__(self, out_dim: int, latent_dim: int = 100,
                 #  normalize: bool = False,
                 hidden_layers: list = [128, 256, 512, 1024]):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_dim = th.prod(th.tensor(out_dim)).item()
        self._original_out_dim = out_dim
        self.hidden_layers = hidden_layers

        self.layers = [latent_dim] + \
            self.hidden_layers + [self.out_dim]
        # self.normalize = normalize
        self._model_generator()

    def _model_generator(self):
        input_dim = self.latent_dim
        for i, dim in enumerate(self.hidden_layers):
            name = f'linear{i}'
            if i == 0:
                layer = self.linear_block(input_dim, dim, normalize=False)
            else:
                layer = self.linear_block(input_dim, dim)

            setattr(self, name, nn.Sequential(*layer))
            input_dim = dim

        self.final = nn.Sequential(
            nn.Linear(self.hidden_layers[-1], self.out_dim),
            nn.Tanh()
        )

    def linear_block(self, input_feat, out_feat, normalize=True):
        layers = [nn.Linear(input_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, eps=0.8))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        return layers

    def forward(self, X):
        for i, _ in enumerate(self.hidden_layers):
            name = f'linear{i}'
            linear = getattr(self, name)
            X = linear(X)

        X = self.final(X)

        return X.view(-1, *self._original_out_dim)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_layers=[512, 256]):
        super().__init__()
        self.input_dim = th.prod(th.tensor(input_dim)).item()
        self.hidden_layers = hidden_layers
        self.layers = [self.input_dim] + self.hidden_layers + [1]
        self._model_generator()

    def _model_generator(self):
        input_dim = self.input_dim
        for i, dim in enumerate(self.hidden_layers):
            name = f'linear{i}'
            layer = self.linear_block(input_dim, dim)
            setattr(self, name, nn.Sequential(*layer))
            input_dim = dim

        self.final = nn.Sequential(nn.Linear(self.hidden_layers[-1], 1),
                                   nn.Sigmoid()
                                   )

    def linear_block(self, input_feat, out_feat):
        layers = [nn.Linear(input_feat, out_feat)]
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        return layers

    def forward(self, X):
        X = th.flatten(X, start_dim=1)
        for i, _ in enumerate(self.hidden_layers):
            name = f'linear{i}'
            linear = getattr(self, name)
            X = linear(X)

        X = self.final(X)

        return X


class GAN:
    def __init__(self, latent_dim, out_dim, criterion, model_args=None):

        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.criterion = criterion

        self.D = Discriminator(input_dim=out_dim, **
                               model_args['discriminator'])
        self.G = Generator(out_dim, latent_dim, ** model_args['generator'])

    # Make the below two functions independent of the class

    def _init_device(self, device):
        self.device = device

    def D_loss(self, batch):
        real, _ = batch
        # Real Loss
        # Here preds -> Prediction whether real or fake
        preds = self.D(real)

        valid = th.ones_like(preds)
        real_loss = self.criterion(preds, valid)

        # real_acc = accuracy(th.round(preds), y)

        # Fake_loss
        z = th.randn(
            real.shape[0], self.latent_dim, device=self.device)
        preds = self.D(self.G(z))

        fake = th.zeros_like(preds)
        fake_loss = self.criterion(preds, fake)
        # fake_acc = accuracy(th.round(preds), y, num_classes=2)

        loss = (real_loss + fake_loss) / 2

        # acc = real_acc + fake_acc

        return loss   # , acc

    def G_loss(self, batch):
        real, _ = batch
        # Fake Loss
        z = th.randn(real.shape[0], self.latent_dim, device=self.device)
        self.gen_imgs = self.G(z)
        preds = self.D(self.gen_imgs)

        valid = th.ones_like(preds, requires_grad=False)
        loss = self.criterion(preds, valid)
        # acc = accuracy(th.round(preds), y, num_classes=2)

        return loss  # , acc


# TODO: Implement the below models with similar interface!


class FMGAN:
    """
    Feature Matching for training the Vanilla GAN model.
    """

    def __init__(self) -> None:
        self.discriminator = None
        self.generator = None


class DCGAN:
    def __init__(self) -> None:
        self.discriminator = None
        self.generator = None


class WassersteinGAN:

    def __init__(self) -> None:
        self.discriminator = None
        self.generator = None


class CGAN:
    def __init__(self) -> None:
        self.discriminator = None
        self.generator = None


class InfoGAN:
    def __init__(self) -> None:
        self.discriminator = None
        self.generator = None


class CycleGAN:
    def __init__(self) -> None:
        self.discriminator = None
        self.generator = None


class BigGAN:
    def __init__(self) -> None:
        self.discriminator = None
        self.generator = None
