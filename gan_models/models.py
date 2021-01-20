import torch as th
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from utils import filtered_kwargs

LOSSES = {
    'bce': F.binary_cross_entropy,
    'bce_logits': F.binary_cross_entropy_with_logits,
    'cross_entropy': F.cross_entropy, 'nll_loss': F.nll_loss,
    'kl_div': F.kl_div, 'mse': F.mse_loss,
    'l1_loss': F.l1_loss
}


class GANModels(ABC):

    class Generator(nn.Module):
        def __init__(self, img_shape, latent_dim, **kwargs):
            super().__init__()
            self.img_shape = img_shape
            self.latent_dim = latent_dim

    class Discriminator(nn.Module):
        def __init__(self, img_shape, **kwargs):
            super().__init__()
            self.img_shape = img_shape

    def __init__(self, latent_dim, img_shape, **model_args) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.G = self.Generator(img_shape, latent_dim,
                                **filtered_kwargs(self.Generator.__init__,
                                                  **model_args['generator']))
        self.D = self.Discriminator(img_shape,
                                    **filtered_kwargs(self.Discriminator.__init__,
                                                      **model_args['discriminator']))

    def _init_device(self, device):
        self.device = device

    @abstractmethod
    def D_loss(self, batch):
        loss = None
        return loss

    @abstractmethod
    def G_loss(self, batch):
        loss = None
        return loss


class VanillaGAN(GANModels):

    class Generator(GANModels.Generator):
        def __init__(self, img_shape: int, latent_dim: int = 100,
                     #  normalize: bool = False,
                     hidden_layers: list = [128, 256, 512, 1024]):
            super().__init__(img_shape, latent_dim)

            self.flattened_img_shape = th.prod(th.tensor(img_shape)).item()
            self.hidden_layers = hidden_layers

            self.layers = [latent_dim] + \
                self.hidden_layers + [self.flattened_img_shape]
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
                nn.Linear(self.hidden_layers[-1], self.flattened_img_shape),
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

            return X.view(-1, *self.img_shape)

    class Discriminator(GANModels.Discriminator):
        def __init__(self, img_shape, hidden_layers=[512, 256]):
            super().__init__(img_shape)
            self.flattened_img_shape = th.prod(th.tensor(img_shape)).item()
            self.hidden_layers = hidden_layers
            self.layers = [self.flattened_img_shape] + self.hidden_layers + [1]
            self._model_generator()

        def _model_generator(self):
            input_dim = self.flattened_img_shape
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

    def __init__(self, latent_dim, img_shape, **model_args) -> None:
        super().__init__(latent_dim, img_shape, **model_args)
        self.criterion = nn.BCELoss()

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

    def __init__(self, latent_dim, img_shape, criterion, **model_args) -> None:

        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.criterion = criterion

        self.D = Discriminator(img_shape=img_shape,
                               **model_args['discriminator'])
        self.G = Generator(img_shape, latent_dim, **model_args['generator'])

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


class DCGAN(GANModels):
    class Generator(GANModels.Generator):
        pass

    class Discriminator(GANModels.Discriminator):
        pass


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
