import torch as th
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from utils import filtered_kwargs


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


class VanillaGAN1D(GANModels):

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


class FMGAN2D(GANModels):
    """
    Feature Matching for training the Vanilla GAN model.
    """

    class Generator(GANModels.Generator):
        def __init__(self, img_shape: int, latent_dim: int = 100,
                     #  normalize: bool = False,
                     hidden_channels: list = [256, 128, 64, 32], **kwargs):
            super().__init__(img_shape, latent_dim)
            self.hidden_channels = hidden_channels
            self.out_channels = self.img_shape[0]

            self._model_generator()

        def conv_block(self, in_channels, out_channels, normalize=True):
            layers = [nn.ConvTranspose2d(
                in_channels, out_channels, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels, eps=0.8))
            layers.append(nn.LeakyReLU(0.02, inplace=True))
            return nn.Sequential(*layers)

        def _model_generator(self):
            input_channels = self.latent_dim
            for i, channels in enumerate(self.hidden_channels):
                name = f'conv{i}'
                layer = self.conv_block(input_channels, channels,
                                        normalize=False if i == 0 else True)

                setattr(self, name, layer)
                input_channels = channels

            self.final = nn.Sequential(*[
                nn.ConvTranspose2d(
                    input_channels, self.out_channels, 4, 2, 1, bias=False),
                nn.Tanh()
            ])

        def forward(self, X):
            X = X.view(-1, self.latent_dim, 1, 1)
            for i, _ in enumerate(self.hidden_channels):
                name = f'conv{i}'
                conv = getattr(self, name)
                X = conv(X)

            X = self.final(X)

            return X

    class Discriminator(GANModels.Discriminator):
        def __init__(self, img_shape, hidden_channels=[32, 64, 128, 256], **kwargs):
            super().__init__(img_shape)
            self.hidden_channels = hidden_channels
            self._model_generator()

        def conv_block(self, in_channels, out_channels,
                       activation=True, normalize=True):

            layers = [nn.Conv2d(in_channels, out_channels,
                                4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            if activation:
                layers.append(nn.LeakyReLU(0.02, inplace=True))
            return nn.Sequential(*layers)

        def get_output_shape(self, model, image_dim):
            return model(th.rand(*(image_dim))).data.shape

        def _model_generator(self):
            in_channels = self.img_shape[0]
            layer_dim = [1] + list(self.img_shape)
            for i, channels in enumerate(self.hidden_channels):
                name = f'conv{i}'

                layer = self.conv_block(in_channels, channels,
                                        activation=False if i == len(
                                            self.hidden_channels)-1 else True,
                                        normalize=False if i == 0 else True)
                layer_dim = self.get_output_shape(layer, layer_dim)

                setattr(self, name, layer)
                in_channels = channels

            self.layer_dim = th.prod(th.tensor(layer_dim)).item()
            self.final = nn.Linear(self.layer_dim, 10)

        def forward(self, X, feature_matching=False):

            for i, _ in enumerate(self.hidden_channels):
                name = f'conv{i}'
                conv = getattr(self, name)
                X = conv(X)

            feature = X.view(-1, self.layer_dim)
            X = self.final(feature)
            if feature_matching:
                return feature

            # X = F.sigmoid(self.final(X))
            return X

    def __init__(self, latent_dim, img_shape, **model_args) -> None:
        super().__init__(latent_dim, img_shape, **model_args)
        self.g_criterion = nn.MSELoss()
        self.d_criterion = nn.CrossEntropyLoss()

    def _init_device(self, device):
        self.device = device

    def log_sum_exp(self, tensor, keepdim=True):
        r"""
        Numerically stable implementation for the `LogSumExp` operation. The
        summing is done along the last dimension.
        Args:
            tensor (torch.Tensor)
            keepdim (Boolean): Whether to retain the last dimension on summing.
        """
        max_val = tensor.max(dim=-1, keepdim=True)[0]
        return max_val + (tensor - max_val).exp().sum(dim=-1, keepdim=keepdim).log()

    def D_loss(self, batch):
        real, label = batch

        # Real preds
        preds = self.D(real)

        # Label Loss
        label_loss = self.d_criterion(preds, label)

        # Unsupervised Real loss
        lse_out = self.log_sum_exp(preds)
        unsupervised_real_loss = - \
            th.mean(lse_out, 0) + th.mean(F.softplus(lse_out, 1), 0)

        # Unsupervised Fake loss
        z = th.randn(
            real.shape[0], self.latent_dim, device=self.device)
        preds = self.D(self.G(z))

        unsupervised_fake_loss = th.mean(
            F.softplus(self.log_sum_exp(preds), 1), 0)

        loss = label_loss + \
            (unsupervised_real_loss + unsupervised_fake_loss) / 2

        return loss

    def G_loss(self, batch):
        real, _ = batch

        real_feature = self.D(real, feature_matching=True)

        # Fake Loss
        z = th.randn(real.shape[0], self.latent_dim, device=self.device)
        self.gen_imgs = self.G(z)
        fake_feature = self.D(self.gen_imgs, feature_matching=True)

        loss = self.g_criterion(fake_feature, real_feature)

        return loss


class DCGAN(GANModels):
    class Generator(GANModels.Generator):
        pass

    class Discriminator(GANModels.Discriminator):
        pass


class WassersteinGAN:

    def __init__(self) -> None:
        self.D = None
        self.D = None


class CGAN:
    def __init__(self) -> None:
        self.D = None
        self.G = None


class InfoGAN:
    def __init__(self) -> None:
        self.D = None
        self.G = None


class CycleGAN:
    def __init__(self) -> None:
        self.D = None
        self.G = None


class BigGAN:
    def __init__(self) -> None:
        self.D = None
        self.G = None
