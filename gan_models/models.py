import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, out_dim: int, latent_dim: int = 100,
                 normalize: bool = False,
                 hidden_layers: list = [128, 256, 512, 1024]):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.hidden_layers = hidden_layers
        self.layers = [latent_dim] + self.hidden_layers + [out_dim]
        self.normalize = normalize
        self._model_generator()

    def _model_generator(self):
        input_dim = self.layers[0]
        for i, dim in enumerate(self.layers[1:]):
            name = f'linear{i}'
            layer = self.linear_block(input_dim, dim)
            setattr(self, name, nn.Sequential(*layer))
            input_dim = dim

    def linear_block(self, input_feat, out_feat):
        layers = [nn.Linear(input_feat, out_feat)]
        if self.normalize:
            layers.append(nn.BatchNorm1d(out_feat, eps=0.8))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        return layers

    def forward(self, X):
        for i, _ in enumerate(self.layers[1:]):
            name = f'linear{i}'
            linear = getattr(self, name)
            X = linear(X)

        return F.tanh(X)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_layers=[512, 256]):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.layers = [input_dim] + self.hidden_layers + [1]
        self._model_generator()

    def _model_generator(self):
        input_dim = self.input_dim
        for i, dim in enumerate(self.layers[1:]):
            name = f'linear{i}'
            layer = self.linear_block(input_dim, dim)
            setattr(self, name, nn.Sequential(*layer))
            input_dim = dim

    def linear_block(self, input_feat, out_feat):
        layers = [nn.Linear(input_feat, out_feat)]
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        return layers

    def forward(self, X):
        for i, _ in enumerate(self.layers[1:]):
            name = f'linear{i}'
            linear = getattr(self, name)
            X = linear(X)
        X = F.sigmoid(X)

        return X


class GAN:
    def __init__(self):
        self.discriminator = Discriminator
        self.generator = Generator


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
