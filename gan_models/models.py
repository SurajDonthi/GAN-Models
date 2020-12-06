import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.layers(latent_dim, out_dim)

    def layers(self, latent_dim, out_dim):
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, out_dim)

    def forward(self, X):
        X = F.leaky_relu(self.fc1(X), 0.2)
        X = F.leaky_relu(self.fc2(X), 0.2)
        X = F.leaky_relu(self.fc3(X), 0.2)
        X = torch.tanh(self.fc4(X))

        return X


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.layers()

    def layers(self):
        self.fc1 = nn.Linear(self.input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, X):
        X = X.view(X.size(0), self.input_dim)
        X = F.leaky_relu(self.fc1(X), 0.2)
        X = F.dropout(X, 0.3)
        X = F.leaky_relu(self.fc2(X), 0.2)
        X = F.dropout(X, 0.3)
        X = F.leaky_relu(self.fc3(X), 0.2)
        X = F.dropout(X, 0.3)
        X = torch.sigmoid(self.fc4(X))
        return X


class GAN:
    def __init__(self):
        self.discriminator = Discriminator
        self.generator = Generator


# ToDo: Implement the below models with similar interface!


class FMGAN:
    """
    Feature Matching for training the Vanilla GAN model.
    """

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
