from abc import ABC, abstractmethod

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from gan_models.utils import filtered_kwargs


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
                     hidden_layers: list = [128, 256, 512, 1024], **kwargs):
            super().__init__(img_shape, latent_dim)

            self.flattened_img_shape = th.prod(th.tensor(img_shape)).item()
            self.hidden_layers = hidden_layers
            self.input_dim = self.latent_dim

            self.layers = [latent_dim] + \
                self.hidden_layers + [self.flattened_img_shape]
            # self.normalize = normalize
            self._model_generator()

        def _model_generator(self):
            input_dim = self.input_dim
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
        def __init__(self, img_shape, hidden_layers=[512, 256, 128], **kwargs):
            super().__init__(img_shape)
            self.flattened_img_shape = th.prod(th.tensor(img_shape)).item()
            self.hidden_layers = hidden_layers
            self.layers = [self.flattened_img_shape] + self.hidden_layers + [1]
            self.input_dim = self.flattened_img_shape
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

        # Fake_loss
        z = th.randn(
            real.shape[0], self.latent_dim, device=self.device)
        preds = self.D(self.G(z))

        fake = th.zeros_like(preds)
        fake_loss = self.criterion(preds, fake)

        loss = (real_loss + fake_loss) / 2

        return loss

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
    """
    DCGAN model.
    """

    class Generator(GANModels.Generator):
        def __init__(self, img_shape: int, latent_dim: int = 100,
                     hidden_channels: list = [256, 128, 64], **kwargs):
            super().__init__(img_shape, latent_dim)
            self.hidden_channels = hidden_channels
            self.out_channels = self.img_shape[0]

            self.__input_channels = self.latent_dim

            self._model_generator()

        def conv_block(self, in_channels, out_channels, kernel_size=4,
                       stride=2, padding=1, normalize=True):
            layers = [nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding,
                bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels, eps=0.8))
            layers.append(nn.LeakyReLU(0.02, True))
            return nn.Sequential(*layers)

        def _model_generator(self):
            input_channels = self.__input_channels
            for i, channels in enumerate(self.hidden_channels):
                name = f'conv{i}'
                layer = self.conv_block(input_channels, channels,
                                        stride=1 if i == 0 else 2,
                                        padding=0 if i == 0 else 1)

                setattr(self, name, layer)
                input_channels = channels

            self.final = nn.Sequential(*[
                nn.ConvTranspose2d(
                    input_channels, self.out_channels, 4, 2, 1, bias=False),
                nn.Tanh()
            ])

        def forward(self, X):
            X = X.view(-1, self.input_channels, 1, 1)
            for i, _ in enumerate(self.hidden_channels):
                name = f'conv{i}'
                conv = getattr(self, name)
                X = conv(X)

            X = self.final(X)

            return X

    class Discriminator(GANModels.Discriminator):
        def __init__(self, img_shape, hidden_channels=[64, 128, 256],
                     kernel_size=4, **kwargs):
            super().__init__(img_shape)
            self.hidden_channels = hidden_channels
            self.kernel_size = kernel_size
            self.in_channels = self.img_shape[0]
            self._model_generator()

        def conv_block(self, in_channels, out_channels,
                       activation=True, normalize=True):

            layers = [nn.Conv2d(in_channels, out_channels,
                                self.kernel_size, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            if activation:
                layers.append(nn.LeakyReLU(0.02, inplace=True))
            return nn.Sequential(*layers)

        def _model_generator(self):
            in_channels = self.in_channels
            for i, channels in enumerate(self.hidden_channels):
                name = f'conv{i}'

                layer = self.conv_block(in_channels, channels,
                                        activation=False if i == len(
                                            self.hidden_channels)-1 else True,
                                        normalize=False if i == 0 else True)

                setattr(self, name, layer)
                in_channels = channels

            self.final = nn.Conv2d(in_channels, 1, 4, 1, 0, bias=False)

        def forward(self, X):

            for i, _ in enumerate(self.hidden_channels):
                name = f'conv{i}'
                conv = getattr(self, name)
                X = conv(X)

            X = self.final(X)
            X = F.sigmoid(X)

            return X.view(-1)

    def __init__(self, latent_dim, img_shape, **model_args) -> None:
        super().__init__(latent_dim, img_shape, **model_args)
        self.criterion = nn.BCELoss()

    def D_loss(self, batch):
        real, _ = batch

        preds = self.D(real)

        # Real Loss
        valid = th.ones_like(preds)
        real_loss = self.criterion(preds, valid)

        # Fake loss
        z = th.randn(real.shape[0], self.latent_dim, device=self.device)
        preds = self.D(self.G(z))

        fake = th.zeros_like(preds)
        fake_loss = self.criterion(preds, fake)

        loss = real_loss + fake_loss

        return loss

    def G_loss(self, batch):
        real, _ = batch

        # Fake Loss
        z = th.randn(real.shape[0], self.latent_dim, device=self.device)
        self.gen_imgs = self.G(z)
        preds = self.D(self.gen_imgs)

        fake = th.ones_like(preds)
        loss = self.criterion(preds, fake)

        return loss


class DCGANMNIST(DCGAN):
    class Generator(DCGAN.Generator):
        def __init__(self, img_shape, latent_dim,
                     hidden_channels=[512, 256, 128], **kwargs):
            super().__init__(img_shape, latent_dim, hidden_channels, **kwargs)

            self._model_generator()

        def _model_generator(self):

            self.linear = nn.Sequential(*[
                nn.Linear(self.latent_dim, self.hidden_channels[0] * 7 * 7),
                nn.LeakyReLU(0.02, True)
            ])
            # h0 x 7 x 7 -> h1 x 14 x 14
            self.conv0 = self.conv_block(self.hidden_channels[0],
                                         self.hidden_channels[1],
                                         normalize=False)
            # h1 x 14 x 14 -> h2 x 15 x 15
            self.conv1 = self.conv_block(self.hidden_channels[1],
                                         self.hidden_channels[2],
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         normalize=True)
            # h2 x 15 x 15 -> 1 x 30 x 30
            self.conv2 = nn.Sequential(*[
                nn.ConvTranspose2d(self.hidden_channels[2], 1, 4, 2, 1),
                nn.Tanh()
            ])

            del self.final

        def forward(self, X):
            # X -> noise
            X = self.linear(X)
            X = X.view(-1, self.hidden_channels[0], 7, 7)

            for i, _ in enumerate(self.hidden_channels):
                name = f'conv{i}'
                conv = getattr(self, name)
                X = conv(X)

            return X

    class Discriminator(DCGAN.Discriminator):

        def _model_generator(self):
            super()._model_generator()

            self.final.stride = (3, 3)


class FMGAN2D(GANModels):
    """
    Implements Feature Matching and semi-supervised learning for DCGAN model.
    """

    class Discriminator(DCGANMNIST.Discriminator):
        def __init__(self, img_shape, hidden_channels=[128, 256, 512], **kwargs):
            super().__init__(img_shape)
            self._model_generator()

        def get_output_shape(self, model, image_dim):
            return model(th.rand(*(image_dim))).data.shape

        def _model_generator(self):
            super()._model_generator()
            # Get sizes of features at each layer
            layer_dim = [1] + list(self.img_shape)
            for i, channels in enumerate(self.hidden_channels):
                name = f'conv{i}'
                layer = getattr(self, name)
                layer_dim = self.get_output_shape(layer, layer_dim)

            self.layer_dim = th.prod(th.tensor(layer_dim)).item()
            self.final = nn.Linear(self.layer_dim, 10)

        def forward(self, X, feature_matching=False):

            for i, _ in enumerate(self.hidden_channels):
                name = f'conv{i}'
                conv = getattr(self, name)
                X = conv(X)

            feature = X.view(-1, self.layer_dim)
            if feature_matching:
                return feature

            X = self.final(feature)
            # X = F.sigmoid(X)
            return X

    def __init__(self, latent_dim, img_shape, **model_args) -> None:
        super().__init__(latent_dim, img_shape, **model_args)

        self.G = DCGANMNIST.Generator(img_shape, latent_dim,
                                      **model_args['generator'])

        self.g_criterion = nn.MSELoss()
        self.d_criterion = nn.CrossEntropyLoss()

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
        self.gen_imgs = self.G(z)       # Also for logging in TensorBoard
        fake_feature = self.D(self.gen_imgs, feature_matching=True)

        loss = self.g_criterion(fake_feature, real_feature)

        return loss


class CGAN(VanillaGAN1D):
    class Generator(VanillaGAN1D.Generator):
        def __init__(self, img_shape: int, latent_dim: int, num_classes: int,
                     hidden_layers: list = [256, 512, 1024]):
            super().__init__(img_shape, latent_dim=latent_dim,
                             hidden_layers=hidden_layers)

            self.num_classes = num_classes
            self.input_dim = self.latent_dim + self.num_classes
            self._model_generator()

            self.embeddings = nn.Embedding(num_classes, num_classes)

        def forward(self, noise, labels):
            noise = th.cat((noise, self.embeddings(labels)), dim=-1)
            return super().forward(noise)

    class Discriminator(VanillaGAN1D.Discriminator):
        def __init__(self, img_shape, num_classes, hidden_layers=[512, 256]):
            super().__init__(img_shape, hidden_layers=hidden_layers)

            self.num_classes = num_classes
            self.input_dim = self.flattened_img_shape + self.num_classes
            self._model_generator()

            self.embeddings = nn.Embedding(num_classes, num_classes)

        def forward(self, imgs, labels):
            imgs = th.cat(
                (imgs.view(imgs.shape[0], -1), self.embeddings(labels)),
                dim=-1)
            return super().forward(imgs)

    def __init__(self, latent_dim, img_shape, num_classes, **model_args) -> None:
        super().__init__(latent_dim, img_shape,  **model_args)

        self.num_classes = num_classes
        self.criterion = nn.BCELoss()

    def G_loss(self, batch):
        real, _ = batch
        # Fake Loss
        z = th.randn(real.shape[0], self.latent_dim, device=self.device)
        self.gen_labels = th.randint(10, (real.shape[0],), device=self.device)
        self.gen_imgs = self.G(z, self.gen_labels)
        preds = self.D(self.gen_imgs, self.gen_labels)

        valid = th.ones_like(preds, requires_grad=False)
        loss = self.criterion(preds, valid)

        return loss

    def D_loss(self, batch):
        real, labels = batch
        # Real Loss
        # Here preds -> Prediction whether real or fake
        preds = self.D(real, labels)

        valid = th.ones_like(preds)
        real_loss = self.criterion(preds, valid)

        # Fake_loss
        z = th.randn(real.shape[0], self.latent_dim, device=self.device)
        gen_labels = th.randint(10, (real.shape[0],), device=self.device)
        gen_imgs = self.G(z, gen_labels)
        preds = self.D(gen_imgs, gen_labels)

        fake = th.zeros_like(preds)
        fake_loss = self.criterion(preds, fake)

        loss = (real_loss + fake_loss) / 2

        return loss


class CGAN2D(CGAN):
    class Generator(DCGAN.Generator):
        def __init__(self, img_shape: int, latent_dim: int, num_classes: int,
                     hidden_channels: list = [256, 128, 64], **kwargs):
            super().__init__(img_shape, latent_dim=latent_dim,
                             hidden_channels=hidden_channels, **kwargs)

            self.num_classes = num_classes
            self.input_channels = self.latent_dim + self.num_classes
            self._model_generator()

            self.embeddings = nn.Embedding(num_classes, num_classes)

        def forward(self, noise, labels):
            noise = th.cat((noise, self.embeddings(labels)), dim=-1)

            return super().forward(noise)

    class Discriminator(DCGAN.Discriminator):
        def __init__(self, img_shape, num_classes,
                     hidden_channels=[64, 128, 256], kernel_size=4, **kwargs):
            super().__init__(img_shape, hidden_channels=hidden_channels,
                             kernel_size=kernel_size, **kwargs)

            self.num_classes = num_classes

            self.label_channels = 1
            if num_classes > (flattened_shape := img_shape[1] * img_shape[2]):
                self.label_channels += 1
            self.in_channels = self.img_shape[0] + self.label_channels

            self._model_generator()

            self.embeddings = nn.Embedding(num_classes, num_classes)
            self.fc = nn.Linear(
                num_classes, flattened_shape * self.label_channels)

        def forward(self, imgs, labels):
            labels = self.embeddings(labels)
            labels = self.fc(labels)
            labels = labels.view(-1, self.label_channels, imgs.shape[2],
                                 imgs.shape[3])
            imgs = th.cat((imgs, labels), dim=1)
            return super().forward(imgs)


class WassersteinGAN(GANModels):
    BASE_MODELS = {'mlp': VanillaGAN1D,
                   'dcgan': DCGAN, 'dcgan_mnist': DCGANMNIST}

    def __init__(self, latent_dim, img_shape, clip_value=0.01, **model_args) -> None:
        super().__init__(latent_dim, img_shape, **model_args)

        model_type = model_args['model_type']
        model = self.BASE_MODELS[model_type]
        self.model_args = model_args
        self.clip_value = clip_value

        # Create G & D again as we are setting MLP or DCGAN as base!
        self.G = model.Generator(
            img_shape, latent_dim, **model_args['generator'])
        self.D = model.Discriminator(img_shape, **model_args['discriminator'])

    def D_loss(self, batch):
        # Clip weights at (before or after doesn't make a diff) every iteration
        for p in self.D.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)

        # Real
        real, _ = batch
        real_preds = self.D(real)
        real_loss = -th.mean(real_preds)

        # Fake
        z = th.randn(real.shape[0], self.latent_dim, device=self.device)
        fake_preds = self.D(self.G(z))
        fake_loss = th.mean(fake_preds)

        loss = real_loss + fake_loss

        return loss

    def G_loss(self, batch):
        real, _ = batch

        # Fake
        z = th.randn(real.shape[0], self.latent_dim, device=self.device)
        self.gen_imgs = self.G(z)       # Also for logging in TensorBoard
        fake_preds = self.D(self.gen_imgs)

        loss = -th.mean(fake_preds)

        return loss


class CramerGAN(GANModels):
    BASE_MODELS = {'mlp': VanillaGAN1D, 'dcgan': DCGAN}

    def __init__(self, latent_dim, img_shape, gp_scale=10, **model_args) -> None:
        super().__init__(latent_dim, img_shape, **model_args)

        self.gp_scale = gp_scale

        model_type = model_args['model_type']
        model = self.BASE_MODELS[model_type]
        self.model_args = model_args

        self.G = model.Generator(
            img_shape, latent_dim, **model_args['generator'])
        self.D = model.Discriminator(img_shape, **model_args['discriminator'])

    def critic(self, x1, x2):
        return th.linalg.norm(x1 - x2, dim=1) - th.linalg.norm(x1, dim=1)

    def G_loss(self, batch):

        real, _ = batch

        real_preds = self.D(real)

        z1 = th.randn(real.shape[0], self.latent_dim, device=self.device)
        fake_preds1 = self.D(self.G(z1))

        z2 = th.randn(real.shape[0], self.latent_dim, device=self.device)
        self.gen_imgs = self.G(z2)      # For logging in TensorBoard
        fake_preds2 = self.D(self.gen_imgs)

        self.g_loss = th.mean(
            self.critic(real_preds, fake_preds1) -
            self.critic(fake_preds1, fake_preds2)
        )

        return self.g_loss

    def calc_gradient_penalty(self, real, fake, fake_preds):
        alpha = th.rand(real.shape[0], *((1,) * len(real.shape[1:])),
                        device=self.device)
        alpha = alpha.expand_as(real)

        interpolates = alpha * real + (1 - alpha) * fake
        interpolates.requires_grad_(True)

        d_interpolates = self.critic(self.D(interpolates), fake_preds)

        grad_outputs = th.ones_like(d_interpolates, requires_grad=True)

        gradients = th.autograd.grad(outputs=d_interpolates,
                                     inputs=interpolates,
                                     grad_outputs=grad_outputs,
                                     create_graph=True, retain_graph=True,
                                     only_inputs=True)[0]

        gradient_penalty = th.mean((gradients.norm(2, dim=1) - 1) ** 2)

        return gradient_penalty

    def D_loss(self, batch):
        real, _ = batch

        real_preds = self.D(real)

        z1 = th.randn(real.shape[0], self.latent_dim, device=self.device)
        fake1 = self.G(z1)
        fake_preds1 = self.D(fake1)

        z2 = th.randn(real.shape[0], self.latent_dim, device=self.device)
        fake2 = self.G(z2)
        fake_preds2 = self.D(fake2)

        surrogate = th.mean(
            self.critic(real_preds, fake_preds2) -
            self.critic(fake_preds1, fake_preds2)
        )

        gradient_penalty = self.calc_gradient_penalty(real, fake1, fake_preds1)

        loss = - surrogate + gradient_penalty * self.gp_scale
        return loss


# This has 3 optmizers & losses! Leave it for now!
# Will have to redefine the whole backend engine!
class InfoGAN(GANModels):
    BASE_MODELS = {'dcgan': DCGAN, 'dcgan_mnist': DCGANMNIST}

    class Discriminator(DCGAN.Discriminator):
        def __init__(self, img_shape, num_classes, code_dim=2,
                     hidden_channels=[64, 128, 256],
                     kernel_size=4, **kwargs):
            super().__init__(img_shape, hidden_channels=hidden_channels,
                             kernel_size=kernel_size, **kwargs)
            self.num_classes = num_classes
            self.code_dim = code_dim

            self._model_generator()

        def get_output_shape(self, model, image_dim):
            return model(th.rand(*(image_dim))).data.shape

        def _model_generator(self):
            super()._model_generator()
            del self.final

            layer_dim = [1] + list(self.img_shape)
            for i, channels in enumerate(self.hidden_channels):
                name = f'conv{i}'
                layer = getattr(self, name)
                layer_dim = self.get_output_shape(layer, layer_dim)

            self.layer_dim = th.prod(th.tensor(layer_dim)).item()
            self.adv_layer = nn.Linear(self.layer_dim, 1)
            self.aux_layer = nn.Sequential(*[
                nn.Linear(self.layer_dim, self.num_classes),
                nn.Softmax()
            ])
            self.latent_layer = nn.Linear(self.layer_dim, self.code_dim)

        def forward(self, X):
            for i, _ in enumerate(self.hidden_channels):
                name = f'conv{i}'
                conv = getattr(self, name)
                X = conv(X)

            X = X.view(-1, self.layer_dim)
            validity = self.adv_layer(X)
            label = self.aux_layer(X)
            latent_code = self.latent_layer(X)

            return validity, label, latent_code

    def __init__(self, latent_dim, img_shape,  num_classes,
                 code_dim=2, lambdas={'cat': 1, 'cont': 0.1},
                 **model_args) -> None:
        super().__init__(latent_dim, img_shape, **model_args)

        model_type = model_args['model_type']
        self.model = self.BASE_MODELS[model_type]
        self.model_args = model_args

        self.adversarial_loss = nn.MSELoss()
        self.categorical_loss = nn.CrossEntropyLoss()
        self.continuous_loss = nn.MSELoss()

        # Create G & D again as we are setting MLP or DCGAN as base!
        self.G = self.model.Generator(
            img_shape, latent_dim, **model_args['generator'])

    def D_loss(self, batch):
        loss = None
        return loss

    def G_loss(self, batch):
        real, _ = batch

        # Fake
        z = th.randn(real.shape[0], self.latent_dim, device=self.device)
        self.gen_imgs = self.G(z)       # Also for logging in TensorBoard
        fake_preds = self.D(self.gen_imgs)

        loss = -th.mean(fake_preds)

        return loss

    def info_loss(self, batch):
        pass


class CycleGAN(GANModels):
    pass


class BigGAN(GANModels):
    pass


MODELS = {
    'gan': VanillaGAN1D,
    'vanilla_gan': VanillaGAN1D,
    'dcgan': DCGAN,
    'dcgan_mnist': DCGANMNIST,
    'feature_matching': FMGAN2D,
    'cgan': CGAN,
    'cgan2d': CGAN2D,
    'wgan': WassersteinGAN,
    'cramer_gan': CramerGAN
}
