import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weights_init_normal(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def convolution_down_level(in_channels, out_channels, activation):
    return nn.Sequential(
        nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(num_features=out_channels),
        activation,
    )


def convolution_up_level(in_channels, out_channels, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1,
                           output_padding=1),
        nn.BatchNorm3d(num_features=out_channels),
        activation,
    )


def convolution_same_level(in_channels, out_channels, activation):
    return nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(num_features=out_channels),
        activation,
    )


def discriminator_block(in_channels, out_channels, activation, normalization=None):
    layers = [nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)]
    if normalization == "batch":
        layers.append(nn.BatchNorm3d(out_channels))
    elif normalization == "instance":
        layers.append(nn.InstanceNorm3d(out_channels))
    layers.append(activation)
    return nn.Sequential(*layers)


def conv_down(in_channels, activation, normalization):
    layers = [
        nn.Conv3d(
            in_channels=in_channels,
            out_channels=2 * in_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        ),
        nn.Conv3d(
            in_channels=2 * in_channels,
            out_channels=4 * in_channels,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        )
    ]
    if normalization == "batch":
        layers.append(nn.BatchNorm3d(4 * in_channels))
    elif normalization == "instance":
        layers.append(nn.InstanceNorm3d(4 * in_channels))
    if activation == "leaky":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def conv_across(in_channels, activation, normalization):
    layers = [
        nn.Conv3d(
            in_channels=in_channels,
            out_channels=2 * in_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        ),
        nn.Conv3d(
            in_channels=2 * in_channels,
            out_channels=4 * in_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        ),
    ]
    if normalization == "batch":
        layers.append(nn.BatchNorm3d(4 * in_channels))
    elif normalization == "instance":
        layers.append(nn.InstanceNorm3d(4 * in_channels))
    if activation == "leaky":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class MonoPatch(nn.Module):
    def __init__(self, shrink_exp, patch_exp, extra_depth, activation, normalization, tensor_type):
        super(MonoPatch, self).__init__()
        self.tensor_type = tensor_type

        conv_layers = [
            conv_across(2 * 4 ** i, activation, normalization)
            if i < extra_depth else
            conv_down(2 * 4 ** i, activation, normalization)
            for i in range(patch_exp + extra_depth)
        ]

        self.model = nn.Sequential(
            torch.nn.AvgPool3d(kernel_size=2 ** shrink_exp),
            *conv_layers,
            nn.Conv3d(2 * 4 ** (patch_exp + extra_depth), 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        )
        logging.debug(self.model)

    def forward(self, gas, dm):
        return self.model(torch.cat((gas, dm), dim=1))

    def evaluate(self, dm, real_gas, pred_gas):
        about_pred = self.model(torch.cat((pred_gas, dm), dim=1))
        return F.mse_loss(about_pred, self.tensor_type(np.ones(about_pred.shape)))

    def loss(self, dm, real_gas, pred_gas):
        about_real = self.model(torch.cat((real_gas, dm), dim=1))
        about_pred = self.model(torch.cat((pred_gas, dm), dim=1))
        return (F.mse_loss(about_real, self.tensor_type(np.ones(about_real.shape)))
                + F.mse_loss(about_pred, self.tensor_type(np.zeros(about_pred.shape))))

class Original(nn.Module):
    def __init__(self, opt):
        super(Original, self).__init__()
        self.tensor_type = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

        channels = 1
        self.lambda_pixel = opt.lambda_pixel
        self.model = nn.Sequential(
            discriminator_block(channels * 2, 64, activation=nn.LeakyReLU(0.2, inplace=True)),
            discriminator_block(64, 128, activation=nn.LeakyReLU(0.2, inplace=True), normalization="instance"),
            discriminator_block(128, 256, activation=nn.LeakyReLU(0.2, inplace=True), normalization="instance"),
            discriminator_block(256, 512, activation=nn.LeakyReLU(0.2, inplace=True), normalization="instance"),
            nn.ConstantPad3d((1, 0, 1, 0, 1, 0), 0),
            nn.Conv3d(512, 1, kernel_size=4, padding=1, bias=False)
        )

    def forward(self, gas, dm):
        # Concatenate image and condition image by channels to produce input
        return self.model(torch.cat((gas, dm), dim=1))

    def evaluate(self, real_dm, real_gas, pred_gas):
        about_pred = self.model(torch.cat((pred_gas, real_dm), dim=1))
        return F.mse_loss(about_pred, self.tensor_type(np.ones(about_pred.shape))) + self.tensor_type([self.lambda_pixel]) * F.l1_loss(pred_gas, real_gas)

    def loss(self, dm, real_gas, pred_gas):
        about_real = self.model(torch.cat((real_gas, dm), dim=1))
        about_pred = self.model(torch.cat((pred_gas, dm), dim=1))
        return (F.mse_loss(about_real, self.tensor_type(np.ones(about_real.shape)))
                + F.mse_loss(about_pred, self.tensor_type(np.zeros(about_pred.shape))))

class OnlyMSE(nn.Module):
    def __init__(self, opt):
        super(OnlyMSE, self).__init__()
        self.tensor_type = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

        # channels = 1
        # self.lambda_pixel = opt.lambda_pixel
        self.model = nn.Conv3d(1, 1, 3)

    def forward(self, gas, dm):
        # Concatenate image and condition image by channels to produce input
        return None

    def evaluate(self, real_dm, real_gas, pred_gas):
        return F.mse_loss(real_gas, pred_gas)

    def loss(self, dm, real_gas, pred_gas):
        return self.tensor_type([0.]).requires_grad_()

class MultiMSE(nn.Module):
    def __init__(self, opt):
        super(MultiMSE, self).__init__()
        self.tensor_type = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

        # channels = 1
        # self.lambda_pixel = opt.lambda_pixel
        self.model = nn.Conv3d(1, 1, 3)

    def forward(self, gas, dm):
        # Concatenate image and condition image by channels to produce input
        return None

    def evaluate(self, real_dm, real_gas, pred_gas):
        return F.mse_loss(real_gas, pred_gas) \
               + F.mse_loss(F.max_pool3d(real_gas, kernel_size=4), F.max_pool3d(pred_gas, kernel_size=4)) \
               + F.mse_loss(F.max_pool3d(real_gas, kernel_size=16), F.max_pool3d(pred_gas, kernel_size=16))

    def loss(self, dm, real_gas, pred_gas):
        return self.tensor_type([0.]).requires_grad_()

class FakeOptimizer():
    def __init__(self):
        pass

    def step(self):
        pass

    def zero_grad(self):
        pass


def make_discriminator(opt):
    tensor_type = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor
    if opt.discriminator == "original":
        discriminator = Original(opt)
    elif opt.discriminator == "monopatch":
        discriminator = MonoPatch(3, 2, 0, "leaky", "instance", tensor_type)
    elif opt.discriminator == "onlymse":
        discriminator = OnlyMSE(opt)
    elif opt.discriminator == "multimse":
        discriminator = MultiMSE(opt)

    if opt.cuda is True:
        discriminator = discriminator.cuda()
    if opt.discriminator in ("original", "monopatch"):
        optimizer = torch.optim.Adam(
                discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
            )
    elif opt.discriminator in ("onlymse", "multimse"):
        optimizer = FakeOptimizer()
    return discriminator, optimizer