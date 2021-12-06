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


def discriminator_block(in_channels, out_channels, kernel_size, stride, padding, activation, normalization):
    layers = [nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                        padding=padding)]
    if normalization == "batch":
        layers.append(nn.BatchNorm3d(out_channels))
    elif normalization == "instance":
        layers.append(nn.InstanceNorm3d(out_channels))
    if activation == "leaky":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
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


class Discriminator(nn.Module):
    def __init__(self, shrink_exp, patch_exp, extra_depth, activation, normalization, tensor_type):
        super(Discriminator, self).__init__()
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

    def evaluate(self, dm, pred_gas):
        about_pred = self.model(torch.cat((pred_gas, dm), dim=1))
        return F.mse_loss(about_pred, self.tensor_type(np.ones(about_pred.shape)))

    def loss(self, dm, real_gas, pred_gas):
        about_real = self.model(torch.cat((real_gas, dm), dim=1))
        about_pred = self.model(torch.cat((pred_gas, dm), dim=1))
        return (F.mse_loss(about_real, self.tensor_type(np.ones(about_real.shape)))
                + F.mse_loss(about_pred, self.tensor_type(np.zeros(about_pred.shape))))
