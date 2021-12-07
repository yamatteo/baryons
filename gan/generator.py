import logging

import torch
import torch.nn as nn


class FlatConvolution(nn.Module):
    def __init__(self, opt):
        super(FlatConvolution, self).__init__()
        self.tensor_type = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

        self.model = nn.Sequential(
            nn.BatchNorm3d(num_features=1),
            *[
                layer
                for i in range(0, opt.generator_depth)
                for layer in [
                    nn.Conv3d(in_channels=2 ** i, out_channels=2 ** (i + 1), kernel_size=3, stride=1, padding=1,
                              bias=False),
                    nn.BatchNorm3d(num_features=2 ** (i + 1)),
                    nn.ReLU(inplace=True),
                ]
            ],
            *[
                layer
                for i in reversed(range(0, opt.generator_depth))
                for layer in [
                    nn.Conv3d(in_channels=2 ** (i + 1), out_channels=2 ** i, kernel_size=3, stride=1, padding=1,
                              bias=False),
                    nn.BatchNorm3d(num_features=2 ** i),
                    nn.ReLU(inplace=True),
                ]
            ],
        )

    def forward(self, real_dm):
        return self.model(real_dm)


def convolution_down_level(in_channels, out_channels, activation):
    return nn.Sequential(
        nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(num_features=out_channels),
        nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(num_features=out_channels),
        activation,
    )


def convolution_up_level(in_channels, out_channels, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1,
                           output_padding=1),
        nn.BatchNorm3d(num_features=out_channels),
        nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(num_features=out_channels),
        activation,
    )


def convolution_same_level(in_channels, out_channels, activation):
    return nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(num_features=out_channels),
        nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(num_features=out_channels),
        activation,
    )


class OriginalUNet(nn.Module):
    def __init__(self, opt):
        super(OriginalUNet, self).__init__()
        in_channels = opt.channels
        out_channels = opt.channels
        num_filters = opt.num_filters
        depth = opt.generator_depth
        activation = nn.LeakyReLU(0.2, inplace=True)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.depth = depth

        self.down_levels = nn.ModuleList([
            convolution_same_level(
                in_channels=in_channels,
                out_channels=num_filters,
                activation=activation,
            )
            if z == 0 else
            convolution_down_level(
                in_channels=num_filters * 2 ** (z - 1),
                out_channels=num_filters * 2 ** z,
                activation=activation,
            )
            for z in range(depth + 1)
        ])

        self.up_levels = nn.ModuleList([
            convolution_same_level(
                in_channels=num_filters,
                out_channels=out_channels,
                activation=activation
            )
            if z == 0 else
            convolution_up_level(
                in_channels=num_filters * 2 ** z,
                out_channels=num_filters * 2 ** z,
                activation=activation,
            )
            for z in range(depth + 1)
        ])

        self.connections = nn.ModuleList([
            convolution_same_level(
                in_channels=num_filters * 3 * 2 ** z,
                out_channels=num_filters * 2 ** z,
                activation=activation,
            )
            for z in range(depth)
        ])

    def forward(self, x):
        stack = [x, ]
        for z in range(self.depth + 1):
            stack.append(self.down_levels[z](stack[-1]))

        for z in reversed(range(self.depth + 1)):
            if z < self.depth:
                stack, queue = stack[:-2], stack[-2:]
                stack.append(self.connections[z](torch.cat(queue, dim=1)))
            stack.append(self.up_levels[z](stack.pop()))
        return stack[-1]


def make_generator(opt):
    if opt.generator == "flat":
        return FlatConvolution(opt)
    elif opt.generator == "original":
        return OriginalUNet(opt)
