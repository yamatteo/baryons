import logging

import torch
import torch.nn as nn


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


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters, depth, activation=None):
        super(Generator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.depth = depth
        if activation is None:
            activation = nn.LeakyReLU(0.2, inplace=True)

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

    def verbose_forward(self, x, **kwargs):
        logging.debug(f"Starting generator forward on tensor {x.shape} with {kwargs = }")
        stack = [x, ]
        for z in range(self.depth + 1):
            stack.append(self.down_levels[z](stack[-1]))
            logging.debug(f"depth {z} - append convoluted {stack[-1].shape} to stack")

        logging.debug("")
        logging.debug(f"stack is {[t.shape for t in stack]}")
        logging.debug("")
        for z in reversed(range(self.depth + 1)):
            if z < self.depth:
                stack, queue = stack[:-2], stack[-2:]
                stack.append(self.connections[z](torch.cat(queue, dim=1)))
                logging.debug(
                    f"concat of {queue[0].shape} and {queue[1].shape} to replace last two stack elements with {stack[-1].shape}")
            stack.append(self.up_levels[z](stack.pop()))
            logging.debug(f"depth {z} - replace last element of stack with deconvoluted {stack[-1].shape}")
        return stack[-1]

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


class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()

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
