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


class DynamicUNet(nn.Module):
    def __init__(self, opt):
        super(DynamicUNet, self).__init__()

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
                nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                   padding=1,
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


##############################
#           U-NET
##############################



class OriginalUNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(OriginalUNet, self).__init__()

        def conv_block_3d(in_dim, out_dim, activation):
            return nn.Sequential(
                nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(out_dim),
                activation, )

        def conv_trans_block_3d(in_dim, out_dim, activation):
            return nn.Sequential(
                nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm3d(out_dim),
                activation, )

        def max_pooling_3d():
            return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        def conv_block_2_3d(in_dim, out_dim, activation):
            return nn.Sequential(
                conv_block_3d(in_dim, out_dim, activation),
                nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(out_dim), )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.pool_5 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)

        # Output
        self.out = conv_block_3d(self.num_filters, out_dim, activation)

    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x)  # -> [1, 4, 128, 128, 128]
        pool_1 = self.pool_1(down_1)  # -> [1, 4, 64, 64, 64]

        down_2 = self.down_2(pool_1)  # -> [1, 8, 64, 64, 64]
        pool_2 = self.pool_2(down_2)  # -> [1, 8, 32, 32, 32]

        down_3 = self.down_3(pool_2)  # -> [1, 16, 32, 32, 32]
        pool_3 = self.pool_3(down_3)  # -> [1, 16, 16, 16, 16]

        down_4 = self.down_4(pool_3)  # -> [1, 32, 16, 16, 16]
        pool_4 = self.pool_4(down_4)  # -> [1, 32, 8, 8, 8]

        down_5 = self.down_5(pool_4)  # -> [1, 64, 8, 8, 8]
        pool_5 = self.pool_5(down_5)  # -> [1, 64, 4, 4, 4]

        # Bridge
        bridge = self.bridge(pool_5)  # -> [1, 128, 4, 4, 4]

        # Up sampling
        trans_1 = self.trans_1(bridge)  # -> [1, 128, 8, 8, 8]
        concat_1 = torch.cat([trans_1, down_5], dim=1)  # -> [1, 192, 8, 8, 8]
        up_1 = self.up_1(concat_1)  # -> [1, 64, 8, 8, 8]

        trans_2 = self.trans_2(up_1)  # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, down_4], dim=1)  # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2)  # -> [1, 32, 16, 16, 16]

        trans_3 = self.trans_3(up_2)  # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, down_3], dim=1)  # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3)  # -> [1, 16, 32, 32, 32]

        trans_4 = self.trans_4(up_3)  # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, down_2], dim=1)  # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4)  # -> [1, 8, 64, 64, 64]

        trans_5 = self.trans_5(up_4)  # -> [1, 8, 128, 128, 128]
        concat_5 = torch.cat([trans_5, down_1], dim=1)  # -> [1, 12, 128, 128, 128]
        up_5 = self.up_5(concat_5)  # -> [1, 4, 128, 128, 128]

        # Output
        out = self.out(up_5)  # -> [1, 3, 128, 128, 128]
        return out

def make_generator(opt):
    if opt.generator == "flat":
        generator = FlatConvolution(opt)
    elif opt.generator == "original":
        generator = OriginalUNet(in_dim=opt.channels, out_dim=opt.channels, num_filters=opt.num_filters)
    elif opt.generator == "dynamic":
        generator =  DynamicUNet(opt)

    if opt.cuda is True:
        generator = generator.cuda()
    optimizer = torch.optim.Adam(
            generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
        )
    return generator, optimizer
