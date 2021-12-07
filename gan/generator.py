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
                    nn.Conv3d(in_channels=2 ** i, out_channels=2 ** (i + 1), kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm3d(num_features=2 ** (i + 1)),
                    nn.ReLU(inplace=True),
                ]
            ],
            *[
                layer
                for i in reversed(range(0, opt.generator_depth))
                for layer in [
                    nn.Conv3d(in_channels=2 ** (i+1), out_channels=2 ** i, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm3d(num_features=2 ** i),
                    nn.ReLU(inplace=True),
                ]
            ],
        )

    def forward(self, real_dm):
        return self.model(real_dm)


def make_generator(opt):
    if opt.generator == "flat":
        return FlatConvolution(opt)
