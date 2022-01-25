import torch
from torch import nn


def down_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, bias=False),
        nn.LeakyReLU(True),
        nn.Dropout3d(p=0.1),
        nn.BatchNorm3d(out_channels),
    )

def upward_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, bias=False),
        nn.ReLU(True),
        # nn.Conv3d(out_channels, out_channels, kernel_size=1),
        # nn.ReLU(True),
        # nn.Dropout(p=0.1),
        nn.BatchNorm3d(out_channels),
    )

class SUNetOutermost(nn.Module):
    def __init__(self, input_nc, inner_nc, outer_nc, submodule):
        super(SUNetOutermost, self).__init__()

        self.model = nn.Sequential(
            nn.Conv3d(input_nc, inner_nc, kernel_size=3, bias=False),
            submodule,
            nn.ReLU(True),
            nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=3, bias=False),
            # nn.ReLU(True),
            # nn.BatchNorm3d(inner_nc),
            # nn.Conv3d(inner_nc, outer_nc, kernel_size=1, bias=False),
            # nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        # logger.debug(f"outermost {x.shape=}")
        output = self.model(x)
        # logger.debug(f"outermost {output.shape=}")
        return output


class SUNetIntermediate(nn.Module):
    def __init__(self, nc, submodule, ):
        super(SUNetIntermediate, self).__init__()

        self.model = nn.Sequential(
            down_layer(nc, 2*nc),
            # nn.LeakyReLU(0.2, True),
            # nn.Conv3d(nc, nc * 2, kernel_size=3, bias=False),
            # nn.BatchNorm3d(nc * 2),
            submodule,
            upward_layer(4*nc, nc),
            # nn.ReLU(True),
            # nn.ConvTranspose3d(nc * 4, nc, kernel_size=3, bias=False, ),
            # nn.BatchNorm3d(nc),
            # nn.Dropout(0.2),
        )

    def forward(self, x):
        # logger.debug(f"intermediate: {x.shape=}")
        output = torch.cat([x, self.model(x)], 1)
        # logger.debug(f"intermediate: {output.shape=}")
        return output


class SUNetInnermost(nn.Module):
    def __init__(self, nc):
        super(SUNetInnermost, self).__init__()

        self.model = nn.Sequential(
            down_layer(nc, nc),
            # nn.LeakyReLU(0.2, True),
            # nn.Conv3d(nc, nc, kernel_size=3, bias=False),
            # nn.ReLU(True),
            upward_layer(nc, nc),
            # nn.ConvTranspose3d(nc, nc, kernel_size=3, bias=False),
            # nn.BatchNorm3d(nc),
        )

    def forward(self, x):
        # logger.debug(f"innermost: {x.shape=}")
        output = torch.cat([x, self.model(x)], 1)
        # logger.debug(f"innermost: {output.shape=}")
        return output


class SUNet(nn.Module):
    def __init__(self, features, levels, channels=1):
        super(SUNet, self).__init__()
        block = SUNetInnermost(
            nc=2 ** levels * features,
        )
        for lvl in reversed(range(levels)):
            block = SUNetIntermediate(2 ** lvl * features, block)
        self.model = SUNetOutermost(
            input_nc=channels,
            inner_nc=features,
            outer_nc=1,
            submodule=block,
        )

    def forward(self, x):
        # x = torch.log(x + 1e-8)
        return self.model(x)

    def initialize(self):
        def _init_weights_(m):
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.normal_(m.weight.data, 0.0, 0.05)
            elif isinstance(m, (nn.BatchNorm3d,)):
                nn.init.normal_(m.weight.data, 0.1, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (
                    nn.LeakyReLU,
                    nn.ReLU,
                    nn.Tanh,
                    nn.Dropout,
                    nn.Dropout3d,
                    nn.InstanceNorm3d,
                    nn.Sequential,
                    SUNetInnermost,
                    SUNetIntermediate,
                    SUNetOutermost,
                    SUNet,
            )):
                pass
            else:
                raise NotImplementedError(f"How to initialize {type(m)}?")

        self.apply(_init_weights_)
