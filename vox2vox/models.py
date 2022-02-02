import torch
from torch import nn
import torch.nn.functional as functional


class X4Compressor(nn.Module):
    def __init__(self, inch):
        super(X4Compressor, self).__init__()
        def down_layer(inch, outch, padding_value):
            return nn.Sequential(
                nn.Conv3d(inch, outch, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(True),
                # nn.Conv3d(outch, outch, kernel_size=3, stride=1, padding=1),
                # nn.LeakyReLU(True),
                # nn.Conv3d(outch, outch, kernel_size=3, stride=1, padding=1),
                # nn.LeakyReLU(True),
                nn.Dropout3d(p=0.1),
                nn.BatchNorm3d(outch),
                nn.MaxPool3d(kernel_size=2),
            )

        def up_layer(inch, outch):
            return nn.Sequential(
                nn.ConvTranspose3d(inch, outch, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                # nn.ConvTranspose3d(outch, outch, kernel_size=3, stride=1, padding=1),
                # nn.ReLU(True),
                # nn.ConvTranspose3d(outch, outch, kernel_size=3, stride=1, padding=1),
                # nn.ReLU(True),
                # # nn.Dropout3d(p=0.1),
                nn.BatchNorm3d(outch),
            )

        self.down1 = down_layer(inch, 4*inch, 0)
        self.down2 = down_layer(4*inch, 16*inch, 0)

        self.up2 = up_layer(16*inch, 16*inch)
        self.up1 = up_layer(16*inch, 16*inch)

        self.final = nn.Conv3d(16*inch, inch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.down2(self.down1(x))

    def roundtrip(self, x):
        # print(f"{x.shape=}")
        # print(f"{self.up1(self.up2(self.down2(self.down1(x)))).shape=}")
        return self.final(self.up1(self.up2(self.down2(self.down1(x)))))


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
                    X4Compressor,
            )):
                pass
            else:
                raise NotImplementedError(f"How to initialize {type(m)}?")

        self.apply(_init_weights_)


class LinearConvolution(nn.Module):
    def __init__(self):
        super(LinearConvolution, self).__init__()

        self.model = nn.Conv3d(1, 1, kernel_size=(1, 1, 1))

    def forward(self, x):
        return self.model(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # self.down2 = nn.Sequential(
        #     nn.AvgPool3d(kernel_size=4, stride=4),
        #     nn.Conv3d(1, 16, kernel_size=(3,) * 3, padding=1, bias=False),
        #     nn.ELU(True),
        #     nn.Conv3d(16, 16, kernel_size=(3,) * 3, padding=1, bias=False),
        #     nn.ELU(True),
        #
        # )

        # self.down1 = nn.Sequential(
        #     nn.AvgPool3d(kernel_size=2),
        #     nn.Conv3d(1, 16, kernel_size=(3,)*3, bias=False),
        #     nn.LeakyReLU(True),
        #     nn.Conv3d(16, 16, kernel_size=(3,)*3, bias=False),
        #     nn.LeakyReLU(True),
        #
        #     nn.Conv3d(16, 16, kernel_size=(1,) * 3, bias=False),
        #
        #     nn.ConvTranspose3d(16, 16, kernel_size=(2,) * 3, stride=(2,) * 3, bias=False),
        #     nn.Conv3d(16, 16, kernel_size=(3,) * 3, bias=False),
        #     nn.ELU(True),
        #     nn.Conv3d(16, 1, kernel_size=(3,) * 3, bias=False),
        #     nn.ELU(True),
        # )
        self.first = nn.Conv3d(1, 1, kernel_size=(3,)*3, bias=False)

        self.second = nn.Sequential(
            nn.AvgPool3d(kernel_size=2),
            nn.Conv3d(1, 4, kernel_size=(3,)*3, bias=False),
            nn.ELU(True),

            nn.Conv3d(4, 4, kernel_size=(1,) * 3, bias=False),

            nn.ConvTranspose3d(4, 4, kernel_size=(2,) * 3, stride=(2,) * 3, bias=False),
            nn.ELU(True),
            nn.Conv3d(4, 1, kernel_size=(3,) * 3, bias=False),
        )

        self.third = nn.Sequential(
            nn.AvgPool3d(kernel_size=4),
            nn.Conv3d(1, 16, kernel_size=(3,)*3, bias=False),
            nn.ELU(True),
            # nn.Conv3d(16, 16, kernel_size=(3,)*3, bias=False),
            # nn.ELU(True),

            nn.Conv3d(16, 16, kernel_size=(1,) * 3, bias=False),

            nn.ConvTranspose3d(16, 16, kernel_size=(4,) * 3, stride=(2,) * 3, bias=False),
            nn.ELU(True),
            nn.ConvTranspose3d(16, 16, kernel_size=(2,) * 3, stride=(2,) * 3, bias=False),
            nn.ELU(True),
            nn.ConvTranspose3d(16, 16, kernel_size=(3,) * 3, bias=False),
            nn.ELU(True),
            nn.ConvTranspose3d(16, 1, kernel_size=(3,) * 3, bias=False),
        )

        self.merger = nn.Conv3d(3, 1, kernel_size=(1,)*3, bias=False)

    def forward(self, x, x1, x3):
        return self.merger(
            torch.cat(
                [
                    self.first(x1),
                    self.second(x3),
                    self.third(x),
                ],
                dim=1,
            )
        )

    def forward_crop(self, x):
        # print(f"{x.shape=}")
        # print(f"{self.down1(x).shape=}")
        return self.final(
            torch.cat(
                [
                    self.down1(x),
                    x[:, :, 6:-6, 6:-6, 6:-6],
                ],
                dim=1,
            )
        )

