import logging

import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################

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


class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet, self).__init__()

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
        logging.debug(f"{down_5.shape = }")
        pool_5 = self.pool_5(down_5)  # -> [1, 64, 4, 4, 4]
        logging.debug(f"{pool_5.shape = }")


        # Bridge
        bridge = self.bridge(pool_5)  # -> [1, 128, 4, 4, 4]
        logging.debug(f"{bridge.shape = }")

        # Up sampling
        trans_1 = self.trans_1(bridge)  # -> [1, 128, 8, 8, 8]
        logging.debug(f"{trans_1.shape = }")
        concat_1 = torch.cat([trans_1, down_5], dim=1)  # -> [1, 192, 8, 8, 8]
        logging.debug(f"{concat_1.shape = }")
        up_1 = self.up_1(concat_1)  # -> [1, 64, 8, 8, 8]
        logging.debug(f"{up_1.shape = }")


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
        out = self.out(up_5)  # -> [1, 1, 256, 256, 256]
        return out


# class UNetDown(nn.Module):
#     def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
#         super(UNetDown, self).__init__()
#         layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
#         if normalize:
#             layers.append(nn.InstanceNorm2d(out_size))
#         layers.append(nn.LeakyReLU(0.2))
#         if dropout:
#             layers.append(nn.Dropout(dropout))
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)


# class UNetUp(nn.Module):
#     def __init__(self, in_size, out_size, dropout=0.0):
#         super(UNetUp, self).__init__()
#         layers = [
#             nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
#             nn.InstanceNorm2d(out_size),
#             nn.ReLU(inplace=True),
#         ]
#         if dropout:
#             layers.append(nn.Dropout(dropout))

#         self.model = nn.Sequential(*layers)

#     def forward(self, x, skip_input):
#         x = self.model(x)
#         x = torch.cat((x, skip_input), 1)

#         return x


# class GeneratorUNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3):
#         super(GeneratorUNet, self).__init__()

#         self.down1 = UNetDown(in_channels, 64, normalize=False)
#         self.down2 = UNetDown(64, 128)
#         self.down3 = UNetDown(128, 256)
#         self.down4 = UNetDown(256, 512, dropout=0.5)
#         self.down5 = UNetDown(512, 512, dropout=0.5)
#         self.down6 = UNetDown(512, 512, dropout=0.5)
#         self.down7 = UNetDown(512, 512, dropout=0.5)
#         self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

#         self.up1 = UNetUp(512, 512, dropout=0.5)
#         self.up2 = UNetUp(1024, 512, dropout=0.5)
#         self.up3 = UNetUp(1024, 512, dropout=0.5)
#         self.up4 = UNetUp(1024, 512, dropout=0.5)
#         self.up5 = UNetUp(1024, 256)
#         self.up6 = UNetUp(512, 128)
#         self.up7 = UNetUp(256, 64)

#         self.final = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ZeroPad2d((1, 0, 1, 0)),
#             nn.Conv2d(128, out_channels, 4, padding=1),
#             nn.Tanh(),
#         )

#     def forward(self, x):
#         # U-Net generator with skip connections from encoder to decoder
#         d1 = self.down1(x)
#         d2 = self.down2(d1)
#         d3 = self.down3(d2)
#         d4 = self.down4(d3)
#         d5 = self.down5(d4)
#         d6 = self.down6(d5)
#         d7 = self.down7(d6)
#         d8 = self.down8(d7)
#         u1 = self.up1(d8, d7)
#         u2 = self.up2(u1, d6)
#         u3 = self.up3(u2, d5)
#         u4 = self.up4(u3, d4)
#         u5 = self.up5(u4, d3)
#         u6 = self.up6(u5, d2)
#         u7 = self.up7(u6, d1)

#         return self.final(u7)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ConstantPad3d((1, 0, 1, 0, 1, 0), 0),
            # F.pad(pad=(1, 0, 1, 0, 1, 0)), # 3D padding #TODO: missing input variable
            nn.Conv3d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
