# Code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
import logging

import torch
import torch.nn as nn
import functools

logger = logging.getLogger("baryons")

class UnetGenerator(nn.Module):
    """Create a Unet-based generator.

    From https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
    Modified for small 3D input.
    """

    def __init__(
        self,
        input_nc,
        output_nc,
        num_downs,
        ngf=16,
        norm_layer=nn.BatchNorm3d,
        use_dropout=False,
    ):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 4,
            ngf * 4,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )  # add the innermost layer
        for i in range(num_downs - 4):  # add intermediate layers with ngf * 4 filters
            unet_block = UnetSkipConnectionBlock(
                ngf * 4,
                ngf * 4,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        # gradually reduce the number of filters from ngf * 4 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

    def init_weights(self):
        def _init_weights_(m):
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, (nn.BatchNorm3d,)):
                nn.init.normal_(m.weight.data, 0.1, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.LeakyReLU, nn.ReLU, nn.Tanh, nn.Dropout, nn.InstanceNorm3d, nn.Sequential, UnetSkipConnectionBlock, UnetGenerator)):
                pass
            else:
                raise NotImplementedError(f"How to initialize {type(m)}?")
        self.apply(_init_weights_)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|

    From https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
    Modified for 3D input.
    """

    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm3d,
        use_dropout=False,
    ):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1
            )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # print(f"input {x.shape=}")
        if self.outermost:
            output = self.model(x)
            # print(f"{output.shape=}")
            return output
        else:  # add skip connections
            output = torch.cat([x, self.model(x)], 1)
            # print(f"{output.shape=}")
            return output


class SUNetOutermost(nn.Module):
    def __init__(self, input_nc, inner_nc, outer_nc, submodule):
        super(SUNetOutermost, self).__init__()

        self.model = nn.Sequential(
            nn.Conv3d(input_nc, inner_nc, kernel_size=3, bias=False),
            submodule,
            nn.ReLU(True),
            nn.ConvTranspose3d(inner_nc * 2, outer_nc, kernel_size=3, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        # logger.debug(f"outermost {x.shape=}")
        output = self.model(x)
        # logger.debug(f"outermost {output.shape=}")
        return output

class SUNetIntermediate(nn.Module):
    def __init__(self, nc, submodule,):
        super(SUNetIntermediate, self).__init__()

        self.model = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(nc, nc * 2, kernel_size=3, bias=False),
            nn.InstanceNorm3d(nc),
            submodule,
            nn.ReLU(True),
            nn.ConvTranspose3d(nc * 4, nc, kernel_size=3, bias=False, ),
            nn.InstanceNorm3d(nc),
            nn.Dropout(0.5),
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
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(nc, nc, kernel_size=3, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose3d(nc, nc, kernel_size=3, bias=False),
            nn.InstanceNorm3d(nc),
        )

    def forward(self, x):
        # logger.debug(f"innermost: {x.shape=}")
        output = torch.cat([x, self.model(x)], 1)
        # logger.debug(f"innermost: {output.shape=}")
        return output

class SUNet(nn.Module):
    def __init__(self, features, levels):
        super(SUNet, self).__init__()
        block = SUNetInnermost(
            nc=2 ** levels * features,
        )
        for lvl in reversed(range(levels)):
            block = SUNetIntermediate(2**lvl * features, block)
        self.model = SUNetOutermost(
            input_nc=1,
            inner_nc=features,
            outer_nc=1,
            submodule=block,
        )

    def forward(self, x):
        return self.model(x)