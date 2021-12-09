import torch.nn as nn
import functools

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator

    From https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
    """

    def __init__(self, input_nc, ndf=32, n_layers=3, norm_layer=nn.BatchNorm3d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kernel_size = (4, 4, 4)
        stride1 = (1, 1, 1)
        stride2 = (2, 2, 2)
        padding = (1, 1, 1)
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kernel_size, stride=stride2, padding=padding), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=stride2, padding=padding, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=stride1, padding=padding, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kernel_size, stride=stride1, padding=padding)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

    def init_weights(self):
        def _init_weights_(m):
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.LeakyReLU, nn.ReLU, nn.Sequential, NLayerDiscriminator)):
                pass
            else:
                raise NotImplementedError(f"How to initialize {type(m)}?")
        self.apply(_init_weights_)