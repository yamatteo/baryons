import torch
import torch.nn as nn
import functools

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator

    From https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
    """

    def __init__(self, input_nc, ndf=4, n_layers=3, norm_layer=nn.BatchNorm3d):
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


class VoxNet(nn.Module):
    """From https://github.com/MonteYang/VoxNet.pytorch.git"""
    def __init__(self, n_classes=2, input_side=32):
        super(VoxNet, self).__init__()
        self.n_classes = n_classes
        self.input_shape = (input_side, input_side, input_side)
        self.feat = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout(p=0.3),
        )
        x = self.feat(torch.autograd.Variable(torch.rand((1, 2) + self.input_shape)))
        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n

        self.mlp = nn.Sequential(
            nn.Linear(dim_feat, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, self.n_classes),
        )

    def init_weights(self):
        def _init_weights_(m):
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, (nn.BatchNorm3d, nn.Linear)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.LeakyReLU, nn.ReLU, nn.Sequential, nn.MaxPool3d, nn.Dropout, VoxNet)):
                pass
            else:
                raise NotImplementedError(f"How to initialize {type(m)}?")
        self.apply(_init_weights_)


    def forward(self, x):
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x