import torch
import torch.nn as nn
import torch.nn.functional as functional
import functools

# class NLayerDiscriminator(nn.Module):
#     """Defines a PatchGAN discriminator
#
#     From https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
#     """
#
#     def __init__(self, input_nc, ndf=4, n_layers=3, norm_layer=nn.BatchNorm3d):
#         """Construct a PatchGAN discriminator
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(NLayerDiscriminator, self).__init__()
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm3d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm3d
#
#
#         kernel_size = (4, 4, 4)
#         stride1 = (1, 1, 1)
#         stride2 = (2, 2, 2)
#         padding = (1, 1, 1)
#         sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kernel_size, stride=stride2, padding=5), nn.LeakyReLU(0.2, True)]
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             sequence += [
#                 nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=stride2, padding=padding, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]
#
#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         sequence += [
#             nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=stride1, padding=padding, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kernel_size, stride=stride1, padding=padding)]  # output 1 channel prediction map
#         self.model = nn.Sequential(*sequence)
#
#     def forward(self, input):
#         """Standard forward."""
#         return self.model(input)
#
#     def init_weights(self):
#         def _init_weights_(m):
#             if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
#             elif isinstance(m, nn.BatchNorm3d):
#                 nn.init.normal_(m.weight.data, 1.0, 0.02)
#                 nn.init.constant_(m.bias.data, 0.0)
#             elif isinstance(m, (nn.LeakyReLU, nn.ReLU, nn.Sequential, NLayerDiscriminator)):
#                 pass
#             else:
#                 raise NotImplementedError(f"How to initialize {type(m)}?")
#         self.apply(_init_weights_)
#
#
# class VoxNet(nn.Module):
#     """From https://github.com/MonteYang/VoxNet.pytorch.git"""
#     def __init__(self, n_classes, input_side):
#         super(VoxNet, self).__init__()
#         self.n_classes = n_classes
#         self.input_shape = (input_side, input_side, input_side)
#         self.feat = nn.Sequential(
#             nn.Conv3d(in_channels=2, out_channels=32, kernel_size=3, stride=1),
#             nn.BatchNorm3d(32),
#             nn.ReLU(),
#             # nn.Dropout(p=0.2),
#             nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3),
#             nn.BatchNorm3d(16),
#             nn.ReLU(),
#             nn.Conv3d(in_channels=16, out_channels=8, kernel_size=3),
#             nn.BatchNorm3d(8),
#             nn.ReLU(),
#             # nn.Dropout(p=0.3),
#         )
#         x = self.feat(torch.autograd.Variable(torch.rand((1, 2) + self.input_shape)))
#         dim_feat = 1
#         for n in x.size()[1:]:
#             dim_feat *= n
#
#         self.mlp = nn.Sequential(
#             nn.Linear(dim_feat, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             # nn.Dropout(p=0.4),
#             nn.Linear(64, self.n_classes),
#         )
#
#     def init_weights(self):
#         def _init_weights_(m):
#             if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
#                 nn.init.normal_(m.weight.data, 0.1, 0.02)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
#                 nn.init.constant_(m.bias.data, 0.0)
#             elif isinstance(m, nn.BatchNorm3d):
#                 nn.init.normal_(m.weight.data, 1, 0.02)
#                 nn.init.constant_(m.bias.data, 0.0)
#             elif isinstance(m, (nn.LeakyReLU, nn.ReLU, nn.Sequential, nn.MaxPool3d, nn.Dropout, VoxNet)):
#                 pass
#             else:
#                 raise NotImplementedError(f"How to initialize {type(m)}?")
#         self.apply(_init_weights_)
#
#
#     def forward(self, x):
#         x = self.feat(x)
#         x = x.view(x.size(0), -1)
#         x = self.mlp(x)
#         return x
#
# class AltDiscriminator(nn.Module):
#     def __init__(self):
#         super(AltDiscriminator, self).__init__()
#         self.model = nn.Linear(in_features=2, out_features=2)
#
#     def forward(self, x):
#         x = torch.mean(x, dim=(2, 3, 4))
#         out = self.model(x)
#         # print(f"{torch.mean(x, dim=0)=}")
#         # print(f"{torch.mean(out, dim=0)=}")
#         return out
#
#     def init_weights(self):
#         def _init_weights_(m):
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight.data, 0.2, 0.02)
#                 nn.init.constant_(m.bias.data, 0.0)
#         self.apply(_init_weights_)


class MseClassifier(nn.Module):
    """Inspired by https://github.com/MonteYang/VoxNet.pytorch.git"""

    def __init__(self):
        super(MseClassifier, self).__init__()
        input_side = 16
        self.classes = classes = 2

        convolution_features = 8
        kernel_side = 3
        extra_convolution_layers = 2

        self.convolutions = nn.Sequential(
            # Two input channels, one for dark matter, one for gas
            nn.Conv3d(
                in_channels=2,
                out_channels=convolution_features,
                kernel_size=kernel_side,
            ),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(),
            *(
                [
                    nn.Conv3d(
                        in_channels=convolution_features,
                        out_channels=convolution_features,
                        kernel_size=kernel_side,
                    ),
                    nn.ReLU(),
                ]
                * extra_convolution_layers
            )
        )

        total_features = (
            convolution_features
            * (  # number of features * number of voxels
                (input_side - (kernel_side - 1)) // 2  # First convolution and MaxPool
                - extra_convolution_layers * (kernel_side - 1)  # Extra convolutions
            )
            ** 3
        )  # side**3 to get the number of voxels

        self.classifier = nn.Sequential(
            nn.Linear(in_features=total_features, out_features=total_features),
            nn.ReLU(),
            nn.InstanceNorm1d(total_features),
            nn.Linear(in_features=total_features, out_features=classes),
            nn.Sigmoid(),
        )

    def init_weights(self):
        def _init_weights_(m):
            if isinstance(m, (nn.Linear, nn.Conv3d)):
                nn.init.normal_(m.weight.data, 0.1, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        self.apply(_init_weights_)

    def forward(self, input):
        convoluted = self.convolutions(input)
        return self.classifier(
            convoluted.view(
                (convoluted.size(0), 1, convoluted.size(1) * convoluted.size(2) ** 3)
            )
        )

    def loss(self, x, target):
        if target.shape == torch.Size([self.classes]):
            target = target.expand((x.size(0), 1, 2))
        output = self.forward(x)
        return functional.mse_loss(output, target)
