import torch.nn as nn


class ShallowGenerator(nn.Module):
    def __init__(self):
        super(ShallowGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=4,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=4,
                out_channels=2,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(2),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=2,
                out_channels=1,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                bias=False,
            ),
        )

    def init_weights(self):
        def __init_weights(m):
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight.data, 0.0, 1.0)

        self.apply(__init_weights)

    def forward(self, input):
        return self.model(input)
