import torch
import torch.nn as nn
import torch.nn.functional as functional
import functools


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
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
                        # nn.Dropout(p=0.1),
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


class ClassifierTrainer:
    def __init__(self, cuda, batch_size, lr):
        self.classifier = Classifier()
        if cuda:
            self.classifier.cuda()
        self.seems_real = torch.tensor([1., 0.], device="cuda" if cuda else "cpu").repeat(batch_size, 1, 1)
        self.seems_fake = torch.tensor([0., 1.], device="cuda" if cuda else "cpu").repeat(batch_size, 1, 1)
        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=lr,
        )

    def __call__(self, dm, rg, pg):
        self.optimizer.zero_grad()
        dpg = self.classifier(torch.cat([dm, pg.detach()], dim=1))
        drg = self.classifier(torch.cat([dm, rg], dim=1))
        drg_loss = functional.mse_loss(drg, self.seems_real)
        dpg_loss = functional.mse_loss(dpg, self.seems_fake)
        (drg_loss + dpg_loss).backward()
        self.optimizer.step()

        dpg = self.classifier(torch.cat([dm, pg], dim=1))
        drg = self.classifier(torch.cat([dm, rg], dim=1))
        return functional.relu(
            functional.mse_loss(dpg, self.seems_real)
            - functional.mse_loss(drg, self.seems_real)
        )

class Quantifier(nn.Module):
    def __init__(self):
        super(Quantifier, self).__init__()
        input_side = 16

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
            nn.Linear(in_features=total_features, out_features=1),
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


class QuantifierTrainer:
    def __init__(self, cuda, batch_size, lr):
        self.quantifier = Quantifier()
        if cuda:
            self.quantifier.cuda()
        self.seems_real = torch.tensor([0.], device="cuda" if cuda else "cpu").repeat(batch_size, 1, 1)
        self.seems_fake = torch.tensor([1.], device="cuda" if cuda else "cpu").repeat(batch_size, 1, 1)
        self.optimizer = torch.optim.Adam(
            self.quantifier.parameters(),
            lr=lr,
        )

    def __call__(self, dm, rg, pg):
        self.optimizer.zero_grad()
        dpg = self.quantifier(torch.cat([dm, pg.detach()], dim=1))
        drg = self.quantifier(torch.cat([dm, rg], dim=1))
        drg_loss = functional.mse_loss(drg, self.seems_real)
        dpg_loss = functional.mse_loss(dpg, self.seems_fake)
        (drg_loss + dpg_loss).backward()
        self.optimizer.step()

        dpg = self.quantifier(torch.cat([dm, pg], dim=1))
        drg = self.quantifier(torch.cat([dm, rg], dim=1))
        return functional.relu(
            functional.mse_loss(dpg, self.seems_real)
            - functional.mse_loss(drg, self.seems_real)
        )

class MseTrainer:
    def __init__(self):
        pass

    def __call__(self, dm, rg, pg):
        return functional.mse_loss(rg, pg)
