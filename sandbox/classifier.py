import torch
import torch.nn as nn
import torch.nn.functional as functional

# class NllClassifier(nn.Module):
#     def __init__(self):
#         super(NllClassifier, self).__init__()
#         self.lr = 1e-6
#
#         self.model = nn.Sequential(
#             nn.Linear(4, 16),
#             nn.Linear(16, 16),
#             nn.Linear(16, 16),
#             nn.Linear(16, 2),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, input):
#         return self.model(input)
#
#     def loss(self, input, target):
#         if isinstance(target, int):
#             target = torch.LongTensor([target]).expand(len(input))
#         # print(f"{input.shape=}")
#         # print(f"{target.shape=}")
#         output = self.forward(input)
#         # print(f"{output.shape=}")
#         # print(f"{functional.nll_loss(output, target).item()}")
#         # print((torch.sum(output) / len(output)).item())
#         return functional.nll_loss(output, target) + 0.3*(torch.sum(output) / len(output))
#
#     def learn(self, input, target):
#         if isinstance(target, int):
#             target = torch.LongTensor([target]).expand(len(input))
#         # output = self.forward(input)
#         # print(f"{output[0]=}")
#         loss = self.loss(input, target)
#         # print(f"{loss=}")
#         loss.backward()
#         with torch.no_grad():
#             for param in self.parameters():
#                 param.data -= self.lr * param.grad

class MseClassifier(nn.Module):
    def __init__(self):
        super(MseClassifier, self).__init__()
        input_side = 16
        classes = 2

        convolution_features = 8
        kernel_side = 3
        convolution_layers = 3

        self.convolutions = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=convolution_features, kernel_size=kernel_side),
            nn.AvgPool3d(kernel_size=2),
            nn.ReLU(),
            *([
                  nn.Conv3d(in_channels=convolution_features, out_channels=convolution_features, kernel_size=kernel_side),
                  nn.ReLU(),
                  # nn.InstanceNorm3d(convolution_features),
              ]*(convolution_layers-1))
        )
        total_features = convolution_features * ((input_side - (kernel_side - 1))//2 - (convolution_layers-1)*(kernel_side-1))**3
        self.classifier = nn.Sequential(
            nn.Linear(in_features=total_features, out_features=total_features),
            nn.ReLU(),
            nn.InstanceNorm1d(total_features),
            nn.Linear(in_features=total_features, out_features=classes),
            nn.Sigmoid(),
        )

    def forward(self, input):
        convoluted = self.convolutions(input)
        return self.classifier(convoluted.view((convoluted.size(0), 1, convoluted.size(1)*convoluted.size(2)**3)))

    def loss(self, input, target):
        if target.shape == torch.Size([2]):
            target = target.expand((len(input), 1, 2))
        # print(f"{input.shape=}")
        # print(f"{target.shape=}")
        output = self.forward(input)
        # print(f"{output.shape=}")
        # print(f"{functional.nll_loss(output, target).item()}")
        # print((torch.sum(output) / len(output)).item())
        return functional.mse_loss(output, target)