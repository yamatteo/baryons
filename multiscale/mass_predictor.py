import torch
import torch.nn as nn
import torch.nn.functional as functional
import functools

# class MassPredictor(nn.Module):
#     """Inspired by https://github.com/MonteYang/VoxNet.pytorch.git"""
#
#     def __init__(self):
#         super(MassPredictor, self).__init__()
#         input_side = 16
#         self.model = nn.Linear(1 + input_side*3, 1)
#
#     def forward(self, input):
#         mean = torch.mean(input, dim=(2, 3, 4)).unsqueeze(2)
#         mean_x = torch.mean(input, dim=(3, 4))
#         mean_y = torch.mean(input, dim=(2, 4))
#         mean_z = torch.mean(input, dim=(2, 3))
#         features = torch.cat((mean, mean_x, mean_y, mean_z), dim=2)
#         return self.model(features)

# class MassPredictor(nn.Module):
#     """Inspired by https://github.com/MonteYang/VoxNet.pytorch.git"""
#
#     def __init__(self):
#         super(MassPredictor, self).__init__()
#         input_side = 16
#         self.model = nn.Sequential(
#             nn.Linear(2, 1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, input):
#         var, mean = torch.var_mean(input, dim=(2, 3, 4))
#         return self.model(torch.stack((var, mean), dim=2))

class MassPredictor(nn.Module):
    """Inspired by https://github.com/MonteYang/VoxNet.pytorch.git"""

    def __init__(self):
        super(MassPredictor, self).__init__()
        input_side = 16
        self.model = nn.Sequential(
            nn.Linear(1, 1),
            # nn.Sigmoid(),
        )

    def forward(self, input):
        sum = torch.sum(input, dim=(2, 3, 4))
        return self.model(sum)

# class MassPredictor(nn.Module):
#     """Inspired by https://github.com/MonteYang/VoxNet.pytorch.git"""
#
#     def __init__(self):
#         super(MassPredictor, self).__init__()
#         input_side = 16
#         self.model = nn.Linear(1, 1)
#
#     def forward(self, input):
#         mean = torch.mean(input, dim=(2, 3, 4))
#         return self.model(mean)