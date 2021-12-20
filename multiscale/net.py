import datetime
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
import torch.utils.data

from .dataset import BlockworkDataset
from .generator import UnetGenerator
from .classifier import MseClassifier
from preprocessing.preprocess import preprocess
from torch.utils.tensorboard import SummaryWriter

# class Classifier:
#     def __init__(self, opt):
#         preprocess(opt)
#
#         self.run_path = Path(opt.output_path)
#         self.batch_size = opt.batch_size
#         self.tensor_type = torch.cuda.FloatTensor
#
#         self.gen = UnetGenerator(1, 1, num_downs=4, use_dropout=True).cuda()
#         self.gen_opt = torch.optim.Adam(
#             self.gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
#         )
#         # self.dis = NLayerDiscriminator(input_nc=2).cuda()
#         self.dis = AltDiscriminator().cuda()
#         self.dis_opt = torch.optim.Adam(
#             self.gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
#         )
#         self.seems_real = self.tensor_type([1., 0.]).view((1, 2)).repeat(self.batch_size, 1)
#         self.seems_fake = self.tensor_type([0., 1.]).view((1, 2)).repeat(self.batch_size, 1)
#
#         self.dataloader = torch.utils.data.DataLoader(
#             BlockworkDataset(
#                 path=(
#                         Path(opt.data_path)
#                         / f"{opt.sim_name}_SNAP{opt.snap_num:03d}_MASS{opt.mass_min:.2e}_{opt.mass_max:.2e}_NGASMIN{opt.n_gas_min}"
#                         / f"nvoxel_{opt.nvoxel}"
#                         / "train"
#                 ),
#             ),
#             batch_size=opt.batch_size,
#             shuffle=True,
#             num_workers=opt.n_cpu,
#             drop_last=True,
#         )
#         self.metrics_functions = {
#             "mse": (lambda dm, rg, pg: functional.mse_loss(pg, rg)),
#             "l1": (lambda dm, rg, pg: functional.l1_loss(pg, rg)),
#         }
#         # self.running_metrics = pd.DataFrame(
#         #     columns=[
#         #         "epoch",
#         #         "time",
#         #         "gen_loss",
#         #         "dis_loss",
#         #         *self.metrics_functions.keys(),
#         #     ]
#         # )
#         self.writer = SummaryWriter()
