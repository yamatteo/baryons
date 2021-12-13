import datetime
import logging
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.cuda
import torch.nn.functional as functional
import torch.utils.data

from .dataset import BlockworkDataset
from .classifier import MseClassifier


class SandboxNet:
    def __init__(self):
        self.tensor = tensor = torch.cuda.FloatTensor
        # self.epochs = None
        self.batch_size = batch_size = 64

        self.classifier = MseClassifier().cuda()
        num_parameters = sum([2 ** sum(map(math.log2, list(p.shape))) for p in self.classifier.parameters()])
        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=1 / (num_parameters**0.7)
        )

        print(f"Classifier: {num_parameters:.2e} parameters")

        self.classes = {
            0: tensor([1., 0.]),
            1: tensor([0., 1.]),
        }

        self.dataloader = torch.utils.data.DataLoader(
            BlockworkDataset(path="data/TNG100-3_SNAP099_MASS1.00e+12_5.00e+12_NGASMIN500/nvoxel_16/train"),
            batch_size=batch_size,
            num_workers=4,
        )

    def save_checkpoint(self, epoch):
        ...

    def load_checkpoint(self, epoch):
        ...

    def resume(self):
        ...

    def calculate_metrics(self, epoch, i, dm, rg, pg, gen_loss, dis_loss):
        ...

    def log_training_step(self, epoch, i, n_epochs, gen_loss, dis_loss, dpg, drg):
        ...

    def save_sample(self, dm, rg, pg, drg, dpg, epoch):
        ...

    def train(self, epochs):
        # self.epochs = epochs
        for epoch in range(epochs):
            for i, (bdm, bgs) in enumerate(self.dataloader):
                raise RuntimeError(f"{bdm.shape=}" + f"{bdm.names=}")
                real = self.tensor(torch.cat((bdm, bgs), dim=1).float().cuda())
                fake = self.tensor(torch.cat((bdm, torch.randn_like(bgs)+torch.mean(bgs)), dim=1).float().cuda())
                if i == 0:
                    with torch.no_grad():
                        print(
                            f"Epoch {epoch:02d} | [",
                            ", ".join(f"{x:.2f}" for x in
                                      torch.mean(self.classifier.forward(
                                          real
                                      ), dim=(0, 1)).cpu().numpy()),
                            "] [",
                            ", ".join(f"{x:.2f}" for x in
                                      torch.mean(self.classifier.forward(
                                          fake
                                      ), dim=(0, 1)).cpu().numpy()),
                            "]",
                        ),
                self.optimizer.zero_grad()
                loss_n = self.classifier.loss(real, self.classes[0])
                loss_n.backward()
                self.optimizer.step()

                self.optimizer.zero_grad()
                loss_n = self.classifier.loss(fake, self.classes[1])
                loss_n.backward()
                self.optimizer.step()
