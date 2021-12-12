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

from .dataset import DatasetMock, mock_batch
from .classifier import MseClassifier


class SandboxNet:
    def __init__(self):
        self.tensor = tensor = torch.cuda.FloatTensor
        # self.epochs = None
        self.batch_size = batch_size = 512

        self.classifier = MseClassifier().cuda()
        num_parameters = sum([2**sum(map(math.log2, list(p.shape))) for p in self.classifier.parameters()])
        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=1 / num_parameters
        )

        print(f"Classifier: {num_parameters:.2e} parameters")

        self.classes = {
            0: tensor([1., 0.]),
            1: tensor([0., 1.]),
        }

        self.dataloader = torch.utils.data.DataLoader(
            DatasetMock(),
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
            with torch.no_grad():
                bta, btb = mock_batch(8)
                print(
                    f"Epoch {epoch:02d} | [",
                    ", ".join(f"{x:.2f}" for x in torch.mean(self.classifier.forward(bta.cuda()), dim=(0, 1)).cpu().numpy()),
                    "] [",
                    ", ".join(f"{x:.2f}" for x in torch.mean(self.classifier.forward(btb.cuda()), dim=(0, 1)).cpu().numpy()),
                    "]",
                ),

            for i, (bta, btb) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                loss_n = self.classifier.loss(bta.cuda(), self.classes[0])
                loss_n.backward()
                self.optimizer.step()

                self.optimizer.zero_grad()
                loss_n = self.classifier.loss(btb.cuda(), self.classes[1])
                loss_n.backward()
                self.optimizer.step()
