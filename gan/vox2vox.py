import argparse
import datetime
import glob
import itertools
import logging
import os
import pprint
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from importlib import reload
from argparse import Namespace

from gan.dataset import Dataset3D

from gan.dynamic_models import Generator, weights_init_normal
import gan.discriminator as discriminators
from gan.generator import make_generator
from visualization.report import save_sample


def make_metric(name):
    if name == "mse":
        def f(real_dm, real_gas, pred_gas):
            return F.mse_loss(pred_gas, real_gas)

        return f
    elif name == "l1":
        def f(real_dm, real_gas, pred_gas):
            return F.l1_loss(pred_gas, real_gas)

        return f


class Vox2Vox:
    def __init__(self, opt):
        if isinstance(opt, dict):
            opt = Namespace(**opt)

        self.tensor_type = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

        self.generator = make_generator(opt)

        if opt.discriminator == "original":
            self.discriminator = discriminators.Original(opt)
        elif opt.discriminator == "monopatch":
            self.discriminator = discriminators.MonoPatch(3, 2, 0, "leaky", "instance", self.tensor_type)
        elif opt.discriminator == "onlymse":
            self.discriminator = discriminators.OnlyMSE(opt)
        elif opt.discriminator == "multimse":
            self.discriminator = discriminators.MultiMSE(opt)

        if opt.cuda is True:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
        )

        transformations = [
            # CP: transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
            # transforms.ToTensor(),
            # CP: transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Normalize((0.5), (0.5)),  # CP: one value for the mean for each channel (here one channel and two images)
        ]

        self.dataloaders = {
            mode: DataLoader(
                Dataset3D(
                    dataset_path=opt.dataset_path,
                    sim_name=opt.sim_name,
                    snap_num=opt.snap_num,
                    mass_min=opt.mass_min,
                    mass_max=opt.mass_max,
                    n_gas_min=opt.n_gas_min,
                    data_path=opt.data_path,
                    nvoxel=opt.nvoxel,
                    mode=mode,
                    transforms=transformations,
                ),
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_cpu,
                drop_last=True,
            )
            for mode in ("train", "valid", "test")
        }
        self.metrics = {metric: make_metric(metric) for metric in opt.metrics}


def save_checkpoint(epoch, opt, v2v):
    torch.save(
        v2v.generator.state_dict(),
        os.path.join(opt.run_path, f"generator_{opt.run_index}_{epoch}.pth"),
    )
    torch.save(
        v2v.discriminator.state_dict(),
        os.path.join(opt.run_path, f"discriminator_{opt.run_index}_{epoch}.pth"),
    )


def load_checkpoint(epoch, opt, v2v):
    logging.info(f"Loading pretrained models from epoch {epoch}")
    v2v.generator.load_state_dict(
        torch.load(os.path.join(opt.run_path, f"generator_{opt.run_index}_{epoch}.pth"))
    )
    v2v.discriminator.load_state_dict(
        torch.load(
            os.path.join(opt.run_path, f"discriminator_{opt.run_index}_{epoch}.pth")
        )
    )


def get_last_checkpoint(opt):
    generators = glob.glob(
        os.path.join(opt.run_path, f"generator_{opt.run_index}_*.pth")
    )
    prestrip = lambda s: s.removeprefix(
        os.path.join(opt.run_path, f"generator_{opt.run_index}_")
    )
    poststrip = lambda s: s.removesuffix(".pth")
    if generators:
        return max(int(poststrip(prestrip(filename))) for filename in generators)
    else:
        return -1



def single_run(opt: Namespace):
    vv = Vox2Vox(opt)
    metrics = pd.DataFrame(columns=["epoch", "time", "loss_gen", "loss_dis", *vv.metrics.keys()])

    start_from_epoch = get_last_checkpoint(opt)
    if opt.may_resume is True and start_from_epoch > 0:
        load_checkpoint(start_from_epoch, opt, vv)
    else:
        # Initialize weights
        start_from_epoch = 0
        vv.generator.apply(weights_init_normal)
        vv.discriminator.apply(weights_init_normal)

    init_time = time.time()

    for epoch in range(start_from_epoch, opt.n_epochs):

        for i, batch in enumerate(vv.dataloaders["train"]):  # batch is a dictionary

            logging.debug(f"Starting batch {i}")

            # Model inputs - shape [batch_size, channels, n_voxel, n_voxel, n_voxel]
            real_dm = batch["DM"].type(vv.tensor_type)
            real_gas = batch["GAS"].type(vv.tensor_type)

            logging.debug(f"Defined real_dm and real_gas - shape {real_dm.shape}")

            # ------------------
            #  Train Generator
            # ------------------

            vv.g_optimizer.zero_grad()

            pred_gas = vv.generator(real_dm)
            loss_gen = vv.discriminator.evaluate(real_dm, real_gas, pred_gas)

            loss_gen.backward()
            vv.g_optimizer.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            vv.d_optimizer.zero_grad()

            # pred_gas = p2p3d.generator(real_dm)
            loss_dis = vv.discriminator.loss(real_dm, real_gas, (pred_gas).detach())

            loss_dis.backward()
            vv.d_optimizer.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(vv.dataloaders["train"]) + i + 1
            batches_left = opt.n_epochs * len(vv.dataloaders["train"]) - batches_done
            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - init_time) / batches_done
            )

            metrics = metrics.append(
                dict(
                    {
                        "epoch": epoch + 1,
                        "time": float(time.time() - init_time),
                        "loss_gen": loss_gen.item(),
                        "loss_dis": loss_dis.item(),
                    },
                    **{name: metric(real_dm, real_gas, pred_gas.detach()).item() for name, metric in vv.metrics.items()}
                ),
                ignore_index=True,
            )

            logging.info(
                f" [Epoch {epoch + 1:02d}/{opt.n_epochs:02d}]"
                + f" [Batch {i + 1}/{len(vv.dataloaders['train'])}]"
                + f" [D loss: {loss_dis.item():.3e}]"
                + f" [G loss: {loss_gen.item():.3e}]"
                + f" ETA: {str(time_left).split('.')[0]}"
            )

            if epoch % opt.sample_interval == 0 and i == 0:
                save_sample(
                    epoch,
                    opt,
                    (real_dm).detach(),
                    (real_gas).detach(),
                    (pred_gas).detach(),
                )

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            save_checkpoint(epoch, opt, vv)

    return metrics
