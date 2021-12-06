import argparse
import datetime
import glob
import itertools
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from importlib import reload
from argparse import Namespace

from gan.dataset import Dataset3D

from gan.dynamic_models import Discriminator, Generator, weights_init_normal
from gan.discriminator import Discriminator as AltDiscriminator


def setup_gan(opt):
    logging.info(f"Setting up GAN")

    tensor_type = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

    generator = Generator(
        in_channels=opt.channels,
        out_channels=opt.channels,
        num_filters=opt.num_filters,
        depth=opt.generator_depth,
    )

    discriminator = AltDiscriminator(3, 2, 0, "leaky", "instance", tensor_type)

    if opt.cuda is True:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
    )

    transformations = [
        # CP: transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        # transforms.ToTensor(),
        # CP: transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.5), (0.5)),  # CP: one value for the mean for each channel (here one channel and two images)
    ]

    dataloaders = {
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

    # Adversarial ground truths
    # shape [batch_size, channels, (n_voxel // patch_side) x 3]
    gt_valid = tensor_type(
        np.ones((opt.batch_size, opt.channels, *([opt.nvoxel // opt.patch_side] * 3)))
    )
    gt_fake = tensor_type(
        np.zeros((opt.batch_size, opt.channels, *([opt.nvoxel // opt.patch_side] * 3)))
    )

    if opt.criterion_gan == "mse":
        criterion_gan = torch.nn.MSELoss()
    else:
        criterion_gan = None

    if opt.criterion_pixelwise == "l1":
        criterion_pixelwise = torch.nn.L1Loss()
    else:
        criterion_pixelwise = None

    return Namespace(
        criterion_gan=criterion_gan,
        criterion_pixelwise=criterion_pixelwise,
        d_optimizer=d_optimizer,
        dataloaders=dataloaders,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        generator=generator,
        gt_fake=gt_fake,
        gt_valid=gt_valid,
        tensor_type=tensor_type,
    )


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
    p2p3d = setup_gan(opt)
    metrics = pd.DataFrame(columns=["epoch", "time", "loss_g", "loss_d"])

    start_from_epoch = get_last_checkpoint(opt)
    if opt.may_resume is True and start_from_epoch > 0:
        load_checkpoint(start_from_epoch, opt, p2p3d)
    else:
        # Initialize weights
        start_from_epoch = 0
        p2p3d.generator.apply(weights_init_normal)
        p2p3d.discriminator.apply(weights_init_normal)

    init_time = time.time()

    logging.info(f"Training begins...")

    for epoch in range(start_from_epoch, opt.n_epochs):

        for i, batch in enumerate(p2p3d.dataloaders["train"]):  # batch is a dictionary

            logging.debug(f"Starting batch {i}")

            # Model inputs - shape [batch_size, channels, n_voxel, n_voxel, n_voxel]
            real_dm = batch["DM"].type(p2p3d.tensor_type)
            real_gas = batch["GAS"].type(p2p3d.tensor_type)

            logging.debug(f"Defined real_dm and real_gas - shape {real_dm.shape}")

            # ------------------
            #  Train Generator
            # ------------------

            p2p3d.g_optimizer.zero_grad()

            pred_gas = p2p3d.generator(real_dm)
            loss_gen = p2p3d.discriminator.evaluate(real_dm, pred_gas)

            loss_gen.backward()
            p2p3d.g_optimizer.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            p2p3d.d_optimizer.zero_grad()

            # pred_gas = p2p3d.generator(real_dm)
            loss_dis = p2p3d.discriminator.loss(real_dm, real_gas, pred_gas.detach())

            loss_dis.backward()
            p2p3d.d_optimizer.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(p2p3d.dataloaders["train"]) + i + 1
            batches_left = opt.n_epochs * len(p2p3d.dataloaders["train"]) - batches_done
            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - init_time) / batches_done
            )

            metrics = metrics.append(
                {
                    "epoch": epoch + 1,
                    "time": float(time.time() - init_time),
                    "loss_gen": loss_gen.item(),
                    "loss_dis": loss_dis.item(),
                },
                ignore_index=True,
            )

            logging.info(
                f" [Epoch {epoch + 1:02d}/{opt.n_epochs:02d}]"
                + f" [Batch {i + 1}/{len(p2p3d.dataloaders['train'])}]"
                + f" [D loss: {loss_dis.item():.3e}]"
                + f" [G loss: {loss_gen.item():.3e}]"
                + f" ETA: {str(time_left).split('.')[0]}"
            )

            # If at sample interval save image TODO save numpy report or something useful
            # if batches_done % opt.sample_interval == 0:
            #     save_report(
            #         real_dm.detach(),
            #         real_gas.detach(),
            #         fake_gas.detach(),
            #         database_name=f"{opt.sim_name}__{opt.mass_range}__{opt.n_voxel}",
            #         root=opt.root,
            #         epoch=epoch,
            #         batch=i,
            #     )

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            save_checkpoint(epoch, opt, p2p3d)

    # TODO see if this is helpful or necessary to free up space in the GPU
    del p2p3d
    del opt

    return metrics
