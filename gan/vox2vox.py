import argparse
import datetime
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
from visualization.report import save_report


def setup_gan(opt):
    logging.info(f"Setting up GAN")

    tensor_type = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

    generator = Generator(
        in_channels=opt.channels,
        out_channels=opt.channels,
        num_filters=opt.num_filters,
        depth=opt.generator_depth,
    )

    discriminator = Discriminator(channels=opt.channels)

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


def single_run(opt: Namespace):
    p2p3d = setup_gan(opt)
    metrics = pd.DataFrame(columns=["epoch", "time", "loss_g", "loss_d"])

    if opt.skip_to_epoch == 0:
        # Initialize weights
        p2p3d.generator.apply(weights_init_normal)
        p2p3d.discriminator.apply(weights_init_normal)
    else:
        # Load pretrained models
        p2p3d.generator.load_state_dict(
            torch.load(
                os.path.join(
                    "saved_models",
                    f"{opt.sim_name}_SNAP{opt.snap_num:03d}_MASS{opt.mass_min:.2e}_{opt.mass_max:.2e}_NGASMIN{opt.n_gas_min}",
                    f"nvoxel_{opt.nvoxel}",
                    f"generator_{opt.skip_to_epoch}.pth"
                )
            )
        )
        p2p3d.discriminator.load_state_dict(
            torch.load(
                os.path.join(
                    "saved_models",
                    f"{opt.sim_name}_SNAP{opt.snap_num:03d}_MASS{opt.mass_min:.2e}_{opt.mass_max:.2e}_NGASMIN{opt.n_gas_min}",
                    f"nvoxel_{opt.nvoxel}",
                    f"discriminator_{opt.skip_to_epoch}.pth"
                )
            )
        )

    init_time = time.time()

    logging.info(f"Training begins...")

    for epoch in range(opt.skip_to_epoch, opt.n_epochs):

        for i, batch in enumerate(p2p3d.dataloaders["train"]):  # batch is a dictionary

            logging.debug(f"Starting batch {i}")

            # Model inputs - shape [batch_size, channels, n_voxel, n_voxel, n_voxel]
            real_dm = batch["DM"].type(p2p3d.tensor_type)
            real_gas = batch["GAS"].type(p2p3d.tensor_type)

            logging.debug(f"Defined real_dm and real_gas - shape {real_dm.shape}")

            # ------------------
            #  Train Generator
            # ------------------

            logging.debug("Starting training Generator")

            p2p3d.g_optimizer.zero_grad()

            # GAN loss
            fake_gas = p2p3d.generator(real_dm)
            logging.debug(f"Generated fake_gas - shape {fake_gas.shape = }")

            pred_fake = p2p3d.discriminator(fake_gas, real_dm)
            logging.debug(
                f"Made fake prediction of discriminator - shape {pred_fake.shape =}"
            )

            loss_gan = p2p3d.criterion_gan(pred_fake, p2p3d.gt_valid)
            logging.debug(f"{loss_gan = }")

            loss_pixel = p2p3d.criterion_pixelwise(fake_gas, real_gas)
            logging.debug(f"{loss_pixel = }")

            loss_g = loss_gan + opt.lambda_pixel * loss_pixel
            logging.debug(f"{loss_g = }")

            loss_g.backward()
            logging.debug("loss_G.backward() done")

            p2p3d.g_optimizer.step()
            logging.debug("optimizer_G.step() done")

            # ---------------------
            #  Train Discriminator
            # ---------------------

            if epoch == 0:
                logging.debug("Starting training Discriminator")

            p2p3d.d_optimizer.zero_grad()

            # Real loss
            pred_real = p2p3d.discriminator(real_gas, real_dm)
            loss_real = p2p3d.criterion_gan(pred_real, p2p3d.gt_valid)
            logging.debug(
                f"Generated pred_real (shape {pred_real.shape}) - loss {loss_real:0.2}"
            )

            # Fake loss
            pred_fake = p2p3d.discriminator(
                fake_gas.detach(), real_dm
            )  # CP: Detach from gradient calculation
            loss_fake = p2p3d.criterion_gan(pred_fake, p2p3d.gt_fake)
            logging.debug(
                f"Generated pred_fake (shape {pred_fake.shape}) - loss {loss_fake:0.2}"
            )

            # Total loss
            loss_d = 0.5 * (loss_real + loss_fake)

            loss_d.backward()
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

            metrics = metrics.append({
                "epoch": epoch + 1,
                "time": float(time.time() - init_time),
                "loss_d": loss_d.item(),
                "loss_g": loss_g.item()
            }, ignore_index=True)

            logging.info(
                f" [Epoch {epoch + 1:02d}/{opt.n_epochs:02d}]"
                + f" [Batch {i + 1}/{len(p2p3d.dataloaders['train'])}]"
                + f" [D loss: {loss_d.item():.3e}]"
                + f" [G loss: {loss_g.item():.3e}, pixel: {loss_pixel.item():.3e}, adv: {loss_gan.item():.3e}]"
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
            dataset_name = f"{opt.sim_name}__{opt.mass_range}__{opt.n_voxel}"
            os.makedirs(
                os.path.join(opt.root, "saved_models", dataset_name), exist_ok=True
            )
            torch.save(
                p2p3d.generator.state_dict(),
                f"saved_models/{dataset_name}/generator_{epoch}.pth",
            )
            torch.save(
                p2p3d.discriminator.state_dict(),
                f"saved_models/{dataset_name}/discriminator_{epoch}.pth",
            )

    # TODO see if this is helpful or necessary to free up space in the GPU
    del p2p3d
    del opt

    return metrics
