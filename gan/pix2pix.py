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

# from gan.models import UNet, Discriminator
from gan.dynamic_models import Discriminator, Generator, weights_init_normal
from visualization.report import save_report


def setup_gan(opt):
    logging.info(f"Setting up GAN: {opt}")

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
                sim_name=opt.sim_name,
                mass_range=opt.mass_range,
                n_voxel=opt.n_voxel,
                mode="train",
                transforms=transformations,
            ),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
            drop_last=True,
        )
        for mode in ("train", "valid")
    }

    # Adversarial ground truths
    # shape [batch_size, channels, n_voxel // patch_side, n_voxel // patch_side, n_voxel // patch_side]
    gt_valid = tensor_type(
        np.ones((opt.batch_size, opt.channels, *([opt.n_voxel // opt.patch_side] * 3)))
    )
    gt_fake = tensor_type(
        np.zeros((opt.batch_size, opt.channels, *([opt.n_voxel // opt.patch_side] * 3)))
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
    reload(logging)  # Avoid duplicate logging in JupyterLab
    logging.basicConfig(
        filename="last_run.log",
        filemode=opt.log_mode,
        format="%(levelname)s: %(message)s",
        level=logging.DEBUG if opt.log_level == 'debug' else logging.INFO,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(console)
    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("parso.python.diff").disabled = True

    p2p3d = setup_gan(opt)
    metric = pd.DataFrame(columns=["epoch", "time", "loss_g", "loss_d"])

    if opt.skip_to_epoch == 0:
        # Initialize weights
        p2p3d.generator.apply(weights_init_normal)
        p2p3d.discriminator.apply(weights_init_normal)
    else:
        # Load pretrained models
        p2p3d.generator.load_state_dict(
            torch.load(
                f"saved_models/{opt.dataset_name}/generator_{opt.skip_to_epoch}.pth"
            )
        )
        p2p3d.discriminator.load_state_dict(
            torch.load(
                f"saved_models/{opt.dataset_name}/discriminator_{opt.skip_to_epoch}.pth"
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

            metric = metric.append({
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

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                save_report(
                    real_dm.detach(),
                    real_gas.detach(),
                    fake_gas.detach(),
                    database_name=f"{opt.sim_name}__{opt.mass_range}__{opt.n_voxel}",
                    root=opt.root,
                    epoch=epoch,
                    batch=i,
                )

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

    del p2p3d
    del opt
    return metric


def many_runs(single_opts: dict, multi_opts: dict) -> pd.DataFrame:
    global_metrics = pd.DataFrame(columns=["epoch", "time", "loss_g", "loss_d"])
    multi_labels = tuple(multi_opts.keys())
    for possible_opt in itertools.product(*multi_opts.values()):
        torch.cuda.empty_cache()
        extra_opt = dict(zip(multi_labels, possible_opt))
        opt = dict(single_opts, **extra_opt)
        try:
            metrics = single_run(Namespace(**opt)).assign(**extra_opt)
            global_metrics = global_metrics.append(metrics, ignore_index=True)
        except (RuntimeError, Exception):
            logging.exception("Bad run...")
    return global_metrics


def test():
    base_opt = Namespace(
        sim_name="TNG300-1",
        mass_range="MASS_1.00e+12_5.00e+12_MSUN",
        n_voxel=128,
        channels=1,

        generator_depth=5,
        patch_side=16,
        num_filters=4,
        criterion_gan=torch.nn.MSELoss(),
        criterion_pixelwise=torch.nn.L1Loss(),
        lambda_pixel=100,

        skip_to_epoch=0,
        n_epochs=20,
        batch_size=2,

        lr=0.0002,
        b1=0.5,
        b2=0.999,
        decay_epoch=100,

        root=os.getcwd(),
        n_cpu=8,
        sample_interval=5,
        checkpoint_interval=-1,
        log_level=logging.INFO,
        cuda=True,
    )
    gm = many_runs(base_opt, {
        "generator_depth": (4, 5, 6),
        "num_filters": (2, 4, 8),
    })
    print(gm)

if __name__ == "__main__":
    test()
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--root", type=str, default=os.path.curdir, help="folder where data/ is"
    # )
    # parser.add_argument("--sim_name", type=str, default="TNG300-1")
    # parser.add_argument("--mass_range", type=str, default="MASS_1.00e+12_5.00e+12_MSUN")
    # parser.add_argument(
    #     "--n_voxel", type=int, default=128, help="number of voxels set for images"
    # )
    # parser.add_argument(
    #     "--skip_to_epoch", type=int, default=0, help="epoch to start training from"
    # )
    # parser.add_argument(
    #     "--n_epochs", type=int, default=20, help="number of epochs of training"
    # )
    # parser.add_argument(
    #     "--dataset_name", type=str, default="facades", help="name of the dataset"
    # )
    # parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    # parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    # parser.add_argument(
    #     "--b1",
    #     type=float,
    #     default=0.5,
    #     help="adam: decay of first order momentum of gradient",
    # )
    # parser.add_argument(
    #     "--b2",
    #     type=float,
    #     default=0.999,
    #     help="adam: decay of first order momentum of gradient",
    # )
    # parser.add_argument(
    #     "--decay_epoch",
    #     type=int,
    #     default=100,
    #     help="epoch from which to start lr decay",
    # )
    # parser.add_argument(
    #     "--generator_depth",
    #     type=int,
    #     default=5,
    #     help="depth of the generator architecture",
    # )
    # parser.add_argument(
    #     "--n_cpu",
    #     type=int,
    #     default=8,
    #     help="number of cpu threads to use during batch generation",
    # )
    # # parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    # # parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    # parser.add_argument(
    #     "--channels", type=int, default=1, help="number of image channels"
    # )
    # parser.add_argument("--patch_side", type=int, default=16)
    # parser.add_argument(
    #     "--sample_interval",
    #     type=int,
    #     default=5,
    #     help="interval between sampling of images from generators",
    # )
    # parser.add_argument(
    #     "--checkpoint_interval",
    #     type=int,
    #     default=-1,
    #     help="interval between model checkpoints",
    # )
    # parser.add_argument(
    #     "--log_level",
    #     type=int,
    #     default=logging.INFO,
    # )
    # parser.add_argument(
    #     "--cuda",
    #     type=bool,
    #     default=True if torch.cuda.is_available() else False,
    # )
    # parser.add_argument(
    #     "--num_filters",
    #     type=int,
    #     default=4,
    # )
    # parser.add_argument(
    #     "--lambda_pixel",
    #     type=int,
    #     default=100,
    # )
    #
    # opt = parser.parse_args()
    #
    # vars(opt)["criterion_gan"] = torch.nn.MSELoss()
    # vars(opt)["criterion_pixelwise"] = torch.nn.L1Loss()
    #
    # single_run(opt)
