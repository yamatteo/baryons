import argparse
import datetime
import logging
import os
import sys
import time

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from types import SimpleNamespace

from gan.dataset import Dataset3D
# from gan.models import UNet, Discriminator
from gan.dynamic_models import Discriminator, Generator, weights_init_normal
from visualization import heatmap_plot, select_slice


def setup(channels, n_voxel, patch_side, generator_depth, lr, b1, b2, sim_name, mass_range, batch_size,
          n_cpu, skip_to_epoch=None, **kwargs):
    assert n_voxel % patch_side == 0, "parameter n_voxel should be a multiple of parameter patch_side"
    logging.info('Setting up gan...')

    cuda = True if torch.cuda.is_available() else False
    if cuda is False:
        logging.warning("Running without cuda!")

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    generator = Generator(in_channels=channels, out_channels=channels, num_filters=4, depth=generator_depth)
    discriminator = Discriminator(channels=channels)

    # Loss functions
    criterion_gan = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = 100

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_gan.cuda()
        criterion_pixelwise.cuda()

    if skip_to_epoch == 0:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
    else:
        # Load pretrained models
        dataset_name = f"{sim_name}__{mass_range}__{n_voxel}"
        generator.load_state_dict(
            torch.load(f"saved_models/{dataset_name}/generator_{skip_to_epoch}.pth")
        )
        discriminator.load_state_dict(
            torch.load(f"saved_models/{dataset_name}/discriminator_{skip_to_epoch}.pth")
        )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # Configure dataloaders
    transforms_ = [
        # CP: transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        # transforms.ToTensor(),
        # CP: transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.5), (0.5)),  # CP: one value for the mean for each channel (here one channel and two images)
    ]

    dataloader = DataLoader(
        Dataset3D(
            sim_name=sim_name,
            mass_range=mass_range,
            n_voxel=n_voxel,
            mode="train",
            transforms=transforms_,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        Dataset3D(
            sim_name=sim_name,
            mass_range=mass_range,
            n_voxel=n_voxel,
            mode="valid",
            transforms=transforms_,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        drop_last=True,
    )

    # Adversarial ground truths
    # shape [batch_size, channels, n_voxel // patch_side, n_voxel // patch_side, n_voxel // patch_side]
    gt_valid = Tensor(np.ones((batch_size, channels, *([n_voxel // patch_side] * 3))))
    gt_fake = Tensor(np.zeros((batch_size, channels, *([n_voxel // patch_side] * 3))))

    return SimpleNamespace(
        cuda=cuda,
        generator=generator,
        discriminator=discriminator,
        criterion_gan=criterion_gan,
        criterion_pixelwise=criterion_pixelwise,
        lambda_pixel=lambda_pixel,
        optimizer_D=optimizer_D,
        optimizer_G=optimizer_G,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        gt_fake=gt_fake,
        gt_valid=gt_valid,
        Tensor=Tensor,
    )


def run(opt):
    logging.basicConfig(
        filename="debug.log",
        filemode="w",
        format='%(levelname)s: %(message)s',
        level=logging.DEBUG,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('parso.python.diff').disabled = True

    logging.info(opt)
    gan = setup(**vars(opt))

    init_time = time.time()

    logging.info(f"Training begins...")

    for epoch in range(opt.skip_to_epoch, opt.n_epochs):

        for i, batch in enumerate(gan.dataloader):  # batch is a dictionary

            if epoch == 0:
                logging.debug(f"Starting batch {i}")

            # Model inputs - shape [batch_size, channels, n_voxel, n_voxel, n_voxel]
            real_dm = Variable(batch["DM"].type(gan.Tensor))
            real_gas = Variable(batch["GAS"].type(gan.Tensor))

            if epoch == 0:
                logging.debug(f"Defined real_dm and real_gas - shape {real_dm.shape}")

            # ------------------
            #  Train Generator
            # ------------------

            if epoch == 0:
                logging.debug("Starting training Generator")

            gan.optimizer_G.zero_grad()

            # GAN loss
            if epoch == 0 and i == 0:
                fake_gas = gan.generator.verbose_forward(real_dm, depth=opt.generator_depth)
            else:
                fake_gas = gan.generator(real_dm)
            if epoch == 0:
                logging.debug(f"Generated fake_gas - shape {fake_gas.shape = }")

            pred_fake = gan.discriminator(fake_gas, real_dm)
            if epoch == 0:
                logging.debug(f"Made fake prediction of discriminator - shape {pred_fake.shape =}")

            loss_gan = gan.criterion_gan(pred_fake, gan.gt_valid)
            if epoch == 0:
                logging.debug(f"{loss_gan = }")

            # Pixel-wise loss
            loss_pixel = gan.criterion_pixelwise(fake_gas, real_gas)
            if epoch == 0:
                logging.debug(f"{loss_pixel = }")

            # Total loss
            loss_G = loss_gan + gan.lambda_pixel * loss_pixel
            if epoch == 0:
                logging.debug(f"{loss_G = }")

            loss_G.backward()
            if epoch == 0:
                logging.debug("loss_G.backward() done")

            gan.optimizer_G.step()
            if epoch == 0:
                logging.debug("optimizer_G.step() done")

            # ---------------------
            #  Train Discriminator
            # ---------------------

            if epoch == 0:
                logging.debug("Starting training Discriminator")

            gan.optimizer_D.zero_grad()

            # Real loss
            pred_real = gan.discriminator(real_gas, real_dm)
            loss_real = gan.criterion_gan(pred_real, gan.gt_valid)
            if epoch == 0:
                logging.debug(f"Generated pred_real (shape {pred_real.shape}) - loss {loss_real:0.2}")

            # Fake loss
            pred_fake = gan.discriminator(fake_gas.detach(), real_dm)  # CP: Detach from gradient calculation
            loss_fake = gan.criterion_gan(pred_fake, gan.gt_fake)
            if epoch == 0:
                logging.debug(f"Generated pred_fake (shape {pred_fake.shape}) - loss {loss_fake:0.2}")

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            gan.optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(gan.dataloader) + i + 1
            batches_left = opt.n_epochs * len(gan.dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - init_time) / batches_done)

            logging.info(
                f" [Epoch {epoch + 1:02d}/{opt.n_epochs:02d}]" +
                f" [Batch {i + 1}/{len(gan.dataloader)}]" +
                f" [D loss: {loss_D.item():.3e}]" +
                f" [G loss: {loss_G.item():.3e}, pixel: {loss_pixel.item():.3e}, adv: {loss_gan.item():.3e}]" +
                f" ETA: {str(time_left).split('.')[0]}"
            )

            # If at sample interval save image TODO
            if batches_done % opt.sample_interval == 0:
                slices = select_slice(real_dm.detach(), real_gas.detach(), fake_gas.detach(), random_dims=(0, 1),
                                      orthogonal_dim=2, weight=0.05)
                plot = heatmap_plot(*[s.squeeze() for s in slices],
                                    subplot_titles=("dark matter", "real gas", "predicted gas"))
                # write_image(plot, os.path.join("images", f"plot_epoch{epoch}_batch{i}.png"))
                # sample_images(batches_done)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints TODO
            dataset_name = f"{opt.sim_name}__{opt.mass_range}__{opt.n_voxel}"
            torch.save(gan.generator.state_dict(), f"saved_models/{dataset_name}/generator_{epoch}.pth")
            torch.save(gan.discriminator.state_dict(), f"saved_models/{dataset_name}/discriminator_{epoch}.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=os.path.join(os.path.curdir, 'data'), help="folder where data/ is")
    parser.add_argument("--sim_name", type=str, default='TNG300-1')
    parser.add_argument("--mass_range", type=str, default='MASS_1.00e+12_5.00e+12_MSUN')
    parser.add_argument("--n_voxel", type=int, default=128, help="number of voxels set for images")
    parser.add_argument("--skip_to_epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--generator_depth", type=int, default=5, help="depth of the generator architecture")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    # parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    # parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--patch_side", type=int, default=16)
    parser.add_argument("--sample_interval", type=int, default=5,
                        help="interval between sampling of images from generators")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")

    run(parser.parse_args())
