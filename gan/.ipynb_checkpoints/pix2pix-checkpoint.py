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

# from gan.dataset import Dataset3D
# from gan.models import UNet, Discriminator, weights_init_normal
# from visualization import write_image, heatmap_plot, select_slice


def main():
    print(f"{sys.path =}")
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=os.path.join(os.path.curdir, 'data'), help="folder where data is")
    parser.add_argument("--sim_name", type=str, default='TNG300-1')
    parser.add_argument("--mass_range", type=str, default='MASS_1.00e+12_5.00e+12_MSUN')
    parser.add_argument("--n_voxel", type=int, default=256, help="number of voxels set for images")
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    # parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    # parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--patch_side", type=int, default=16)
    parser.add_argument("--sample_interval", type=int, default=5,
                        help="interval between sampling of images from generators")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
    opt = parser.parse_args()
    print(opt)

    os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False
    cuda = False

    # Loss functions
    criterion_gan = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = 100

    # Calculate output of image discriminator (PatchGAN)
    # TODO: investigate whether 2**4 is working in our case
    # Isola+18 suggests a patch size of 70x70
    # patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4, opt.img_width // 2 ** 4)
    assert opt.n_voxel % opt.patch_side == 0, "parameter n_voxel should be a multiple of parameter patch_side"
    patch = (opt.channels, opt.n_voxel // opt.patch_side, opt.n_voxel // opt.patch_side, opt.n_voxel // opt.patch_side)

    # Initialize generator and discriminator
    generator = UNet(in_dim=opt.channels, out_dim=opt.channels, num_filters=4)
    # generator = UNet(in_channels=opt.channels, out_channels=opt.channels)
    discriminator = Discriminator(in_channels=opt.channels)

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_gan.cuda()
        criterion_pixelwise.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
        discriminator.load_state_dict(
            torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Configure dataloaders
    transforms_ = [
        # CP: transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        # transforms.ToTensor(),
        # CP: transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.5), (0.5)),  # CP: one value for the mean for each channel (here one channel and two images)
    ]

    dataloader = DataLoader(
        Dataset3D(
            sim_name=opt.sim_name,
            mass_range=opt.mass_range,
            n_voxel=opt.n_voxel,
            mode="train",
            transforms=transforms_,
        ),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )
    print('Loaded train set.')

    val_dataloader = DataLoader(
        Dataset3D(
            sim_name=opt.sim_name,
            mass_range=opt.mass_range,
            n_voxel=opt.n_voxel,
            mode="valid",
            transforms=transforms_,
        ),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )
    print('Loaded validation set.')

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def sample_images(batches_done):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(val_dataloader))
        real_A = Variable(imgs["GAS"].type(Tensor))
        real_B = Variable(imgs["DM"].type(Tensor))
        fake_B = generator(real_A)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
        # TODO: make sure save_image is able to deal with 3D numpy arrays
        save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)

    # ----------
    #  Training
    # ----------

    prev_time = time.time()

    for epoch in range(opt.epoch, opt.n_epochs):

        print('\nStarting epoch ', epoch)

        for i, batch in enumerate(dataloader):  # batch is a dictionary

            print('Starting batch ', i)

            # Model inputs - shape [batch_size, channels, n_voxel, n_voxel, n_voxel]
            real_dm = Variable(batch["DM"].type(Tensor))
            real_gas = Variable(batch["GAS"].type(Tensor))

            print(f"Defined real_dm and real_gas - shape {real_dm.shape}")

            # Adversarial ground truths
            # shape [batch_size, channels, n_voxel // patch_side, n_voxel // patch_side, n_voxel // patch_side]
            valid = Tensor(np.ones((real_dm.size(0), *patch)))
            fake = Tensor(np.zeros((real_dm.size(0), *patch)))

            print(f"Defined valid and fake - shape {valid.shape}")

            # ------------------
            #  Train Generators
            # ------------------

            print("Starting training Generator")

            optimizer_G.zero_grad()

            # GAN loss
            fake_gas = generator(real_dm)
            print(f"Generated fake_gase - shape {fake_gas.shape = }")

            pred_fake = discriminator(fake_gas, real_dm)
            print(f"Made fake prediction of discriminator - shape {pred_fake.shape =}")

            loss_gan = criterion_gan(pred_fake, valid)
            print(f"{loss_gan = }")

            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_gas, real_gas)
            print(f"{loss_pixel = }")

            # Total loss
            loss_G = loss_gan + lambda_pixel * loss_pixel
            print(f"{loss_G = }")

            loss_G.backward()
            print("loss_G.backward() done")

            optimizer_G.step()
            print("optimizer_G.step() done")

            # ---------------------
            #  Train Discriminator
            # ---------------------

            print("Starting training Discriminator")

            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(real_gas, real_dm)
            loss_real = criterion_gan(pred_real, valid)
            print(f"Generated pred_real (shape {pred_real.shape}) - loss {loss_real:0.2}")

            # Fake loss
            pred_fake = discriminator(fake_gas.detach(), real_dm)  # CP: Detach from gradient calculation
            loss_fake = criterion_gan(pred_fake, fake)
            print(f"Generated pred_fake (shape {pred_fake.shape}) - loss {loss_fake:0.2}")

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                f"\r[Epoch {epoch}/{opt.n_epochs}]" +
                f" [Batch {i}/{len(dataloader)}]" +
                f" [D loss: {loss_D.item()}]" +
                f" [G loss: {loss_G.item()}, pixel: {loss_pixel.item()}, adv: {loss_gan.item()}]" +
                f" ETA: {time_left}"
            )

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                slices = select_slice(real_dm.detach(), real_gas.detach(), fake_gas.detach(), orthogonal_dim=2, weight=0.0)
                plot = heatmap_plot(*[s.squeeze() for s in slices], subplot_titles=("dark matter", "real gas", "predicted gas"))
                write_image(plot, os.path.join("images", f"plot_epoch{epoch}_batch{i}.png"))
                # sample_images(batches_done)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
            torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))


if __name__ == '__main__':
    main()
