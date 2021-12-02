import argparse
import torch
import os
from argparse import Namespace

from gan.pix2pix import many_runs
from gan.defaults import defaults

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='...', argument_default=argparse.SUPPRESS)  # TODO: add meaningful description
    parser.add_argument(
        "--b1",
        type=list,
        help="[list] adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--b2",
        type=list,
        help="[list] adam: decay of first order momentum of gradient",  # TODO CP: maybe second order?
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="size of the batches"
    )
    parser.add_argument(
        "--channels",
        type=int,
        help="number of image channels"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        help="interval between model checkpoints; -1 means no checkpoint",
    )
    parser.add_argument(
        "--cuda",
        type=bool,
        default=True if torch.cuda.is_available() else False,
    )
    parser.add_argument(
        "--decay_epoch",
        type=int,
        help="epoch from which to start lr decay",
    )
    parser.add_argument(
        "--generator_depth",
        type=eval,
        help="[list] depth of the generator architecture",
    )
    parser.add_argument(
        "--lambda_pixel",
        type=int,
        help="magnifying factor for criterion_pixelwise loss"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        choices=["info", "debug"],
        help="log level, either default 'info' or 'debug' to log more messages"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="adam: learning rate"
    )
    parser.add_argument(
        "--mass_range",
        type=str,
        help="mass range of the considered halos, e.g. 'MASS_1.00e+12_5.00e+12_MSUN'",
    )
    parser.add_argument(
        "--n_cpu",
        type=int,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        help="number of epochs of training",
    )
    parser.add_argument(
        "--n_voxel",
        type=int,
        help="number of voxels set for images",
    )
    parser.add_argument(
        "--num_filters",
        type=int,
        help="number of features after the first convolution"
    )
    parser.add_argument(
        "--patch_side",
        type=int,
        help="side length (voxels) for the patch of the discriminator"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=os.getcwd(),
        help="folder where data/ is",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        help="interval between sampling of images from generators; -1 means no images",
    )
    parser.add_argument(
        "--sim_name",
        type=str,
        help="name of the illustris simulation to consider"
    )
    parser.add_argument(
        "--skip_to_epoch",
        type=int,
        help="epoch to start training from, loading from previous checkpoints"
    )

    opt = dict(vars(defaults), **vars(parser.parse_args()))
    base_opt = Namespace(**{
        key: value for (key, value) in opt.items() if not isinstance(value, list)
    })
    prog_opt = {
        key: value for (key, value) in opt.items() if isinstance(value, list)
    }

    metrics = many_runs(base_opt, prog_opt)

    metrics.to_csv(os.path.join(base_opt.root, "last_run.csv"))