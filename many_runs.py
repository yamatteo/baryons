import argparse
import datetime

import torch
import os

import gan.pix2pix
import gan.defaults

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="...", argument_default=argparse.SUPPRESS
    )  # TODO: add meaningful description
    parser.add_argument(
        "--b1",
        type=eval,
        help="float or list[float] -- adam optimization: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--b2",
        type=eval,
        help="float or list[float] -- adam optimization: decay of first order momentum of gradient",  # TODO CP: maybe second order?
    )
    parser.add_argument("--batch_size", type=int, help="int -- size of the batches")
    parser.add_argument(
        "--channels", type=eval, help="int or list[int] -- number of image channels"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=-1,
        help="int -- interval between model checkpoints; -1 means no checkpoint",
    )
    parser.add_argument(
        "--cuda",
        type=bool,
        default=True if torch.cuda.is_available() else False,
    )
    parser.add_argument(
        "--decay_epoch",
        type=eval,
        help="int or list[int] -- epoch from which to start lr decay",
    )
    parser.add_argument(
        "--generator_depth",
        type=eval,
        help="int or list[int] -- depth of the generator architecture",
    )
    parser.add_argument(
        "--lambda_pixel",
        type=eval,
        help="int or list[int] -- magnifying factor for criterion_pixelwise loss",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        choices=["info", "debug"],
        help="str -- log level, either default 'info' or 'debug' to log more messages",
    )
    parser.add_argument(
        "--log_mode",
        type=str,
        default="a",
        choices=["w", "a"],
        help="str -- log mode, either default 'w' to overwrite previous logs or 'a' to append to previous log",
    )
    parser.add_argument(
        "--lr",
        type=eval,
        help="float or list[float] -- adam optimization: learning rate",
    )
    parser.add_argument(
        "--mass_range",
        type=str,
        help="str -- mass range of the considered halos, e.g. 'MASS_1.00e+12_5.00e+12_MSUN'",
    )
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=8,
        help="int -- number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        help="int -- number of epochs of training",
    )
    parser.add_argument(
        "--n_voxel",
        type=int,
        help="int -- number of voxels set for images",
    )
    parser.add_argument(
        "--num_filters",
        type=eval,
        help="int or list[int] -- number of features after the first convolution",
    )
    parser.add_argument(
        "--patch_side",
        type=eval,
        help="int or list[int] -- side length (voxels) for the patch of the discriminator",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=os.getcwd(),
        help="path -- folder where dataset/ is",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=5,
        help="int -- interval between sampling of images from generators; -1 means no images",
    )
    parser.add_argument(
        "--sim_name",
        type=str,
        help="str -- name of the illustris simulation to consider",
    )
    parser.add_argument(
        "--skip_to_epoch",
        type=int,
        default=0,
        help="int -- epoch to start training from, loading from previous checkpoints",
    )

    defaults = vars(gan.defaults.defaults)
    commands = vars(parser.parse_args())
    opts = dict(defaults, **commands)
    single_opts = {
        key: value
        for (key, value) in opts.items()
        if isinstance(value, (int, float, str))
    }
    multi_opts = {
        key: value for (key, value) in opts.items() if isinstance(value, list)
    }

    with open(os.path.join(opts["root"], "last_run.log"), mode="w") as f:
        f.write(f"many_runs.py {datetime.datetime.now()}")
        f.write(os.linesep)

    metrics = gan.pix2pix.many_runs(single_opts, multi_opts)

    metrics.to_csv(os.path.join(opts["root"], "last_run.csv"))
