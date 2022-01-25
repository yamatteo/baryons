import argparse
import itertools
import os
import pprint
import shutil
from pathlib import Path

import torch.cuda

from options import opts
from preprocessing import assert_preprocessing
from vox2vox import env_vars, logger, init, train, get_hash, evaluate, apply

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument("command", choices=(
    "train",
    "evaluate",
    "apply",
    "create",
    "remove",
    "reset",
    "clear_reports",
))
# parser.add_argument("-r", "--root",              type=str,   default='/gpfswork/rech/qbf/uiu95bi/predicting_baryons/data/TNG300-1_MASS_1.00e+12_5.00e+12_MSUN/nvoxel_256/3d_on_the_fly/',   help="folder where data is")
# parser.add_argument("--n_voxel",                 type=int,   default=256,    help="number of voxels set for images")
# parser.add_argument("--epoch",                   type=int,   default=0,      help="epoch to start training from")
parser.add_argument("-n", "--n_epochs", type=int)
parser.add_argument("--console_log_level", type=str, default="info")
parser.add_argument("--delete_previous_voxels", type=bool, default=False)
parser.add_argument("--delete_previous_runs", type=bool, default=False)
parser.add_argument("--preprocessing_fixed_size", type=eval, default=None)
parser.add_argument("-b", "--batch_size", type=eval)
# parser.add_argument("--lr",                      type=float, default=0.0002, help="adam: learning rate")
# parser.add_argument("--b1",                      type=float, default=0.5,    help="adam: decay of first order momentum of gradient")
# parser.add_argument("--b2",                      type=float, default=0.999,  help="adam: decay of first order momentum of gradient")
# parser.add_argument("--decay_epoch",             type=int,   default=10,     help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, help="cpus to use in batch generation")
# parser.add_argument("--img_height",              type=int,   default=256,    help="size of image height")
# parser.add_argument("--img_width",               type=int,   default=256,    help="size of image width")
# parser.add_argument("--channels",                type=int,   default=1,      help="number of image channels")
# parser.add_argument("-s", "--sample_interval",   type=int,   default=None,   help="interval between sampling of images from generators")
# parser.add_argument("--checkpoint_interval",     type=int,   default=50,     help="interval between model checkpoints")
# parser.add_argument("-e", "--evaluate", dest='evaluate', action='store_true', help="evaluates model if present")
# parser.add_argument("--use_adabelief", dest='use_adabelief', action='store_true', help="whether to use AdaBelief instead of Adam")
opts.update(env_vars)
opts.update(vars(parser.parse_args()))

if "cuda" not in opts.keys():
    opts["cuda"] = torch.cuda.is_available()

# from logger import logger
logger.info(f'Found GPU: {opts["cuda"]}')

# Ensure paths
if opts["delete_previous_voxels"]:
    shutil.rmtree(opts["voxels_base_path"], ignore_errors=True)
os.makedirs(opts["voxels_base_path"], exist_ok=True)

if opts["delete_previous_runs"]:
    shutil.rmtree(opts["run_base_path"], ignore_errors=True)
os.makedirs(opts["run_base_path"], exist_ok=True)

os.makedirs(opts["models_base_path"], exist_ok=True)

# Setup logging
logger.debug(f"Running {__file__}")

# Ensure preprocessing
opts["restricted_sim_name"] = (
        f"{opts['sim_name']}"
        + f"_SNAP{opts['snap_num']:03d}"
        + f"_MASS{opts['mass_min']:.2e}"
        + f"_{opts['mass_max']:.2e}"
        + f"_NGASMIN{opts['n_gas_min']}"
)
assert assert_preprocessing(
    target_path=Path(opts["voxels_base_path"]) / opts["restricted_sim_name"],
    nvoxel=opts['nvoxel'],
    fixed_size=opts["preprocessing_fixed_size"],
), f"Preprocessing not completed for {opts['restricted_sim_name']}"


# Setup eventual multiple runs
simple_opts = {
    key: value
    for (key, value) in opts.items()
    if isinstance(value, (int, float, str, tuple))
}

multi_opts = {
    key: value
    for (key, value) in opts.items()
    if isinstance(value, list)
}

combinations = list(itertools.product(*multi_opts.values()))
multi_labels = tuple(multi_opts.keys())
command = opts["command"]

if len(multi_opts) == 0:
    logger.info(
        f" Launching single action with options:\n {pprint.pformat({key: value for (key, value) in opts.items()}, indent=4)}")
    model_id = get_hash(opts)
    if command == "create":
        if os.path.exists(Path(opts["models_base_path"]) / model_id):
            logger.warning(f"Aborting creation of {model_id}: it already exists.")
        else:
            os.mkdir(Path(opts["models_base_path"]) / model_id)
            init(model_id, opts)
            logger.info(f"Model {model_id} created and initialized.")
    elif command == "reset":
        if os.path.exists(Path(opts["models_base_path"]) / model_id):
            init(model_id, opts)
            logger.info(f"Model {model_id} reset to initial state.")
        else:
            logger.info(f"Model {model_id} does not exists.")
    elif command == "remove":
        if os.path.exists(Path(opts["models_base_path"]) / model_id):
            shutil.rmtree(Path(opts["models_base_path"]) / model_id, ignore_errors=True)
            logger.info(f"Model {model_id} removed.")
        else:
            logger.info(f"Model {model_id} does not exists.")
    elif command == "clear_reports":
        if os.path.exists(Path(opts["run_base_path"]) / opts["restricted_sim_name"] / model_id):
            shutil.rmtree(Path(opts["run_base_path"]) / opts["restricted_sim_name"] / model_id, ignore_errors=True)
            logger.info(f"Reports of {model_id} cleared.")
        else:
            logger.info(f"No reports for {model_id}.")
    elif command == "train":
        if os.path.exists(Path(opts["models_base_path"]) / model_id):
            train(model_id, opts)
            logger.info(f"Model {model_id} trained for {opts['n_epochs']} epochs.")
        else:
            logger.info(f"Model {model_id} does not exists.")
    elif command == "evaluate":
        if os.path.exists(Path(opts["models_base_path"]) / model_id):
            output = evaluate(model_id, opts)
            logger.info(f"Evaluated a batch of {opts['batch_size']} halos.")
            for label, std_mean in output.items():
                logger.info(f"  {label}: {std_mean[1].item():.3e} ±{std_mean[0].item():.0e}")
        else:
            logger.info(f"Model {model_id} does not exists.")
    elif command == "apply":
        if os.path.exists(Path(opts["models_base_path"]) / model_id):
            apply(model_id, opts)
            logger.info(f"Applied {model_id}")
        else:
            logger.info(f"Model {model_id} does not exists.")
    # roundrun(opts["rounds"], opts, "opts")
else:
    logger.info(
        f"Launching multiple actions with base options:\n{pprint.pformat({key: value for (key, value) in simple_opts.items()}, indent=4)}")
    for i, possible_opt in enumerate(combinations):
        logger.info(
            f" Launching run {i + 1}/{len(combinations)} with options:\n {pprint.pformat({key: value for (key, value) in zip(multi_labels, possible_opt)}, indent=4)}"
        )
        extra_opt = dict(zip(multi_labels, possible_opt))
        run_opts = dict(simple_opts, **extra_opt, run_index=i)
        model_id = get_hash(run_opts)
        if command == "create":
            if os.path.exists(Path(run_opts["models_base_path"]) / model_id):
                logger.warning(f"Aborting creation of {model_id}: it already exists.")
            else:
                os.mkdir(Path(run_opts["models_base_path"]) / model_id)
                init(model_id, run_opts)
                logger.info(f"Model {model_id} created and initialized.")
        elif command == "reset":
            if os.path.exists(Path(run_opts["models_base_path"]) / model_id):
                init(model_id, run_opts)
                logger.info(f"Model {model_id} reset to initial state.")
            else:
                logger.info(f"Model {model_id} does not exists.")
        elif command == "remove":
            if os.path.exists(Path(run_opts["models_base_path"]) / model_id):
                shutil.rmtree(Path(run_opts["models_base_path"]) / model_id, ignore_errors=True)
                logger.info(f"Model {model_id} removed.")
            else:
                logger.info(f"Model {model_id} does not exists.")
        elif command == "clear_reports":
            if os.path.exists(Path(run_opts["run_base_path"]) / run_opts["restricted_sim_name"] / model_id):
                shutil.rmtree(Path(run_opts["run_base_path"]) / run_opts["restricted_sim_name"] / model_id, ignore_errors=True)
                logger.info(f"Reports of {model_id} cleared.")
            else:
                logger.info(f"No reports for {model_id}.")
        elif command == "train":
            if os.path.exists(Path(run_opts["models_base_path"]) / model_id):
                train(model_id, run_opts)
                logger.info(f"Model {model_id} trained for {run_opts['n_epochs']} epochs.")
            else:
                logger.info(f"Model {model_id} does not exists.")
        elif command == "evaluate":
            if os.path.exists(Path(run_opts["models_base_path"]) / model_id):
                output = evaluate(model_id, run_opts)
                logger.info(f"Evaluated a batch of {run_opts['batch_size']} halos.")
                for label, std_mean in output.items():
                    logger.info(f"  {label}: {std_mean[1].item():.3e} ±{std_mean[0].item():.0e}")
            else:
                logger.info(f"Model {model_id} does not exists.")
        elif command == "apply":
            if os.path.exists(Path(run_opts["models_base_path"]) / model_id):
                apply(model_id, run_opts)
                logger.info(f"Applied {model_id}")
            else:
                logger.info(f"Model {model_id} does not exists.")
        # roundrun(run_opts["rounds"], run_run_opts, str("~").join([k + str(o) for k, o in extra_opt.items()]))
# print(opts)
