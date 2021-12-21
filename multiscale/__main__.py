import itertools
import logging
import os
import datetime
import pprint
import shutil
from functools import reduce
from pathlib import Path

import torch

from options import opts
from preprocessing import preprocess
from . import Vox2Vox, roundrun

# from .discriminator import VoxNet

# Ensure paths
os.makedirs(opts["simulation_base_path"], exist_ok=True)

if opts["delete_previous_voxels"]:
    shutil.rmtree(opts["voxels_base_path"], ignore_errors=True)
os.makedirs(opts["voxels_base_path"], exist_ok=True)

if opts["delete_previous_runs"]:
    shutil.rmtree(opts["run_base_path"], ignore_errors=True)
os.makedirs(opts["run_base_path"], exist_ok=True)

# Setup logging
from logger import logger
logger.debug(f"Running {__file__}")

# Preprocessing
ids, stats = preprocess(
        source_path=Path(opts["simulation_base_path"]) / opts['sim_name'] / "output",
        target_path=Path(opts["voxels_base_path"])
        / (
            f"{opts['sim_name']}"
            + f"_SNAP{opts['snap_num']:03d}"
            + f"_MASS{opts['mass_min']:.2e}"
            + f"_{opts['mass_max']:.2e}"
            + f"_NGASMIN{opts['n_gas_min']}"
        ),
        sim_name=opts['sim_name'],
        snap_num=opts['snap_num'],
        mass_min=opts['mass_min'],
        mass_max=opts['mass_max'],
        nvoxel=opts['nvoxel'],
        n_gas_min=opts['n_gas_min'],
        fixed_size=opts["preprocessing_fixed_size"],
    )

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

if len(multi_opts) == 0:
    logger.info(f" Launching single run with options:\n {pprint.pformat({key: value for (key, value) in opts.items()}, indent=4)}")
    roundrun(opts["rounds"], opts, "opts")
else:
    logger.info(f"Launching multiple runs with base options:\n{pprint.pformat({key: value for (key, value) in simple_opts.items()}, indent=4)}")
    for i, possible_opt in enumerate(combinations):
        logger.info(
            f" Launching run {i+1}/{len(combinations)} with options:\n {pprint.pformat({key: value for (key, value) in zip(multi_labels, possible_opt)}, indent=4)}"
        )
        extra_opt = dict(zip(multi_labels, possible_opt))
        run_opts = dict(simple_opts, **extra_opt, run_index=i)
        roundrun(opts["rounds"], run_opts, str("~").join([k+str(o) for k, o in extra_opt.items()]))

# bvv = BlockworkVox2Vox(**opts)
#
# bvv.train(opts["n_epochs"], opts["sample_interval"], opts["checkpoint_interval"])
# bvv.writer.close()