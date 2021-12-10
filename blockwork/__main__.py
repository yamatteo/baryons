import logging
import os
import datetime
import pprint
from functools import reduce

from options import opts
from .bwnet import BlockworkVox2Vox
from .discriminator import VoxNet
from .logger import set_logger
from argparse import Namespace

opts["run_path"] = "runs"
bvv = BlockworkVox2Vox(Namespace(**opts))

os.makedirs(bvv.run_path, exist_ok=True)
set_logger(bvv.run_path)

with open(bvv.run_path / "last_run.log", mode="w") as f:
    f.writelines(
        [
            f"Running {__file__}",
            os.linesep,
            f"Starting at {datetime.datetime.now()}",
            os.linesep,
            f"Options are:",
            os.linesep,
            pprint.pformat(opts, indent=4),
            os.linesep,
        ]
    )

bvv.train(opts["n_epochs"], opts["sample_interval"], opts["checkpoint_interval"])
bvv.writer.close()