import logging
import os
import datetime
import pprint

from options import opts
from . import BlockworkVox2Vox
from .logger import set_logger
from argparse import Namespace

opts["run_path"] = "test"
opts["lr"] = 0.005
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

bvv.train(5, 1, -1)