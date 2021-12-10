import logging
import os
import datetime
import pprint

from options import opts
from . import BlockworkVox2Vox
from .logger import set_logger
from argparse import Namespace
from torch.utils.tensorboard import SummaryWriter

opts["run_path"] = "runs"
opts["lr"] = 0.005
bvv = BlockworkVox2Vox(Namespace(**opts))

os.makedirs(bvv.run_path, exist_ok=True)
set_logger(bvv.run_path)
writer = SummaryWriter()

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

bvv.train(1, 1, -1)
writer.add_graph(bvv.gen, bvv.dataloader[0][0].float().cuda())
writer.close()