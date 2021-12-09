import logging
import os
import datetime
import pprint

from options import opts
from . import BlockworkVox2Vox
from argparse import Namespace

opts["run_path"] = "test"
opts["lr"] = 0.005
bvv = BlockworkVox2Vox(Namespace(**opts))

os.makedirs(bvv.run_path, exist_ok=True)
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

logging.basicConfig(
    filename="last_run.log",
    filemode="a",
    format="%(levelname)s: %(message)s",
    level=logging.DEBUG if opts["log_level"] == "debug" else logging.INFO,
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(console)
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("parso.python.diff").disabled = True

bvv.train(20, 1, -1)