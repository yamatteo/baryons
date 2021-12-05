import argparse
import datetime
import itertools
import logging
from importlib import reload

import pandas as pd
import os
from argparse import Namespace

import torch

import gan.vox2vox
from defaults import defaults
from options import options

if __name__ == "__main__":
    opts = dict(defaults, **options)
    opts["cuda"] = torch.cuda.is_available()

    logging.basicConfig(
        filename="last_run.log",
        filemode="w",
        format="%(levelname)s: %(message)s",
        level=logging.DEBUG if opts["log_level"] == "debug" else logging.INFO,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(console)
    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("parso.python.diff").disabled = True

    with open(os.path.join(opts["root"], "last_run.log"), mode="w") as f:
        f.write(f"main.py > vox2vox.py @ {datetime.datetime.now()}")
        f.write(os.linesep)

    simple_opts = {
        key: value
        for (key, value) in opts.items()
        if isinstance(value, (int, float, str))
    }

    multi_opts = {
        key: value for (key, value) in opts.items() if isinstance(value, list)
    }

    multi_labels = tuple(multi_opts.keys())
    global_metrics = pd.DataFrame(columns=["epoch", "time", "loss_g", "loss_d", *multi_labels])
    for possible_opt in itertools.product(*multi_opts.values()):
        torch.cuda.empty_cache()
        extra_opt = dict(zip(multi_labels, possible_opt))
        opt = dict(simple_opts, **extra_opt)
        try:
            metrics = gan.vox2vox.single_run(Namespace(**opt)).assign(**extra_opt)
            global_metrics = global_metrics.append(metrics, ignore_index=True)
        except (RuntimeError, Exception):
            logging.exception("Bad run...")

    global_metrics.to_csv(os.path.join(opts["root"], "last_run.csv"), index=False)
