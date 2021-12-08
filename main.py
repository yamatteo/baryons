from argparse import Namespace
import datetime
import itertools
import logging
import os
import pprint
import pickle
import hashlib

import pandas as pd
import torch

import gan.vox2vox


def load_opts():
    try:
        from options import opts

        opts["cuda"] = torch.cuda.is_available()
        opts["run_id"] = hashlib.md5(str(opts).encode("utf8")).hexdigest()
        opts["run_path"] = os.path.join(opts["output_path"], f"run_{opts['run_id']}")
        return opts
    except ImportError:
        raise ImportError(
            f"No module options.py found in {os.getcwd()}. Copy, rename and modify defaults.py"
        )


def setup_logging(opts: dict):
    with open(os.path.join(opts["output_path"], "last_run.log"), mode="w") as f:
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


if __name__ == "__main__":
    opts = load_opts()
    setup_logging(opts)

    os.makedirs(opts["run_path"], exist_ok=True)

    if os.path.exists(os.path.join(opts["run_path"], "global_metrics.csv")):
        logging.exception("This run is already completed")

    simple_opts = {
        key: value
        for (key, value) in opts.items()
        if isinstance(value, (int, float, str, tuple))
    }

    multi_opts = {
        key: value for (key, value) in opts.items() if isinstance(value, list)
    }

    multi_labels = tuple(multi_opts.keys())

    try:
        assert opts["may_resume"]
        with open(os.path.join(opts["run_path"], "state"), "rb") as state_file:
            state = pickle.load(state_file)
            global_metrics = state["global_metrics"]
            completed = state["completed"]
            logging.info(f"Resume from run {completed}")
    except (FileNotFoundError, KeyError, AssertionError):
        state = None
        global_metrics = pd.DataFrame(
            columns=["epoch", "time", "loss_g", "loss_d", *multi_labels]
        )
        completed = -1

    for i, possible_opt in enumerate(itertools.product(*multi_opts.values())):
        if i > completed:
            logging.info(
                f"Launching run {i} with options:\n{pprint.pformat({key: value for (key, value) in zip(multi_labels, possible_opt)}, indent=4)}"
            )
            torch.cuda.empty_cache()
            extra_opt = dict(zip(multi_labels, possible_opt))
            opt = dict(simple_opts, **extra_opt, run_index=i)
            logging.debug(f"Before single_run, {torch.cuda.memory_allocated() = }")
            metrics = gan.vox2vox.single_run(Namespace(**opt)).assign(**extra_opt)
            global_metrics = global_metrics.append(metrics, ignore_index=True)
            logging.debug(f"After single_run, {torch.cuda.memory_allocated() = }")

            state = dict(
                simple_opts=simple_opts,
                multi_opts=multi_opts,
                global_metrics=global_metrics,
                completed=i,
            )

            with open(os.path.join(opts["run_path"], "state"), "wb") as state_file:
                pickle.dump(state, state_file)

    global_metrics.to_csv(
        os.path.join(opts["run_path"], "global_metrics.csv"), index=False
    )
