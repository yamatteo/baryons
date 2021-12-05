from argparse import Namespace
import datetime
import itertools
import logging
import os
import pprint

import pandas as pd
import torch

import gan.vox2vox


def load_opts():
    import defaults

    try:
        import options

        opts = dict(defaults.opts, **options.opts)
    except ImportError:
        opts = dict(defaults.opts)
    opts["cuda"] = torch.cuda.is_available()
    return opts


def setup_logging(opts: dict):
    with open(os.path.join(opts["root"], "last_run.log"), mode="w") as f:
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

    simple_opts = {
        key: value
        for (key, value) in opts.items()
        if isinstance(value, (int, float, str))
    }

    multi_opts = {
        key: value for (key, value) in opts.items() if isinstance(value, list)
    }

    multi_labels = tuple(multi_opts.keys())
    global_metrics = pd.DataFrame(
        columns=["epoch", "time", "loss_g", "loss_d", *multi_labels]
    )
    for possible_opt in itertools.product(*multi_opts.values()):
        logging.info(
            f"Launching run with options:\n{pprint.pformat({key: value for (key, value) in zip(multi_labels, possible_opt)}, indent=4)}"
        )
        torch.cuda.empty_cache()
        extra_opt = dict(zip(multi_labels, possible_opt))
        opt = dict(simple_opts, **extra_opt)
        try:
            metrics = gan.vox2vox.single_run(Namespace(**opt)).assign(**extra_opt)
            global_metrics = global_metrics.append(metrics, ignore_index=True)
        except (RuntimeError, Exception):
            logging.exception("Bad run...")

    global_metrics.to_csv(os.path.join(opts["root"], "last_run.csv"), index=False)
