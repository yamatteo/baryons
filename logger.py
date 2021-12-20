import logging
from pathlib import Path
from options import opts

logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("parso.python.diff").disabled = True

run_path = Path(opts["run_base_path"])

logger = logging.getLogger("baryons")
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(
    {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARN,
    }[opts["console_log_level"]]
)
console.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console)

logfile = logging.FileHandler(filename=run_path / "last_run.log", mode="w")
logfile.setLevel(logging.DEBUG)
logfile.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
logger.addHandler(logfile)