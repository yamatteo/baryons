import logging

logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("parso.python.diff").disabled = True


def set_logger(run_path):
    logger = logging.getLogger("blockwork")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)

    logfile = logging.FileHandler(filename=run_path / "last_run.log", mode="w")
    logfile.setLevel(logging.DEBUG)
    logfile.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
    logger.addHandler(logfile)

    logger.setLevel(logging.DEBUG)
