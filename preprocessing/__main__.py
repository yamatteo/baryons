from options import opts
from . import preprocess
from pathlib import Path
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console)

if __name__ == "__main__":
    preprocess(
        source_path=Path(opts["simulation_base_path"]) / opts['sim_name'] / "output",
        target_path=Path(opts["preprocessing_path"]) / opts["preprocessing_name"],
        sim_name=opts['sim_name'],
        snap_num=opts['snap_num'],
        mass_min=opts['mass_min'],
        mass_max=opts['mass_max'],
        nvoxel=opts['nvoxel'],
        n_gas_min=opts['n_gas_min'],
    )
