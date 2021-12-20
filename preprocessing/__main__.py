import logger

from options import opts
from . import preprocess
from pathlib import Path

if __name__ == "__main__":
    preprocess(
        source_path=Path(opts["simulation_base_path"]) / opts['sim_name'] / "output",
        target_path=Path(opts["voxels_base_path"])
        / (
            f"{opts['sim_name']}"
            + f"_SNAP{opts['snap_num']:03d}"
            + f"_MASS{opts['mass_min']:.2e}"
            + f"_{opts['mass_max']:.2e}"
            + f"_NGASMIN{opts['n_gas_min']}"
        ),
        sim_name=opts['sim_name'],
        snap_num=opts['snap_num'],
        mass_min=opts['mass_min'],
        mass_max=opts['mass_max'],
        nvoxel=opts['nvoxel'],
        n_gas_min=opts['n_gas_min'],
    )
