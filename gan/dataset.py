import glob
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision


class Dataset3D(torch.utils.data.Dataset):
    def __init__(self, opt, mode, transformations):
        self.mode = mode
        self.nvoxel = opt.nvoxel
        self.transform = torchvision.transforms.Compose(transformations)

        self.files = list(zip(
            sorted((
                Path(opt.data_path)
                / f"{opt.sim_name}_SNAP{opt.snap_num:03d}_MASS{opt.mass_min:.2e}_{opt.mass_max:.2e}_NGASMIN{opt.n_gas_min}"
                / f"nvoxel_{opt.nvoxel}"
                / mode
            ).glob("halo_*_dm_coalesced.npy")),
            sorted((
                Path(opt.data_path)
                / f"{opt.sim_name}_SNAP{opt.snap_num:03d}_MASS{opt.mass_min:.2e}_{opt.mass_max:.2e}_NGASMIN{opt.n_gas_min}"
                / f"nvoxel_{opt.nvoxel}"
                / mode
            ).glob("halo_*_gas_coalesced.npy"))
        ))

    def __getitem__(self, index):
        return (
            self.transform(torch.load(
                self.files[index][0]
            ).to_dense()).unsqueeze(0),
            self.transform(torch.load(
                self.files[index][1]
            ).to_dense()).unsqueeze(0),
        )

    def __len__(self):
        return len(self.files)
