import glob
import os
import pandas as pd
import torch
import torchvision
import torch.nn.functional as F

SHRINK_FACTOR = 2

class Dataset3D(torch.utils.data.Dataset):

    def __init__(self, sim_name, mass_range, n_voxel, mode, transforms):
        path = os.path.join("dataset", sim_name, mass_range, f"NVOXEL_{n_voxel * SHRINK_FACTOR}", mode.upper())
        self.n_voxel = n_voxel
        self.transform = torchvision.transforms.Compose(transforms)
        self.files = sorted(glob.glob(os.path.join(path, "halo_*_nvoxel_*.csv")))

    def __getitem__(self, index):
        halo = pd.read_csv(self.files[index]).astype({'gas/nogas': bool})
        dm_sparse = halo[halo['gas/nogas'] == False]
        gas_sparse = halo[halo['gas/nogas'] == True]
        temp_dict = {
            "DM": self.transform(
                torch.sparse_coo_tensor(
                    indices=dm_sparse[['Xvox', 'Yvox', 'Zvox']].values.T,
                    values=dm_sparse.Mvox.values,
                    size=[self.n_voxel * SHRINK_FACTOR] * 3)
                    .to_dense()).unsqueeze(0),
            "GAS": self.transform(
                torch.sparse_coo_tensor(
                    indices=gas_sparse[['Xvox', 'Yvox', 'Zvox']].values.T,
                    values=gas_sparse.Mvox.values,
                    size=[self.n_voxel * SHRINK_FACTOR] * 3)
                    .to_dense()).unsqueeze(0),
        }
        return {
            key: F.avg_pool3d(t, kernel_size=SHRINK_FACTOR, divisor_override=1)
            for key, t in temp_dict.items()
        }

    def __len__(self):
        return len(self.files)
