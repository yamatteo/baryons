import glob
import os
import pandas as pd
import torch
import torchvision


class Dataset3D(torch.utils.data.Dataset):

    def __init__(self, sim_name, mass_range, n_voxel, mode, transforms):
        path = os.path.join("data", sim_name, mass_range, f"NVOXEL_{n_voxel}", mode.upper())
        self.n_voxel = n_voxel
        self.transform = torchvision.transforms.Compose(transforms)
        self.files = sorted(glob.glob(os.path.join(path, "halo_*_nvoxel_*.csv")))

    def __getitem__(self, index):
        halo = pd.read_csv(self.files[index]).astype({'gas/nogas': bool})
        dm_sparse = halo[halo['gas/nogas'] == False]
        gas_sparse = halo[halo['gas/nogas'] == True]
        return {
            "DM": self.transform(
                torch.sparse_coo_tensor(
                    indices=dm_sparse[['Xvox', 'Yvox', 'Zvox']].values.T,
                    values=dm_sparse.Mvox.values,
                    size=[self.n_voxel] * 3)
                    .to_dense()).unsqueeze(0),
            "GAS": self.transform(
                torch.sparse_coo_tensor(
                    indices=gas_sparse[['Xvox', 'Yvox', 'Zvox']].values.T,
                    values=gas_sparse.Mvox.values,
                    size=[self.n_voxel] * 3)
                    .to_dense()).unsqueeze(0),
        }

    def __len__(self):
        return len(self.files)
