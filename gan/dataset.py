import glob
import logging
import os
import pandas as pd
import torch
import torchvision

import preprocess.quick_voxelization
import preprocess.find_halos_and_save_particles_from_hdf5


class Dataset3D(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        sim_name,
        snap_num,
        mass_min,
        mass_max,
        n_gas_min,
        data_path,
        nvoxel,
        mode,
        transforms,
    ):
        # Check if voxel csv are ready
        ready_nvoxels = glob.glob(
            os.path.join(
                data_path,
                f"{sim_name}_SNAP{snap_num:03d}_MASS{mass_min:.2e}_{mass_max:.2e}_NGASMIN{n_gas_min}",
                f"nvoxel_{nvoxel}",
                mode,
                f"halo_*_nvoxel_{nvoxel}.csv",
            )
        )

        if len(ready_nvoxels) == 0:
            logging.info("Voxel are not ready, need to generate them.")
            ready_halos = glob.glob(
                os.path.join(
                    data_path,
                    f"{sim_name}_SNAP{snap_num:03d}_MASS{mass_min:.2e}_{mass_max:.2e}_NGASMIN{n_gas_min}",
                    f"halos",
                    f"halo_*_particles_from_hdf5.csv",
                )
            )

            if len(ready_halos) == 0:
                logging.info("Halos are not ready, need to generate them.")
                preprocess.find_halos_and_save_particles_from_hdf5.find_halos(
                    dataset_path,
                    sim_name,
                    snap_num,
                    mass_min,
                    mass_max,
                    n_gas_min,
                    data_path,
                )

            preprocess.quick_voxelization.voxelize(
                os.path.join(
                    data_path,
                    f"{sim_name}_SNAP{snap_num:03d}_MASS{mass_min:.2e}_{mass_max:.2e}_NGASMIN{n_gas_min}",
                ),
                nvoxel,
            )
            ready_nvoxels = glob.glob(
                os.path.join(
                    data_path,
                    f"{sim_name}_SNAP{snap_num:03d}_MASS{mass_min:.2e}_{mass_max:.2e}_NGASMIN{n_gas_min}",
                    f"nvoxel_{nvoxel}",
                    mode,
                    f"halo_*_nvoxel_{nvoxel}.csv",
                )
            )

        self.nvoxel = nvoxel
        self.transform = torchvision.transforms.Compose(transforms)
        self.files = ready_nvoxels

    def __getitem__(self, index):
        halo = pd.read_csv(self.files[index]).astype({"gas/nogas": bool})
        dm_sparse = halo[halo["gas/nogas"] == False]
        gas_sparse = halo[halo["gas/nogas"] == True]
        return {
            "DM": self.transform(
                torch.sparse_coo_tensor(
                    indices=dm_sparse[["Xvox", "Yvox", "Zvox"]].values.T,
                    values=dm_sparse.Mvox.values,
                    size=[self.nvoxel] * 3,
                ).to_dense()
            ).unsqueeze(0),
            "GAS": self.transform(
                torch.sparse_coo_tensor(
                    indices=gas_sparse[["Xvox", "Yvox", "Zvox"]].values.T,
                    values=gas_sparse.Mvox.values,
                    size=[self.nvoxel] * 3,
                ).to_dense()
            ).unsqueeze(0),
        }

    def __len__(self):
        return len(self.files)
