from pathlib import Path

import torch


def unpack(halo_dict):
    return (
        halo_dict["dm"].to_dense().unsqueeze(0),
        halo_dict["gas"].to_dense().unsqueeze(0),
    )


class BlockworkDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        if isinstance(path, str):
            path = Path(path)
        self.files = list(path.glob("halo_*_coalesced.npy"))

    def __getitem__(self, index):
        """
        Returns:
            tuple(
                Tensor: (1, NVOXEL, NVOXEL, NVOXEL),  --> dark matter
                Tensor: (1, NVOXEL, NVOXEL, NVOXEL),  --> gas
            )
        """
        return unpack(torch.load(self.files[index]))

    def __len__(self):
        return len(self.files)
