import random
from pathlib import Path

import torch
import torch.utils.data


def unpack(halo_dict):
    return {
        "dm": halo_dict["dm"].to_dense().unsqueeze(0),
        "rg": halo_dict["rg"].to_dense().unsqueeze(0),
    }


class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, path, fixed_size=None):
        if isinstance(path, str):
            path = Path(path)
        # self.files = list(path.glob("halo_*_coalesced.npy"))
        self.halos = [
            unpack(torch.load(filename))
            for filename in list(path.glob("halo_*_coalesced.npy"))
        ]
        # if fixed_size is not None:
        #     assert len(self.files) >= fixed_size
        #     self.files = self.files[:fixed_size]

    def __getitem__(self, index):
        # halo_dict = torch.load(self.files[index])
        # return {
        #     "dm": halo_dict["dm"].to_dense().unsqueeze(0),
        #     "rg": halo_dict["rg"].to_dense().unsqueeze(0),
        # }
        return self.halos[index]

    def get_random(self):
        return self.halos[random.randint(0, len(self.halos) - 1)]
        # return self.__getitem__(random.randint(0, len(self.files) - 1))


    def __len__(self):
        return len(self.halos)
        # return len(self.files)
