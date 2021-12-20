import pickle
from pathlib import Path

import torch
import torch.utils.data

def to_dense_log_distribution(sparse_tensor):
    values = sparse_tensor.coalesce().values()
    grain = torch.min(values)
    log_values = torch.log(values / (grain / 2))
    log_total = torch.sum(log_values)
    dm_log_distribution = torch.sparse_coo_tensor(
        indices=sparse_tensor.indices(),
        values=torch.div(log_values, log_total),
        size=sparse_tensor.size()
    ).coalesce().to_dense().unsqueeze(0)


class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, path, fixed_size=None):
        if isinstance(path, str):
            path = Path(path)
        self.files = list(path.glob("halo_*_coalesced.npy"))
        if fixed_size is not None:
            assert len(self.files) >= fixed_size
            self.files = self.files[:fixed_size]

    def __getitem__(self, index):
        halo_dict = torch.load(self.files[index])
        return {
            "dm": halo_dict["dm"].to_dense().unsqueeze(0),
            "rg": halo_dict["rg"].to_dense().unsqueeze(0),
        }

    def __len__(self):
        return len(self.files)


class LogarithmicDataset(torch.utils.data.Dataset):
    def __init__(self, path, fixed_size=None):
        super(LogarithmicDataset, self).__init__()
        if isinstance(path, str):
            path = Path(path)
        self.files = list(path.glob("halo_*_coalesced.npy"))
        if fixed_size is not None:
            assert len(self.files) >= fixed_size
            self.files = self.files[:fixed_size]
        stats = pickle.load(open(path / ".." / ".." / "stats.npy", "rb"))
        self.dm_grain = stats["dm_min"]
        self.rg_grain = stats["rg_min"]
        print(stats)

    def __getitem__(self, index):
        halo_dict = torch.load(self.files[index])
        dm_values = torch.log(halo_dict["dm"].coalesce().values() / (self.dm_grain / 2))
        rg_values = torch.log(halo_dict["rg"].coalesce().values() / (self.rg_grain / 2))
        return {
            "dm": torch.sparse_coo_tensor(
                indices=halo_dict["dm"].coalesce().indices(),
                values=dm_values,
                size=halo_dict["dm"].size(),
            ).coalesce().to_dense().unsqueeze(0),
            "rg": torch.sparse_coo_tensor(
                indices=halo_dict["rg"].coalesce().indices(),
                values=rg_values,
                size=halo_dict["rg"].size(),
            ).coalesce().to_dense().unsqueeze(0),
        }

    def __len__(self):
        return len(self.files)
