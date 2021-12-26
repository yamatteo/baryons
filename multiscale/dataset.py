import pickle
from pathlib import Path

import torch
import torch.utils.data


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


# def sparse_to_dense_log_distribution(sparse_tensor):
#     sparse_tensor = sparse_tensor.coalesce()
#     values = sparse_tensor.values()
#     grain = torch.min(values)
#     log_values = torch.log(values / grain) + 1
#     log_total = torch.sum(log_values)
#     log_distribution = torch.sparse_coo_tensor(
#         indices=sparse_tensor.indices(),
#         values=torch.div(log_values, log_total),
#         size=sparse_tensor.size()
#     ).coalesce().to_dense().unsqueeze(0)
#     return log_distribution, log_total, grain
#
#
# def dld_to_mass(log_distribution, log_total, grain):
#     return torch.exp(log_distribution * log_total - 1) * grain
#
#
# class LogarithmicDataset(torch.utils.data.Dataset):
#     def __init__(self, path, fixed_size=None):
#         super(LogarithmicDataset, self).__init__()
#         if isinstance(path, str):
#             path = Path(path)
#         self.files = list(path.glob("halo_*_coalesced.npy"))
#         if fixed_size is not None:
#             assert len(self.files) >= fixed_size
#             self.files = self.files[:fixed_size]
#
#     def __getitem__(self, index):
#         halo_dict = torch.load(self.files[index])
#         dm, dm_log_total, dm_grain = sparse_to_dense_log_distribution(halo_dict["dm"])
#         rg, rg_log_total, rg_grain = sparse_to_dense_log_distribution(halo_dict["rg"])
#         return {
#             "dm": dm,
#             "dm_log_total": dm_log_total,
#             "dm_grain": dm_grain,
#             "rg": rg,
#             "rg_log_total": rg_log_total,
#             "rg_grain": rg_grain,
#         }
#
#     def __len__(self):
#         return len(self.files)
