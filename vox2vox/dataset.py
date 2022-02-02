import random
from pathlib import Path

import torch
import torch.utils.data
import torch.nn.functional as functional

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, blur=None, crop=None, fold=None, nvoxel=None, batch_size=None):
        if isinstance(path, str):
            path = Path(path)
        self.halos = [
            torch.load(filename)
            for filename in list(path.glob("halo_*_voxels.npy"))
        ]
        self.blur = blur
        self.crop = crop
        self.fold = fold
        self.nvoxel = nvoxel
        self.batch_size = batch_size

    def __getitem__(self, index):
        halos = self.halos[index]
        if self.blur:
            halos = [functional.avg_pool3d(x, kernel_size=3, stride=1, padding=1) for x in halos]
        if self.crop:
            # From shape (1, n, n, n) to (1, m, m, m)
            (n, m) = self.crop
            cx, cy, cz = random.randrange(n - m), random.randrange(n - m), random.randrange(n - m)
            halos = [x[:, cx:(cx + m), cy:(cy + m), cz:(cz + m)] for x in halos]
        # if self.fold == 2:
        #     B, N = self.batch_size, self.nvoxel // 2
        #     halos = [
        #         t.view(B, 1, N, N, N, N, N, N)
        #         .permute(0, 2, 4, 6, 1, 3, 5, 7)
        #         .reshape(8*B, 1, N, N, N)
        #         for t in halos
        #     ]
        return halos

    def get_random(self):
        return self.halos[random.randrange(len(self.halos))]

    def __len__(self):
        return len(self.halos)
