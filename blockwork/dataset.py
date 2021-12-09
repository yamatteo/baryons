from pathlib import Path

import torch
import torch.nn.functional as F


def unpack(halo_dict):
    return (
        halo_dict["dm"].to_dense().unsqueeze(0),
        halo_dict["gas"].to_dense().unsqueeze(0),
    )


def extend(inputs, margin):
    return (
        F.pad(inputs[0], (margin, margin, margin, margin, margin, margin)),
        F.pad(inputs[1], (margin, margin, margin, margin, margin, margin)),
    )


def crop(inputs, block_side, world_side, margin):
    opening_corner = torch.randint(low=0, high=world_side-block_side, size=(3,))
    closing_corner = opening_corner + block_side + 2*margin
    return (
        inputs[0][
            :,
            opening_corner[0] : closing_corner[0],
            opening_corner[1] : closing_corner[1],
            opening_corner[2] : closing_corner[2],
        ],
        inputs[1][
            :,
            opening_corner[0] : closing_corner[0],
            opening_corner[1] : closing_corner[1],
            opening_corner[2] : closing_corner[2],
        ],
    )


class BlockworkDataset(torch.utils.data.Dataset):
    """
    Use with batch_size = None
    """
    def __init__(self, path, block_side, world_side, margin, batch_size):
        self.block_side = block_side
        self.world_side = world_side
        self.margin = margin
        self.batch_size = batch_size
        if isinstance(path, str):
            path = Path(path)
        self.files = list(path.glob("halo_*_coalesced.npy"))

    def __getitem__(self, index):
        """
        Returns:
            (
                Tensor: (BATCH_SIZE, 1, BLOCK_SIDE+2*MARGIN, BLOCK_SIDE+2*MARGIN, BLOCK_SIDE+2*MARGIN),  --> dark matter
                Tensor: (BATCH_SIZE, 1, BLOCK_SIDE+2*MARGIN, BLOCK_SIDE+2*MARGIN, BLOCK_SIDE+2*MARGIN),  --> gas
            )
        """
        halos = extend(unpack(torch.load(self.files[index])), self.margin)
        dms = list()
        gss = list()
        for batch in range(self.batch_size):
            dm, gs = crop(halos, self.block_side, self.world_side, self.margin)
            dms.append(dm)
            gss.append(gs)
        return torch.stack(dms), torch.stack(gss)

    def __len__(self):
        return len(self.files)
