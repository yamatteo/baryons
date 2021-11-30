import numpy as np
import torch
from random import choice


def select_slice(*tensors, orthogonal_dim: int = 1, index: int = None, weight: float = 1.0):
    """Return a slice for every tensor given, over the same index, orthogonal to `ortho_dim`.

    Accept any number of tensors as positional arguments, provided they have the same shape.
    Keyword arguments:
        - ortho_dim (default = 1) the slice will be orthogonal to this dimension;
        - index (default = None) allow to choose the index of the slice, otherwise the index will be randomly selected
         among the ones that respect the weight constrain - in any case it is the same for every tensor;
        - weight (default = 1.0) the sum over the whole slice will be bigger than this value, for every tensor.
    """

    assert all(tensor.shape == tensors[0].shape for tensor in tensors), "All tensors must have the same shape"

    # Will sum over every dimension except the one indicated by `ortho_dim`
    dims = tuple(n for n in range(tensors[0].dim()) if n != orthogonal_dim)

    # By default, the index of the slice is not provided so it will be selected based on the weight constrain
    if index is None:
        for tensor in tensors:
            valid_indexes = np.argwhere(torch.sum(tensor[:, :, :, :], dim=dims) > weight).squeeze().tolist()
            try:
                # When indexes is already defined, intersect it with the valid indexes of the new tensor
                indexes.intersection_update(valid_indexes)
            except NameError:
                # For the first tensor in the list, `indexes` is defined as the set of valid indexes (the ones
                # respecting the weight constrain)
                indexes = set(valid_indexes)
            print(f"{indexes = }")
        index = torch.tensor([choice(list(indexes))], dtype=torch.int32)

    return list(torch.index_select(tensor, orthogonal_dim, index) for tensor in tensors)
