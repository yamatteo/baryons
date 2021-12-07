import numpy as np
import torch
from random import choice, randrange


def select_slice(*tensors, random_dims=(0, 1), orthogonal_dim: int = 2, index: int = None, weight: float = 1.0):
    """Return a slice for every tensor given, over the same index, orthogonal to `ortho_dim`.

    Accept any number of tensors as positional arguments, provided they have the same shape.
    Keyword arguments:
        - random_dims (default = (0, 1)) tuple of dimensions to select from randomly, usually batch and channels
        - ortho_dim (default = 1) the slice will be orthogonal to this dimension;
        - index (default = None) allow to choose the index of the slice, otherwise the index will be randomly selected
         among the ones that respect the weight constrain - in any case it is the same for every tensor;
        - weight (default = 1.0) the sum over the whole slice will be bigger than this value, for every tensor.
    """

    assert all(tensor.shape == tensors[0].shape for tensor in tensors), "All tensors must have the same shape"

    tensors = [ t.cpu() for t in tensors ]
    # Will sum over every dimension except the one indicated by `ortho_dim`
    dims = tuple(n for n in range(tensors[0].dim()) if n != orthogonal_dim)

    # By default, the index of the slice is not provided so it will be selected based on the weight constrain
    if index is None:
        sup = torch.max(torch.sum(tensors[0], dim=dims))
        valid_indexes = np.argwhere(torch.sum(tensors[0], dim=dims) >= sup).squeeze().tolist()
        # try:
        #     # When indexes is already defined, intersect it with the valid indexes of the new tensor
        #     indexes.intersection_update(valid_indexes)
        # except NameError:
        #     # For the first tensor in the list, `indexes` is defined as the set of valid indexes (the ones
        #     # respecting the weight constrain)
        #     indexes = set(valid_indexes)
        if isinstance(valid_indexes, int):
            index = valid_indexes
        elif isinstance(valid_indexes, list):
            index = choice(list(valid_indexes))

    for d in random_dims:
        start = randrange(tensors[0].shape[d])
        tensors = [torch.narrow(tensor, dim=d, start=start, length=1) for tensor in tensors]

    tensors = [torch.narrow(tensor, dim=orthogonal_dim, start=index, length=1) for tensor in tensors]
    return [tensor.squeeze() for tensor in tensors]
