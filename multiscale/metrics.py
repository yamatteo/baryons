import torch
import torch.nn.functional as functional


def mse(dm, rg, pg):
    return functional.mse_loss(rg, pg)


def l1(dm, rg, pg):
    return functional.l1_loss(rg, pg)

def totalmass(dm, rg, pg):
    return torch.mean((torch.sum(pg, dim=(2, 3, 4)) - torch.sum(rg, dim=(2, 3, 4)))/torch.sum(rg, dim=(2, 3, 4)))


metrics_dict = {
    "mse": mse,
    "l1": l1,
    "totalmass": totalmass,
}
