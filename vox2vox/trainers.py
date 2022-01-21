import torch
import torch.nn as nn
import torch.nn.functional as functional
import functools


class L1ProjectionsTrainer:
    def __init__(self):
        pass

    def __call__(self, dm, rg, pg):
        return (
                functional.l1_loss(rg, pg)
                + functional.l1_loss(torch.sum(rg, dim=-1), torch.sum(pg, dim=-1))
                + functional.l1_loss(torch.sum(rg, dim=-2), torch.sum(pg, dim=-2))
                + functional.l1_loss(torch.sum(rg, dim=-3), torch.sum(pg, dim=-3))
                + functional.l1_loss(torch.sum(rg, dim=(-1, -2)), torch.sum(pg, dim=(-1, -2)))
                + functional.l1_loss(torch.sum(rg, dim=(-1, -3)), torch.sum(pg, dim=(-1, -3)))
                + functional.l1_loss(torch.sum(rg, dim=(-2, -3)), torch.sum(pg, dim=(-2, -3)))
                + functional.l1_loss(torch.sum(rg, dim=(-1, -2, -3)), torch.sum(pg, dim=(-1, -2, -3)))
        )



class L1SumTrainer:
    def __init__(self):
        pass

    def __call__(self, dm, rg, pg):
        return (
                functional.l1_loss(rg, pg, reduction="sum")
        )
