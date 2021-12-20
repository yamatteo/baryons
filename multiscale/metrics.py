import torch.nn.functional as functional


def mse(dm, rg, pg):
    return functional.mse_loss(rg, pg)


def l1(dm, rg, pg):
    return functional.l1_loss(rg, pg)


metrics_dict = {
    "mse": mse,
    "l1": l1,
}
