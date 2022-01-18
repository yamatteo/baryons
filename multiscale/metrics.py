import itertools
import torch
import torch.nn.functional as functional

from options import opts
nvoxel = opts["nvoxel"]

def get_radial_density(voxelized_halo, nvoxel):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    center_of_mass = (
        sum(
            torch.tensor((x, y, z), device=dev) * torch.abs(voxelized_halo[:, :, x, y ,z])
            for (x, y, z) in itertools.product(range(nvoxel), range(nvoxel), range(nvoxel))
        )
    ) / torch.sum(torch.abs(voxelized_halo), dim=(2, 3, 4))
    density_at_distance = torch.zeros(voxelized_halo.size(0), voxelized_halo.size(0), 2*(nvoxel+2), device=dev)
    for d, m in [(((torch.tensor((x, y, z), device=dev) - center_of_mass).float().norm() - 1).ceil().abs().int().item(), voxelized_halo[:, :, x, y, z]) for (x, y, z) in itertools.product(range(nvoxel), range(nvoxel), range(nvoxel))]:
        try:
            density_at_distance[:, :, d] += m / (d+1)**2
        except IndexError:
            print(f"{center_of_mass=}")
            print(f"{density_at_distance=}")
            print(f"{(d, m)=}")
    return density_at_distance

def get_radial_density2(voxelized_halo, nvoxel):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    coords = torch.cartesian_prod(
        torch.arange(0, nvoxel),
        torch.arange(0, nvoxel),
        torch.arange(0, nvoxel)
    )
    center_of_mass = torch.sum(
        functional.relu(
            coords * voxelized_halo.view(
                voxelized_halo.size(0),
                voxelized_halo.size(1),
                nvoxel, nvoxel, nvoxel,
                1
            ).expand(
                voxelized_halo.size(0),
                voxelized_halo.size(1),
                nvoxel, nvoxel, nvoxel,
                3
            ).to(device=dev)
        ), dim=(2, 3, 4)
    ) / torch.sum(functional.relu(voxelized_halo), dim=(2, 3, 4))
    density_at_distance = torch.zeros(voxelized_halo.size(0), voxelized_halo.size(0), 2*(nvoxel+2), device=dev)
    for d, m in [(((torch.tensor((x, y, z), device=dev) - center_of_mass).float().norm() - 1).ceil().abs().int().item(), voxelized_halo[:, :, x, y, z]) for (x, y, z) in itertools.product(range(nvoxel), range(nvoxel), range(nvoxel))]:
        try:
            density_at_distance[:, :, d] += m / (d+1)**2
        except IndexError:
            print(f"{center_of_mass=}")
            print(f"{density_at_distance=}")
            print(f"{(d, m)=}")
    return density_at_distance

def mse(dm, rg, pg):
    return functional.mse_loss(rg, pg)


def l1(dm, rg, pg):
    return functional.l1_loss(rg, pg)

def totalmass(dm, rg, pg):
    return torch.mean((torch.sum(pg, dim=(2, 3, 4)) - torch.sum(rg, dim=(2, 3, 4)))/torch.sum(rg, dim=(2, 3, 4)))

def radial_density(dm, rg, pg):
    return functional.l1_loss(
        get_radial_density(rg, nvoxel),
        get_radial_density(pg, nvoxel),
    )



metrics_dict = {
    "mse": mse,
    "l1": l1,
    "totalmass": totalmass,
    "radial": radial_density,
}
