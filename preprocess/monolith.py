import logging
import numpy as np
import os.path
import pandas as pd
import time
from pathlib import Path

import torch

import illustris_python as il
import my_utils_illustris as myil


def find_ids(base_path, hsmall, snap_num, mass_min, mass_max, n_gas_min):
    """TODO docstring

    Args:
        base_path: DATABASE/SIMNAME/output
        hsmall:
        snap_num:
        mass_min:
        mass_max:
        n_gas_min:

    Returns:

    """
    logging.info(f"Finding ids...")

    # Get IDs of central subhalos from halo catalogue
    groups = il.groupcat.loadHalos(base_path, snap_num, fields=["GroupFirstSub"])

    # IDs of central subhalo for each group (-1 means no subhalos, we are loosing those for the moment)
    ids_central = groups[groups != -1]

    subgroups = il.groupcat.loadSubhalos(
        base_path,
        snap_num,
        fields=[
            "SubhaloMass",
            "SubhaloLenType",
            "SubhaloMassInHalfRad",
            "SubhaloMassInMaxRad",
            "SubhaloVmaxRad",
        ],
    )

    subgroups["index"] = np.arange(0, subgroups["count"])  # Keep track of index

    filt_mass = (subgroups["SubhaloMass"] >= mass_min * hsmall / 1e10) & (
        subgroups["SubhaloMass"] < mass_max * hsmall / 1e10
    )

    filt_gas = subgroups["SubhaloLenType"][:, 0] > n_gas_min

    ids_within_mass = subgroups["index"][filt_mass & filt_gas]

    return np.intersect1d(ids_central, ids_within_mass)


def find_halo(base_path, halo_id, lbox, mdm, snap_num):
    """TODO docstring

    Args:
        base_path:
        halo_id:
        lbox:
        mdm:
        snap_num:

    Returns:

    """
    sub_dict = il.groupcat.loadSingle(base_path, snap_num, subhaloID=halo_id)
    cm = sub_dict["SubhaloCM"]

    # DM
    dmpos = il.snapshot.loadSubhalo(
        base_path, snap_num, halo_id, "dm", fields=["Coordinates"]
    )  # kpc/h
    dmpos_shifted = myil.utils.get_periodic_coords(cm, dmpos, lbox)

    dm_halo = np.hstack(
        [
            dmpos_shifted,
            np.repeat(mdm / 1e10, dmpos_shifted.shape[0]).reshape(-1, 1),
        ]
    )

    # GAS
    gaspos = il.snapshot.loadSubhalo(
        base_path, snap_num, halo_id, "gas", fields=["Coordinates"]
    )  # kpc/h
    gaspos_shifted = myil.utils.get_periodic_coords(cm, gaspos, lbox)
    gasmasses = il.snapshot.loadSubhalo(
        base_path, snap_num, halo_id, "gas", fields=["Masses"]
    )

    gas_halo = np.hstack(
        [
            gaspos_shifted,
            gasmasses.reshape(-1, 1),
        ]
    )

    return dm_halo, gas_halo


def voxelize(dm_halo, gas_halo, nvoxel):
    """TODO docstring

    Args:
        dm_halo:
        gas_halo:
        nvoxel:

    Returns:

    """
    # Split input
    dm_pos, dm_mass = dm_halo[:, 0:3], dm_halo[:, 3]
    gas_pos, gas_mass = gas_halo[:, 0:3], gas_halo[:, 3]

    # Shifting to origin
    minimum = min(np.min(dm_pos), np.min(gas_pos))
    dm_pos = dm_pos - minimum
    gas_pos = gas_pos - minimum

    # Resize to nvoxel and floor
    maximum = max(np.max(dm_pos), np.max(gas_pos))
    dm_pos = np.floor(dm_pos * (nvoxel - 1) / maximum).astype(np.int64)
    gas_pos = np.floor(gas_pos * (nvoxel - 1) / maximum).astype(np.int64)

    dm_coalesced = torch.sparse_coo_tensor(
        indices=dm_pos.T, values=dm_mass, size=(nvoxel, nvoxel, nvoxel)
    ).coalesce()
    gas_coalesced = torch.sparse_coo_tensor(
        indices=gas_pos.T, values=gas_mass, size=(nvoxel, nvoxel, nvoxel)
    ).coalesce()

    return dm_coalesced, gas_coalesced


def preprocess(opt):
    """TODO docstring

    Args:
        opt:

    Returns:

    """
    data_sim_path = (
        Path(opt.data_path)
        / f"{opt.sim_name}_SNAP{opt.snap_num:03d}_MASS{opt.mass_min:.2e}_{opt.mass_max:.2e}_NGASMIN{opt.n_gas_min}"
    )
    dataset_sim_path = Path(opt.dataset_path) / opt.sim_name
    for mode in ("train", "valid", "test"):
        (data_sim_path / f"nvoxel_{opt.nvoxel}" / mode).mkdir(
            parents=True, exist_ok=True
        )

    if not (dataset_sim_path / "output" / f"groups_{opt.snap_num:03d}").exists():
        raise IOError(
            f"Snapshot {opt.snap_num} of simulation {opt.sim_name} doesn't seem to be in {dataset_sim_path / 'output'}."
        )

    hsmall = il.groupcat.loadHeader(str(dataset_sim_path / "output"), opt.snap_num)[
        "HubbleParam"
    ]
    lbox = il.groupcat.loadHeader(str(dataset_sim_path / "output"), opt.snap_num)[
        "BoxSize"
    ]  # kpc/h
    mdm = {
        "TNG300-1": 4e7 / hsmall,
        "TNG100-1": 5.1e6 / hsmall,
        "TNG100-3": 3.2e8 / hsmall,
    }[opt.sim_name]

    try:
        logging.info("Loading ids...")
        ids = np.load(str(data_sim_path / "ids.npy"))
    except FileNotFoundError:
        logging.info(" -!- Did not found ids.npy")
        ids = find_ids(
            base_path=str(dataset_sim_path / "output"),
            hsmall=hsmall,
            snap_num=opt.snap_num,
            mass_min=opt.mass_min,
            mass_max=opt.mass_max,
            n_gas_min=opt.n_gas_min,
        )
        np.save(str(data_sim_path / "ids.npy"), ids)

    ready_dm = list((data_sim_path / f"nvoxel_{opt.nvoxel}").glob("*/halo_*_dm_coalesced.npy"))
    ready_gas = list((data_sim_path / f"nvoxel_{opt.nvoxel}").glob("*/halo_*_gas_coalesced.npy"))

    if len(ids) == len(ready_dm) == len(ready_gas) > 0:
        logging.info("Preprocessing is already complete.")
    else:
        logging.info(f"Preprocessing begins ({len(ids)} halos to process)...")

        for i, halo_id in enumerate(ids):
            dm_halo, gas_halo = find_halo(
                base_path=str(dataset_sim_path / "output"),
                halo_id=halo_id,
                lbox=lbox,
                mdm=mdm,
                snap_num=opt.snap_num,
            )

            dm_coalesced, gas_coalesced = voxelize(dm_halo, gas_halo, nvoxel=opt.nvoxel)

            mode = {
                0: "train",
                1: "valid",
                2: "test",
            }[i % 3]

            torch.save(
                dm_coalesced,
                data_sim_path
                / f"nvoxel_{opt.nvoxel}"
                / mode
                / f"halo_{halo_id}_dm_coalesced.npy",
            )
            torch.save(
                gas_coalesced,
                data_sim_path
                / f"nvoxel_{opt.nvoxel}"
                / mode
                / f"halo_{halo_id}_gas_coalesced.npy",
            )
