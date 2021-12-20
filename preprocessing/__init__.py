import logging
import numpy as np
import os.path
import pandas as pd
import time
from pathlib import Path
import pickle

import torch

import illustris_python as il
import my_utils_illustris as myil

logger = logging.getLogger("baryons")

def find_ids(base_path, hsmall, snap_num, mass_min, mass_max, n_gas_min, fixed_size=None):
    logger.info(f"Finding halos in {base_path} -- snap {snap_num}...")

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

    if fixed_size is None:
        return np.intersect1d(ids_central, ids_within_mass)
    else:
        return np.intersect1d(ids_central, ids_within_mass)[:3*fixed_size]


def find_halo(base_path, halo_id, lbox, mdm, snap_num):
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


def preprocess(source_path, target_path, sim_name, snap_num, mass_min, mass_max, nvoxel, n_gas_min, fixed_size=None):
    for mode in ("train", "valid", "test"):
        (target_path / f"nvoxel_{nvoxel}" / mode).mkdir(
            parents=True, exist_ok=True
        )

    if not (source_path / f"groups_{snap_num:03d}").exists():
        raise IOError(
            f"Snapshot {snap_num} of simulation {sim_name} doesn't seem to be in {source_path}."
        )

    hsmall = il.groupcat.loadHeader(str(source_path), snap_num)[
        "HubbleParam"
    ]
    lbox = il.groupcat.loadHeader(str(source_path), snap_num)[
        "BoxSize"
    ]  # kpc/h
    mdm = {
        "TNG300-1": 4e7 / hsmall,
        "TNG100-1": 5.1e6 / hsmall,
        "TNG100-3": 3.2e8 / hsmall,
    }[sim_name]

    try:
        ids = np.load(str(target_path / "ids.npy"))
        if fixed_size is not None:
            assert len(ids) == 3*fixed_size
        logger.info("Loaded existing ids.")
    except (FileNotFoundError, AssertionError):
        ids = find_ids(
            base_path=str(source_path),
            hsmall=hsmall,
            snap_num=snap_num,
            mass_min=mass_min,
            mass_max=mass_max,
            n_gas_min=n_gas_min,
            fixed_size=fixed_size,
        )
        np.save(str(target_path / "ids.npy"), ids)

    ready_halos = list((target_path / f"nvoxel_{nvoxel}").glob("*/halo_*_coalesced.npy"))

    if (fixed_size is None and len(ids) == len(ready_halos) > 0) \
            or fixed_size is not None and 3*fixed_size == len(ids) == len(ready_halos) > 0:
        logger.info("Preprocessing is already complete.")
        stats = np.load(str(target_path / "stats.npy"))
    else:
        for halo in ready_halos:
            halo.unlink()
        logger.info(f"Preprocessing begins ({len(ids)} halos to process)...")
        stats = np.ndarray((len(ids), 6), dtype=float)

        for i, halo_id in enumerate(ids):
            dm_halo, gas_halo = find_halo(
                base_path=str(source_path),
                halo_id=halo_id,
                lbox=lbox,
                mdm=mdm,
                snap_num=snap_num,
            )

            dm_coalesced, rg_coalesced = voxelize(dm_halo, gas_halo, nvoxel=nvoxel)
            stats[i, 0] = torch.min(dm_coalesced.values())
            stats[i, 1] = torch.max(dm_coalesced.values())
            stats[i, 2] = torch.sum(dm_coalesced.values())
            stats[i, 3] = torch.min(rg_coalesced.values())
            stats[i, 4] = torch.max(rg_coalesced.values())
            stats[i, 5] = torch.sum(rg_coalesced.values())

            mode = {
                0: "train",
                1: "valid",
                2: "test",
            }[i % 3]

            torch.save(
                {"dm": dm_coalesced, "rg": rg_coalesced},
                target_path
                / f"nvoxel_{nvoxel}"
                / mode
                / f"halo_{halo_id}_coalesced.npy",

            )

        np.save(str(target_path / "stats.npy"), stats)
    return ids, stats
