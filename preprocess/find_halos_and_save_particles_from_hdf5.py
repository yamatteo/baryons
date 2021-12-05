# Find IDs of central subhalos within certain mass range
# Nov. 5, 2019
import logging
import numpy as np
import os.path
import pandas as pd
import time

import illustris_python as il
import my_utils_illustris as myil


def find_ids(base_path, snap_num, mass_min, mass_max, hsmall, n_gas_min):
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


def find_halos(
    dataset_path, sim_name, snap_num, mass_min, mass_max, n_gas_min, data_path
):
    # mass_min = 1e12  # Illustris mass units in 1e10 Msun
    # mass_max = 5e12
    # n_gas_min = 500

    base_path = os.path.join(dataset_path, sim_name, "output")
    if not os.path.exists(os.path.join(base_path, f"groups_{snap_num:03d}")):
        raise IOError(
            f"Snapshot {snap_num} of simulation {sim_name} doesn't seem to be in {base_path}."
        )

    hsmall = il.groupcat.loadHeader(base_path, snap_num)["HubbleParam"]

    mdm = {
        "TNG300-1": 4e7 / hsmall,
        "TNG100-1": 5.1e6 / hsmall,
        "TNG100-3": 3.2e8 / hsmall,
    }[sim_name]

    prep_path = os.path.join(
        f"{sim_name}_SNAP{snap_num:03d}_MASS{mass_min:.2e}_{mass_max:.2e}_NGASMIN{n_gas_min}",
        "halos",
    )
    os.makedirs(os.path.join(data_path, prep_path), exist_ok=True)

    if os.path.exists(os.path.join(data_path, prep_path, "ids.npy")):
        logging.info("Loading previous ids...")
        ids = np.load(os.path.join(data_path, prep_path, "ids.npy"))
    else:
        ids = find_ids(base_path, snap_num, mass_min, mass_max, hsmall, n_gas_min)
        np.save(os.path.join(data_path, prep_path, "ids.npy"), ids)

    logging.info(f"Finding halos ({mass_min = }, {mass_max = }, {n_gas_min = })...")

    ## RETREIVE PARTICLES AND SAVE TO CVS
    Lbox = il.groupcat.loadHeader(base_path, snap_num)["BoxSize"]  # kpc/h

    # number of wanted halos
    nhalos = len(ids)

    for i in range(nhalos):
        halo_id = ids[i]
        # print('\nhalo = ', halo_id)
        start = time.time()

        # Load subhalo info
        sub_dict = il.groupcat.loadSingle(base_path, snap_num, subhaloID=halo_id)
        CM = sub_dict["SubhaloCM"]
        # Msub = sub_dict['SubhaloMass'] / hsmall * 1e10  # Msun
        # part = sub_dict['SubhaloLenType']
        # print('Msub [Msun] = %.2e' % Msub)
        # print('Npart = ', part)

        # DM
        dmpos = il.snapshot.loadSubhalo(
            base_path, snap_num, halo_id, "dm", fields=["Coordinates"]
        )  # kpc/h
        dmpos_shifted = myil.utils.get_periodic_coords(CM, dmpos, Lbox)

        # GAS
        gaspos = il.snapshot.loadSubhalo(
            base_path, snap_num, halo_id, "gas", fields=["Coordinates"]
        )  # kpc/h
        gaspos_shifted = myil.utils.get_periodic_coords(CM, gaspos, Lbox)
        gasmasses = il.snapshot.loadSubhalo(
            base_path, snap_num, halo_id, "gas", fields=["Masses"]
        )

        # Write CSV file
        np_gas = len(gasmasses)
        np_dm = len(dmpos[:, 0])
        dfgas = np.array(
            list(
                zip(
                    gaspos_shifted[:, 0],
                    gaspos_shifted[:, 1],
                    gaspos_shifted[:, 2],
                    gasmasses,
                    np.repeat(1, np_gas),
                )
            )
        )
        dfdm = np.array(
            list(
                zip(
                    dmpos_shifted[:, 0],
                    dmpos_shifted[:, 1],
                    dmpos_shifted[:, 2],
                    np.repeat(mdm / 1e10, np_dm),
                    np.repeat(0, np_dm),
                )
            )
        )
        df = pd.DataFrame(np.vstack((dfgas, dfdm)))
        df.to_csv(
            os.path.join(
                data_path, prep_path, f"halo_{halo_id}_particles_from_hdf5.csv"
            ),
            header=["X", "Y", "Z", "mp", "gas/nogas"],
            index=False,
        )

        # CHECK
        # print('num gas particles = ', np_gas)
        # print('num dm particles  = ', np_dm)

    # note:
    # print('DM particle mass = ', mdm)
    # print('simulation path = ', base_path)
