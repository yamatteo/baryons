# Find IDs of central subhalos within certain mass range
# Nov. 5, 2019
import os.path

import matplotlib as mpl

mpl.use('Agg')  # for $DISPLAY variable

import numpy as np
import pandas as pd
import illustris_python as il
import my_utils_illustris as myil
import time

find_IDs = True


def find_halos(dataset_path, sim_name, snap_num):
    MMIN = 1e12  # Illustris mass units in 1e10 Msun
    MMAX = 5e12
    NGASMIN = 500

    base_path = os.path.join(dataset_path, sim_name, "output")
    if not os.path.exists(os.path.join(base_path, f"groups_{snap_num:03d}")):
        raise IOError(f"Snapshot {snap_num} of simulation {sim_name} doesn't seem to be in {dataset_path}.")

    hsmall = il.groupcat.loadHeader(base_path, snap_num)['HubbleParam']

    mdm = {
        "TNG300-1": 4e7 / hsmall,
        "TNG100-1": 5.1e6 / hsmall,
        "TNG100-3": 3.2e8 / hsmall,
    }[sim_name]

    ###-------------------------------------------------------------------------------------------------------------------------

    if find_IDs:

        ## FIND CENTRAL SUBHALOS IDs##

        # Get IDs of central subhalos from halo catalogue
        group_fields = ['GroupFirstSub']
        groups = il.groupcat.loadHalos(base_path, snap_num, fields=group_fields)
        print('\nLoaded halo catalogue')
        filt = (
                    groups != -1)  # IDs of central subhalo for each group (-1 means no subhalos, we are loosing those for the moment)
        IDs_CENTRAL = groups[filt]

        ## FIND SUBHALOS WITH GIVEN PROPERTIES

        subgroups_fields = ['SubhaloMass', 'SubhaloLenType', 'SubhaloMassInHalfRad', 'SubhaloMassInMaxRad',
                            'SubhaloVmaxRad']
        subgroups = il.groupcat.loadSubhalos(base_path, snap_num, fields=subgroups_fields)
        print('\nLoaded subhalo catalogue')
        subgroups['index'] = np.arange(0, subgroups['count'])  # Keep track of index

        filt_mass = (subgroups['SubhaloMass'] >= MMIN * hsmall / 1e10) & (
                    subgroups['SubhaloMass'] < MMAX * hsmall / 1e10)

        filt_gas = (subgroups['SubhaloLenType'][:, 0] > NGASMIN)

        IDs_within_MASS = subgroups['index'][filt_mass & filt_gas]

        ## SAVE SUBHALOS IDs

        IDs = np.intersect1d(IDs_CENTRAL, IDs_within_MASS)
        print('Found %i subhalos' % len(IDs))
        np.save('IDs_' + sim_name + '_MASS_%.2e' % MMIN + '_%.2e' % MMAX + '_MSUN.npy', IDs)

    else:
        ## READ IDs FILE
        IDs = np.load('IDs_' + sim_name + '_MASS_%.2e' % MMIN + '_%.2e' % MMAX + '_MSUN.npy')

    ###-------------------------------------------------------------------------------------------------------------------------

    ## RETREIVE PARTICLES AND SAVE TO CVS
    Lbox = il.groupcat.loadHeader(base_path, snap_num)['BoxSize']  # kpc/h

    # number of wanted halos
    nhalos = len(IDs)

    for i in range(nhalos):
        halo_id = IDs[i]
        print('\nhalo = ', halo_id)
        start = time.time()

        # Load subhalo info
        sub_dict = il.groupcat.loadSingle(base_path, snap_num, subhaloID=halo_id)
        CM = sub_dict['SubhaloCM']
        Msub = sub_dict['SubhaloMass'] / hsmall * 1e10  # Msun
        part = sub_dict['SubhaloLenType']
        print('Msub [Msun] = %.2e' % Msub)
        print('Npart = ', part)

        # DM
        dmpos = il.snapshot.loadSubhalo(base_path, snap_num, halo_id, 'dm', fields=['Coordinates'])  # kpc/h
        dmpos_shifted = myil.utils.get_periodic_coords(CM, dmpos, Lbox)

        # GAS
        gaspos = il.snapshot.loadSubhalo(base_path, snap_num, halo_id, 'gas', fields=['Coordinates'])  # kpc/h
        gaspos_shifted = myil.utils.get_periodic_coords(CM, gaspos, Lbox)
        gasmasses = il.snapshot.loadSubhalo(base_path, snap_num, halo_id, 'gas', fields=['Masses'])

        # Write CSV file
        np_gas = len(gasmasses)
        np_dm = len(dmpos[:, 0])
        dfgas = np.array(list(
            zip(gaspos_shifted[:, 0], gaspos_shifted[:, 1], gaspos_shifted[:, 2], gasmasses, np.repeat(1, np_gas))))
        dfdm = np.array(list(
            zip(dmpos_shifted[:, 0], dmpos_shifted[:, 1], dmpos_shifted[:, 2], np.repeat(mdm / 1e10, np_dm),
                np.repeat(0, np_dm))))
        df = pd.DataFrame(np.vstack((dfgas, dfdm)))
        header = ['X', 'Y', 'Z', 'mp', 'gas/nogas']
        df.to_csv('halo_' + str(halo_id) + '_particles_from_hdf5.csv', header=header, index=False)

        # CHECK
        print('num gas particles = ', np_gas)
        print('num dm particles  = ', np_dm)

    # note:
    print('DM particle mass = ', mdm)
    print('simulation path = ', base_path)
