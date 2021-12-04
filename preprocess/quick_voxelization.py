# Aug 26, 2019
# Voxelization with FLOOR
#
# input:  particles CSV files
# output: non-zero voxels CSV files
import os.path

import numpy as np
import pandas as pd
import math
import glob


def voxelize(path, nvoxel):
    particle_filenames = glob.glob(os.path.join(path, "halos", "halo_*_particles_from_hdf5.csv"))
    os.makedirs(os.path.join(path, f"nvoxel_{nvoxel}"), exist_ok=True)

    for particle_filename in particle_filenames:
        # halo_id = particle_filename[5:-24]
        # voxel_filename = 'halo_' + halo_id + '_nvoxel_' + str(nvoxel) + '.csv'  # output file
        voxel_filename = os.path.basename(particle_filename).replace("particles_from_hdf5", f"nvoxel_{nvoxel}")

        data = pd.read_csv(particle_filename)

        # Shifting to origin
        data['Xshifted'] = data['X'] - np.min(data['X'])
        data['Yshifted'] = data['Y'] - np.min(data['Y'])
        data['Zshifted'] = data['Z'] - np.min(data['Z'])

        data = data.drop(columns=['X', 'Y', 'Z'])

        max_coord = max(data[['Xshifted', 'Yshifted', 'Zshifted']].max())

        data['Xnorm'] = data['Xshifted'] / max_coord
        data['Ynorm'] = data['Yshifted'] / max_coord
        data['Znorm'] = data['Zshifted'] / max_coord

        bin_length = 1.0 / (nvoxel - 1)
        data['Xvox'] = (data['Xnorm'] / bin_length).apply(math.floor)
        data['Yvox'] = (data['Ynorm'] / bin_length).apply(math.floor)
        data['Zvox'] = (data['Znorm'] / bin_length).apply(math.floor)

        data = data.drop(columns=['Xnorm', 'Ynorm', 'Znorm', 'Xshifted', 'Yshifted', 'Zshifted'])

        grouped = data.groupby(by=['gas/nogas', 'Xvox', 'Yvox', 'Zvox'])

        df_out = pd.DataFrame(list(grouped.groups.keys()), columns=['gas/nogas', 'Xvox', 'Yvox', 'Zvox'])
        df_out['Mvox'] = np.asarray(grouped.sum())
        df_out.to_csv(os.path.join(path, f"nvoxel_{nvoxel}", voxel_filename), index=False)
