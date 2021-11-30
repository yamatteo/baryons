# 23 May, 2020 
# Creates dataset for 2D training from 3D images
# updated: uses torch sparse util
#


import glob
import numpy as np
import pandas as pd
#import sparse
import torch
import os

input_folder  = './TNG300-1_MASS_1.00e+12_5.00e+12_MSUN/' 
output_folder = './TNG300-1_MASS_1.00e+12_5.00e+12_MSUN/2D-images/'


#def create_box_channel(data, n_voxel=None):
#    return sparse.COO(data[['Xvox', 'Yvox', 'Zvox']].values.T, data.Mvox.values, shape=((n_voxel,) * 3)).todense()

def create_box_channel_pytorch(data, n_voxel=None):
    image = torch.sparse_coo_tensor(indices=data[['Xvox', 'Yvox', 'Zvox']].values.T, values = data.Mvox.values, size=[n_voxel, n_voxel, n_voxel])
    return image.to_dense()


def build_2D_dataset(n_voxel, in_folder, out_folder):

    if not os.path.exists(output_folder):
           os.makedirs(output_folder)

    files = sorted(glob.glob(in_folder + 'halo_11*_nvoxel_*.csv')) #+ 'halo_*_nvoxel_*.csv'))
    print('Found ', len(files), 'files.')
    for f in files:
        print('Reading file', f)  # logging.info('Reading file', f)
        data_3d = pd.read_csv(f)
        data_3d = data_3d.astype({'gas/nogas': bool})
        dm  = create_box_channel_pytorch(data_3d[data_3d['gas/nogas']==False], n_voxel=n_voxel)
        gas = create_box_channel_pytorch(data_3d[data_3d['gas/nogas']==True],  n_voxel=n_voxel)

        # numpy flatten maybe helpful
        for index, _ in enumerate(dm):
            np.save(out_folder + f[len(f)-f[::-1].find('/'):-4] + '_dm_'  + str(index) + '.npy', dm[index])
            np.save(out_folder + f[len(f)-f[::-1].find('/'):-4] + '_gas_' + str(index) + '.npy', gas[index])


if __name__ == '__main__':
    build_2D_dataset(n_voxel=256, in_folder=input_folder, out_folder=output_folder)
