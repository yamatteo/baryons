import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path
import h5py
import glob
from scipy import optimize
#-----------------------------------------------------------------------------------------------------------------------------------
Ggrav = 4.302e-6 # kpc Msun-1 (km/s)^2
rho_crit   = 2.77519737e11 * 1e-9 #  (Msun/h)/(kpc/h)**3
const = 4./3.*np.pi
#-----------------------------------------------------------------------------------------------------------------------------------
def get_periodic_coords(center, pos, Lbox):

    shifted_pos  = pos - center

    Lbox_half = Lbox/2.
    periodic_filt_1 = shifted_pos < -Lbox_half  #>   Lbox/2.
    periodic_filt_2 = shifted_pos >  Lbox_half

    len_filt_1 = len(shifted_pos[periodic_filt_1])
    len_filt_2 = len(shifted_pos[periodic_filt_2])

    if (len_filt_1==0 and len_filt_2==0): # No need to correct
       # print('no periodic boundary conditions used')
       pass

    else:   # At least a correction needed
       # print('boundary conditions used')

       if len_filt_1!=0:

          if len(shifted_pos[:,0][periodic_filt_1[:,0]])!=0:
             shifted_pos[:,0] = [dx+Lbox if dx<-Lbox_half else dx for dx in shifted_pos[:,0]]

          if len(shifted_pos[:,1][periodic_filt_1[:,1]])!=0:
             shifted_pos[:,1] = [dx+Lbox if dx<-Lbox_half else dx for dx in shifted_pos[:,1]]

          if len(shifted_pos[:,2][periodic_filt_1[:,2]])!=0:
             shifted_pos[:,2] = [dx+Lbox if dx<-Lbox_half else dx for dx in shifted_pos[:,2]]

       if len_filt_2!=0:

          if len(shifted_pos[:,0][periodic_filt_2[:,0]])!=0:
             shifted_pos[:,0] = [dx-Lbox if dx>Lbox_half else dx for dx in shifted_pos[:,0]]

          if len(shifted_pos[:,1][periodic_filt_2[:,1]])!=0:
             shifted_pos[:,1] = [dx-Lbox if dx>Lbox_half else dx for dx in shifted_pos[:,1]]

          if len(shifted_pos[:,2][periodic_filt_2[:,2]])!=0:
             shifted_pos[:,2] = [dx-Lbox if dx>Lbox_half else dx for dx in shifted_pos[:,2]]

    return shifted_pos # kpc/h

#-----------------------------------------------------------------------------------------------------------------------------------
def get_squared_distances(dist, Lbox):
    """ OK
        Check particle distances and corrects them according to bondary conditions given Lbox
        Note that Illustris has [0,Lbox] coordinates

        HOWTO SubGroups:
        import illustris_python as il
        import my_utils_illustris as myil
        dmpos = il.snapshot.loadSubhalo(basePath, snapNum, halo_id, 'dm', fields=['Coordinates'])
        dist  = dmpos - subgroups['SubhaloCM'][IDsubhalo]
        dist2 = myil.utils.get_squared_distances(dist, Lbox)
    """
    Lbox_half = Lbox/2.
    periodic_filt_1 = dist < -Lbox_half  #>   Lbox/2.
    periodic_filt_2 = dist >  Lbox_half

    len_filt_1 = len(dist[periodic_filt_1])
    len_filt_2 = len(dist[periodic_filt_2])

    if (len_filt_1==0 and len_filt_2==0): # No need to correct
       print('no periodic boundary conditions used')
       dist2 = np.sum(dist**2.0,axis=1)
    else:   # At least a correction needed
       print('boundary conditions used')
       if len_filt_1!=0:
          if len(dist[:,0][periodic_filt_1[:,0]])!=0:
             dist[:,0] = map(lambda dx: dx+Lbox if dx<-Lbox_half else dx, dist[:,0])
          if len(dist[:,1][periodic_filt_1[:,1]])!=0:
             dist[:,1] = map(lambda dx: dx+Lbox if dx<-Lbox_half else dx, dist[:,1]) 
          if len(dist[:,2][periodic_filt_1[:,2]])!=0:
             dist[:,2] = map(lambda dx: dx+Lbox if dx<-Lbox_half else dx, dist[:,2])

       if len_filt_2!=0:

          if len(dist[:,0][periodic_filt_2[:,0]])!=0:
             dist[:,0] = [dx-Lbox if dx>Lbox_half else dx for dx in dist[:,0]]

          if len(dist[:,1][periodic_filt_2[:,1]])!=0:
             dist[:,1] = [dx-Lbox if dx>Lbox_half else dx for dx in dist[:,1]]

          if len(dist[:,2][periodic_filt_2[:,2]])!=0:
             dist[:,2] = [dx-Lbox if dx>Lbox_half else dx for dx in dist[:,2]]

       dist2 = np.sum(dist**2.0,axis=1)

    return dist2 # kpc/h
#-----------------------------------------------------------------------------------------------------------------------------------
