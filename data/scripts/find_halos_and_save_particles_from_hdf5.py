# Find IDs of central subhalos within certain mass range
# Nov. 5, 2019

import matplotlib as mpl
mpl.use('Agg')  # for $DISPLAY variable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import illustris_python as il
import my_utils_illustris as myil
import time

find_IDs = True

SIMNAME = 'TNG300-1'
MMIN = 1e12 # Illustris mass units in 1e10 Msun
MMAX = 5e12
NGASMIN = 500

basePath = '/gpfsdswork/dataset/tng-project/'+SIMNAME+'/output'
snapNum = 99
hsmall = il.groupcat.loadHeader(basePath, snapNum)['HubbleParam']

if   SIMNAME == 'TNG300-1':       # Msun
     mdm = 4e7/hsmall
elif SIMNAME == 'TNG100-1':
     mdm = 5.1e6/hsmall
elif SIMNAME == 'TNG100-3':
     mdm = 3.2e8/hsmall

###-------------------------------------------------------------------------------------------------------------------------

if find_IDs:

   ## FIND CENTRAL SUBHALOS IDs##

   # Get IDs of central subhalos from halo catalogue
   group_fields = ['GroupFirstSub']
   groups = il.groupcat.loadHalos(basePath, snapNum, fields=group_fields)
   print('\nLoaded halo catalogue')
   filt = (groups!=-1) # IDs of central subhalo for each group (-1 means no subhalos, we are loosing those for the moment)
   IDs_CENTRAL = groups[filt]

   ## FIND SUBHALOS WITH GIVEN PROPERTIES

   subgroups_fields  = ['SubhaloMass', 'SubhaloLenType', 'SubhaloMassInHalfRad', 'SubhaloMassInMaxRad', 'SubhaloVmaxRad']
   subgroups = il.groupcat.loadSubhalos(basePath, snapNum, fields=subgroups_fields)
   print('\nLoaded subhalo catalogue')
   subgroups['index'] = np.arange(0, subgroups['count']) # Keep track of index

   filt_mass = (subgroups['SubhaloMass']>=MMIN*hsmall/1e10) & (subgroups['SubhaloMass']<MMAX*hsmall/1e10)

   filt_gas = (subgroups['SubhaloLenType'][:,0]>NGASMIN)

   IDs_within_MASS = subgroups['index'][filt_mass & filt_gas]

   ## SAVE SUBHALOS IDs

   IDs = np.intersect1d(IDs_CENTRAL, IDs_within_MASS)
   print('Found %i subhalos' % len(IDs))
   np.save('IDs_'+SIMNAME+'_MASS_%.2e' %MMIN+'_%.2e' %MMAX+'_MSUN.npy', IDs)

else:
   ## READ IDs FILE
   IDs = np.load('IDs_'+SIMNAME+'_MASS_%.2e' %MMIN+'_%.2e' %MMAX+'_MSUN.npy')

###-------------------------------------------------------------------------------------------------------------------------

## RETREIVE PARTICLES AND SAVE TO CVS
Lbox = il.groupcat.loadHeader(basePath, snapNum)['BoxSize'] # kpc/h

# number of wanted halos
nhalos = len(IDs)

for i in range(nhalos):

    halo_id = IDs[i] 
    print('\nhalo = ', halo_id)
    start = time.time()

    # Load subhalo info
    sub_dict = il.groupcat.loadSingle(basePath, snapNum, subhaloID=halo_id)
    CM   = sub_dict['SubhaloCM']
    Msub = sub_dict['SubhaloMass']/hsmall*1e10            # Msun
    part = sub_dict['SubhaloLenType']
    print('Msub [Msun] = %.2e' %Msub)
    print('Npart = ', part)

    # DM
    dmpos         = il.snapshot.loadSubhalo(basePath, snapNum, halo_id, 'dm',  fields=['Coordinates'])    # kpc/h
    dmpos_shifted = myil.utils.get_periodic_coords(CM, dmpos, Lbox)

    # GAS
    gaspos         = il.snapshot.loadSubhalo(basePath, snapNum, halo_id, 'gas', fields=['Coordinates'])   # kpc/h
    gaspos_shifted = myil.utils.get_periodic_coords(CM, gaspos, Lbox)
    gasmasses = il.snapshot.loadSubhalo(basePath, snapNum, halo_id, 'gas', fields=['Masses'])

    # Write CSV file
    np_gas = len(gasmasses)
    np_dm  = len(dmpos[:,0])
    dfgas = np.array(list(zip(gaspos_shifted[:,0], gaspos_shifted[:,1], gaspos_shifted[:,2], gasmasses, np.repeat(1,np_gas))))
    dfdm  = np.array(list(zip( dmpos_shifted[:,0],  dmpos_shifted[:,1],  dmpos_shifted[:,2], np.repeat(mdm/1e10,np_dm), np.repeat(0,np_dm))))
    df = pd.DataFrame(np.vstack((dfgas,dfdm)))
    header = ['X', 'Y', 'Z', 'mp', 'gas/nogas']
    df.to_csv('halo_'+str(halo_id)+'_particles_from_hdf5.csv', header=header, index=False)

    # CHECK
    print('num gas particles = ', np_gas)
    print('num dm particles  = ', np_dm)


# note:
print('DM particle mass = ', mdm)
print('simulation path = ', basePath)


#--------------------------------------------------------------------------------------------------------------------------------------------------------



'''
# Load all particles/cells of one type for a specific subhalo
halo_id = 16921
gaspos    = il.snapshot.loadSubhalo(basePath, snapNum, halo_id, 'gas', fields=['Coordinates'])
gasmasses = il.snapshot.loadSubhalo(basePath, snapNum, halo_id, 'gas', fields=['Masses'])
dmpos     = il.snapshot.loadSubhalo(basePath, snapNum, halo_id, 'dm',  fields=['Coordinates'])

# Write CSV file
np_gas = len(gasmasses)
np_dm  = len(dmpos[:,0])
dfgas = np.array(list(zip(gaspos[:,0], gaspos[:,1], gaspos[:,2], gasmasses, np.repeat(1,np_gas))))
dfdm  = np.array(list(zip(dmpos[:,0], dmpos[:,1], dmpos[:,2], np.repeat(mdm/1e10,np_dm), np.repeat(0,np_dm))))
df = pd.DataFrame(np.vstack((dfgas,dfdm)))
header = ['X', 'Y', 'Z', 'mp', 'gas/nogas']
df.to_csv('halo_'+str(halo_id)+'_particles_from_hdf5.csv', header=header, index=False)

# Printing checks
print()
print('num gas particles = ', np_gas)
print('num dm particles  = ', np_dm)
'''



