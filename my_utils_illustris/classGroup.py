# Dec 28, 2019
#
# Class for groups/halos (i.e. outputs of FOF) 
# 
# Gathers particles of halo given its ID
# Shifts them and accounts for periodic box if needed
# Calculates M_delta for any delta
# Plots images
#

import matplotlib as mpl
#mpl.use('Agg')  # for $DISPLAY variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import illustris_python as il
import my_utils_illustris as myil
import utils_sparsity as utils_sp
from scipy import optimize
import time
import math
import glob

class Group:

    def __init__(self, path, snap, simname, ID, load_ext_delta_files_DM=True, load_ext_delta_files_BAR=True):

        self.simname = simname
        self.hsmall = il.groupcat.loadHeader(path, snap)['HubbleParam']
        self.Lbox = il.groupcat.loadHeader(path, snap)['BoxSize'] # kpc/h
        self.ID = ID

        if   self.simname == 'TNG300-1':       # Msun/h
             self.mdm = 4e7
        elif self.simname == 'TNG100-1':
             self.mdm = 5.1e6
        elif self.simname == 'TNG100-3':
             self.mdm = 3.2e8
        elif self.simname == 'TNG300-1-Dark':
             self.mdm = 7e7
        elif self.simname == 'TNG100-1-Dark':
             self.mdm = 6e6
        elif self.simname == 'TNG100-3-Dark':
             self.mdm = 3.8e8
        else:
             print('SIMNAME NOT RECOGNISED')

        # Group info
        d = il.groupcat.loadSingle(basePath=path, snapNum=snap, haloID=ID)
        self.center = d['GroupPos']
        self.M200   = d['Group_M_Crit200']/self.hsmall*1e10            # Msun
        self.M500   = d['Group_M_Crit500']/self.hsmall*1e10            # Msun
        self.R200   = d['Group_R_Crit200']/self.hsmall                 # kpc
        self.R500   = d['Group_R_Crit500']/self.hsmall                 # kpc
        self.part   = d['GroupLenType']

        # Get Rdelta, Mdelta previoiusly calculated for DARK MATTER
        if load_ext_delta_files_DM:

           files_path = path.replace(self.simname+'/output', '') + 'M1000_M2500/' + self.simname + '/'
           files = sorted(glob.glob(files_path + '*.npy'))

           # If self.ID is not in ID list, these will stay null 
           self.R1000 = 0.0
           self.M1000 = 0.0
           self.R2500 = 0.0
           self.M2500 = 0.0

           for n, f in enumerate(files):
               # Looking for self.ID in npy files and reading relative cvs file
               IDs, M200, R200, M500, R500 = np.load(f)

               if self.ID in IDs:
                  filename = 'IDs_M200_M500_M1000_M2500_'+ str(n) +'.csv'
                  print('Reading '+filename)
                  df = pd.read_csv(files_path + filename)
                  self.R1000 = float(df['R1000'][df['IDs']==self.ID])
                  self.M1000 = float(df['M1000'][df['IDs']==self.ID])
                  self.R2500 = float(df['R2500'][df['IDs']==self.ID])
                  self.M2500 = float(df['M2500'][df['IDs']==self.ID])
                  break


        # Get Rdelta, Mdelta previoiusly calculated for BARYONS
        if load_ext_delta_files_BAR:

           # Where to find calculated files
           files_path = path.replace(self.simname+'/output', '') + 'M_delta_baryons/' + self.simname + '/'

           # npy files with halos IDs
           files = sorted(glob.glob(path.replace(self.simname+'/output', '') + 'M1000_M2500/' + self.simname + '/' + '*.npy'))

           # If self.ID is not in ID list, these will stay null
           self.Mgas200   = 0.0
           self.Mstar200  = 0.0
           self.Mgas500   = 0.0
           self.Mstar500  = 0.0
           self.Mgas1000  = 0.0
           self.Mstar1000 = 0.0
           self.Mgas2500  = 0.0
           self.Mstar2500 = 0.0

           for n, f in enumerate(files):
               # Looking for self.ID in npy files and reading relative cvs file

               IDs, M200, R200, M500, R500 = np.load(f)
               print('np.load(', f, ')')
               print(len(IDs))
               print(IDs)

               if self.ID in IDs:
                  filename = 'IDs_Mgas_Mstar_'+ str(n) +'.csv'
                  print('Reading '+filename)
                  df = pd.read_csv(files_path + filename)
                  print(df.columns)
                  print(df['Mgas200'][df['IDs']==self.ID])
                  print(df['IDs'])
                  print(self.ID)
                  self.Mgas200   = float(df['Mgas200'][df['IDs']==self.ID])
                  self.Mstar200  = float(df['Mstar200'][df['IDs']==self.ID])
                  self.Mgas500   = float(df['Mgas500'][df['IDs']==self.ID])
                  self.Mstar500  = float(df['Mstar500'][df['IDs']==self.ID])
                  self.Mgas1000  = float(df['Mgas1000'][df['IDs']==self.ID])
                  self.Mstar1000 = float(df['Mstar1000'][df['IDs']==self.ID])
                  self.Mgas2500  = float(df['Mgas2500'][df['IDs']==self.ID])
                  self.Mstar2500 = float(df['Mstar2500'][df['IDs']==self.ID])
                  break

        ##
        print('HALO props:')

        print('#part = ', self.part)

        print('M200 [Msun] = %.5e' %self.M200)
        print('R200 [kpc]  = %f'   %self.R200)
        if load_ext_delta_files_BAR:
           print('Mgas  in M200 [Msun] = %.5e' %self.Mgas200)
           print('Mstar in M200 [Msun] = %.5e' %self.Mstar200)

        print('M500 [Msun] = %.5e' %self.M500)
        print('R500 [kpc]  = %f'   %self.R500)
        if load_ext_delta_files_BAR:
           print('Mgas  in M500 [Msun] = %.5e' %self.Mgas500)
           print('Mstar in M500 [Msun] = %.5e' %self.Mstar500)

        print('M1000 [Msun] = %.5e' %self.M1000)
        print('R1000 [kpc]  = %f'   %self.R1000)
        if load_ext_delta_files_BAR:
           print('Mgas  in M1000 [Msun] = %.5e' %self.Mgas1000)
           print('Mstar in M1000 [Msun] = %.5e' %self.Mstar1000)

        print('M2500 [Msun] = %.5e' %self.M2500)
        print('R2500 [kpc]  = %f'   %self.R2500)
        if load_ext_delta_files_BAR:
           print('Mgas  in M2500 [Msun] = %.5e' %self.Mgas2500)
           print('Mstar in M2500 [Msun] = %.5e' %self.Mstar2500)
        ##

        print('Getting halo particles (account for periodic box)')

        # DM
        dmpos              = il.snapshot.loadHalo(path, snap, ID, 'dm',  fields=['Coordinates'])      # kpc/h
        self.dmpos_shifted = myil.utils.get_periodic_coords(self.center, dmpos, self.Lbox)

        # GAS
        if self.part[0]>0:
           gaspos              = il.snapshot.loadHalo(path, snap, ID, 'gas', fields=['Coordinates'])
           self.gaspos_shifted = myil.utils.get_periodic_coords(self.center, gaspos, self.Lbox)
           self.gasmasses      = il.snapshot.loadHalo(path, snap, ID, 'gas', fields=['Masses'])
        else:
           self.gaspos_shifted = np.empty((0,3))
           self.gasmasses      = np.array([])

        # STARS
        if self.part[4]>0:
           starpos              = il.snapshot.loadHalo(path, snap, ID, 'star', fields=['Coordinates'])
           self.starpos_shifted = myil.utils.get_periodic_coords(self.center, starpos, self.Lbox)
           self.starmasses      = il.snapshot.loadHalo(path, snap, ID, 'star', fields=['Masses'])
        else:
           self.starpos_shifted = np.empty((0,3))
           self.starmasses      = np.array([])

        # BH
        if self.part[5]>0:
           BHpos                = il.snapshot.loadHalo(path, snap, ID, 'bh', fields=['Coordinates'])
           self.BHpos_shifted   = myil.utils.get_periodic_coords(self.center, BHpos, self.Lbox)
           self.BHmasses        = il.snapshot.loadHalo(path, snap, ID, 'bh', fields=['Masses'])
           # Count the BHs as stars
           # self.starpos_shifted = np.append(self.starpos_shifted, BHpos_shifted, axis=0)
           # self.starmasses = np.append(self.starmasses, BHmasses)

    @staticmethod
    def fnc(radius, delta_goal, d2, m):
        '''
           Calculates difference between seeked for density and density at a given radius
           (needed by M_delta)
        '''
        # Seeked for density
        rho_goal   = delta_goal * utils_sp.rho_crit  # Msun/h/(kpc/h)**3

        const = 4./3.*np.pi

        # Consider only particles within radius
        infilt = (d2<radius*radius)

        # Returns how far off we are from goal
        return (np.sum(m[infilt]) / (const*radius**3) - delta_goal * utils_sp.rho_crit)


    def M_delta(self, delta_goal, start_radius):
        '''
           Calculates mass within given density contrast for a group given its group-ID.
           ** Assumes hydrodynamical simulation **
           delta_goal: density/critical_density
           start_radius: radius greater than the one seeked
        '''
        # Group DM, GAS, STAR and BH particles
        pos  = np.concatenate((self.dmpos_shifted, self.gaspos_shifted, self.starpos_shifted, self.BHpos_shifted), axis=0)                         # kpc/h
        mass = np.concatenate((np.ones(len(self.dmpos_shifted))*self.mdm, self.gasmasses*1e10, self.starmasses*1e10, self.BHmasses*1e10), axis=0)  # Msun/h

        # Calculate their distance from halo center
        d2  = np.sum(pos**2.0, axis=1)

        # Find radius via bisection method
        r_delta = optimize.bisect(Group.fnc, 1, start_radius, args=(delta_goal, d2, mass))   # kpc/h

        # Find mass
        m_delta = np.sum(mass[d2<r_delta*r_delta])                                           # Msun/h

        return (r_delta/self.hsmall, m_delta/self.hsmall) # (kpc, Msun)


    def baryons_within_R(self, radius, ptype=None):
        '''
           radius: distance from center within all particles of ptype will be summed up [kpc]
           ptype: 'GAS' or 'STARS'
           returns: mass of a given type within radius
        '''
        
        radius = radius * self.hsmall    # kpc/h

        if   ptype == 'GAS':
             pos  = self.gaspos_shifted  # kpc/h
             mass = self.gasmasses*1e10  # Msun/h

        elif ptype == 'STARS':
             pos  = self.starpos_shifted # kpc/h
             mass = self.starmasses*1e10 # Msun/h

        else:
             print('Particle type not recognised')

        # Calculate particle squared distance from halo center
        d2  = np.sum(pos**2.0, axis=1)

        # Sum up mass within radius
        return np.sum(mass[d2<radius*radius]) / self.hsmall



    def halo_img(self, bins=128, axis=[0,1], curve=False, radii=None, vel_curve_hist=None):

        print('Plotting halo images')

        if curve==True:
           nplots = 4
           width_img = 20
        else:
           nplots = 3
           width_img = 15

        plt.clf()
        fig, ax = plt.subplots(1, nplots, figsize=(width_img,5)) #, sharex=True, sharey=True, figsize=(width_img,5))

        labels = ['DM', 'GAS', 'STARS']
        xpos = [self.dmpos_shifted[:,axis[0]], self.gaspos_shifted[:,axis[0]], self.starpos_shifted[:,axis[0]]]
        ypos = [self.dmpos_shifted[:,axis[1]], self.gaspos_shifted[:,axis[1]], self.starpos_shifted[:,axis[1]]]

        for i, axis in enumerate(ax):
            if i==3: # velocity curve
               axis.plot(radii, vel_curve_hist, '.', color='purple')
               axis.axvline(x=self.rmax/self.hsmall, color='lightgreen')
               axis.set_xlabel('distance from center [kpc]')
               axis.set_ylabel('sqrt(GM/r) [km/s]')
            else:
               axis.hist2d(xpos[i], ypos[i], bins=bins, norm=mpl.colors.LogNorm())
               if i==0: # DM
                  xlims = axis.get_xlim()
                  ylims = axis.get_ylim()
               else:
                  if i==2: # STAR
                     axis.text(xlims[0]+(xlims[1]-xlims[0])/40., ylims[1]-(ylims[1]-ylims[0])/40., 'halo #%i' % self.ID, ha='left', va='top', color='m', fontsize=15)
                     #axis.text(xlims[0]+(xlims[1]-xlims[0])/40., ylims[1]-(ylims[1]-ylims[0])/12., 'Msub = %.2e Msun' % (self.Msub/self.hsmall), ha='left', va='top', color='m', fontsize=15)
                     #axis.text(xlims[0]+(xlims[1]-xlims[0])/40., ylims[1]-(ylims[1]-ylims[0])/7.,  'Mhalf = %.2e Msun' % self.Mhalfr, ha='left', va='top', color='m', fontsize=15)
                     #axis.text(xlims[0]+(xlims[1]-xlims[0])/40., ylims[1]-(ylims[1]-ylims[0])/5.,  'Mmax = %.2e Msun' % self.Mrmax, ha='left', va='top', color='m', fontsize=15)
                     #axis.text(xlims[0]+(xlims[1]-xlims[0])/40., ylims[1]-(ylims[1]-ylims[0])/3.9, 'Rhalf = %.2d kpc' % (self.halfr/self.hsmall), ha='left', va='top', color='m', fontsize=15)
                     #axis.text(xlims[0]+(xlims[1]-xlims[0])/40., ylims[1]-(ylims[1]-ylims[0])/3.2, 'Rmax = %.2d kpc' % (self.rmax/self.hsmall), ha='left', va='top', color='m', fontsize=15)
                  axis.set_xlim(xlims)
                  axis.set_ylim(ylims)
               axis.text(xlims[1]-(xlims[1]-xlims[0])/40., ylims[0]+(ylims[1]-ylims[0])/40., labels[i], horizontalalignment='right', color='k', fontsize=15)
        plt.tight_layout()
        plt.savefig('image_ID'+str(self.ID)+'_sim_'+self.simname+'.pdf')
                


#ID = 4000

#SIMNAME = 'TNG300-1'
#basePath = '/efiler1/cpenzo/IllustrisTNG/'+SIMNAME+'/dm_only'
#snapNum = 99

#start = time.time()
#print('\nhalo = ', ID)
#halo = Group(basePath, snapNum, SIMNAME, ID)
#print('Lasted %d secs' % (time.time()-start))
