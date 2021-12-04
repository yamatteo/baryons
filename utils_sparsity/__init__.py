import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path
import h5py
import glob
from scipy import optimize
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
#-----------------------------------------------------------------------------------------------------------------------------------
rho_crit   = 2.77519737e11 * 1e-9 #  (Msun/h)/(kpc/h)**3
const = 4./3.*np.pi
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def sparsity_vs_xaxis(sp, xaxis, bins=[], xaxis_label=None, color_sparsity=None, pdf_filename=None, yaxis_label=r'$\sigma^2_{\Delta200,\Delta500}$', nbins=16, log_xaxis=None):
    """ calculate binned sparsity and eventually build plots """

    if len(bins)==0:
       # calculate bins
       if log_xaxis:
          bin_step = (np.log10(np.max(xaxis)) - np.log10(np.min(xaxis)))/nbins
          x_bins = np.arange(np.log10(np.min(xaxis)), np.log10(np.max(xaxis)), bin_step)
          yhist, xhist = np.histogram(np.log10(xaxis), bins=x_bins)
       else:
          bin_step = (np.max(xaxis) - np.min(xaxis))/nbins
          x_bins = np.arange(np.min(xaxis), np.max(xaxis), bin_step)
          yhist, xhist = np.histogram(xaxis, bins=x_bins)

    else:
       xhist = bins

    plot_bins = (xhist[1:]+xhist[:-1])/2.0

    # binning
    nelements = []
    mean_spar = []
    std_spar  = []
    for i in range(len(plot_bins)):
        filt = (xaxis>10**xhist[i])&(xaxis<=10**xhist[i+1])
        spar = sp[filt]

        nelements.append(len(spar))
        mean_spar.append(np.mean(spar))
        std_spar.append(np.std(spar))

    if (pdf_filename):
    # plot

       fig1, ax1 = plt.subplots(ncols=1, nrows=2, sharex=False)

       if log_xaxis:
          ax1[0].scatter(10**plot_bins, mean_spar, color=color_sparsity)
          ax1[1].scatter(10**plot_bins, np.asarray(std_spar)**2, color=color_sparsity)
          ax1[0].set_xscale('log')
          ax1[1].set_xscale('log')
       else:
          ax1[0].scatter(plot_bins, mean_spar, color=color_sparsity)
          ax1[1].scatter(plot_bins, np.asarray(std_spar)**2, color=color_sparsity)

       ax1[0].set_ylabel(r's$_{\Delta200,\Delta500}$',   fontsize=18)
       #ax1[0].set_ylim([np.min(), np.max()])

       ax1[1].set_xlabel(xaxis_label, fontsize=18)
       ax1[1].set_ylabel(r'$\sigma^2_{\Delta200,\Delta500}$',   fontsize=18)

       #ax1[0].legend(loc=2)
       fig1.savefig(pdf_filename)

    return np.asarray(xhist), np.asarray(mean_spar), np.asarray(std_spar), np.asarray(nelements)
#-----------------------------------------------------------------------------------------------------------------------------------
def save_hist(x, xlabel, units='', nbins=50, log=True, prob_dens=False, kde=False):
    ''' Calculates histogram and saves it to pdf'''
    plt.clf()
    fig, ax = plt.subplots(ncols=1, nrows=1, sharex=False)
    print('\nPlotting %s' % xlabel)

    if log:
       print('xLog plot')
       if min(x)==0:
          x+=1
          print('There are zeros in your array! Added one to make a log plot')
       bin_step = (np.log10(np.max(x)) - np.log10(np.min(x)))/nbins
       xbins = np.arange(np.log10(np.min(x)), np.log10(np.max(x)), bin_step)
       n, bins, patches = plt.hist(x, bins=10**xbins, density=prob_dens, label='normed hist', alpha=0.5)
       bottom, top = plt.ylim()
       plt.vlines(min(x), bottom, top/10., linewidth=3, color='y')
       plt.vlines(max(x), bottom, top/10., linewidth=3, color='y', label='min/max element')
       plt.xscale('log')

    else:
       bin_step = (np.max(x) - np.min(x))/nbins
       xbins = np.arange(np.min(x), np.max(x), bin_step)
       n, bins, patches = plt.hist(x, bins=xbins, density=prob_dens, label='normed hist', alpha=0.5)
       bottom, top = plt.ylim()
       plt.vlines(min(x), bottom, top/10., linewidth=3, color='y')
       plt.vlines(max(x), bottom, top/10., linewidth=3, color='y', label='min/max element')

       if kde:
          # Gaussian Kernel Estimation
          data = np.asarray(x).reshape(-1, 1)
          kde = KernelDensity(kernel='gaussian', bandwidth=bin_step).fit(data)
          xkde = np.arange(np.min(bins),np.max(bins), bin_step/3)
          ykde = kde.score_samples(xkde.reshape(-1, 1))
          plt.plot(xkde, np.exp(ykde), color='magenta', label='gaussian KDE')
          # Find minima
          minima = argrelextrema(ykde, np.less)[0]
          print('Minima: ', minima)
          plt.scatter(xkde[minima], np.exp(ykde[minima]), edgecolors='k', color='white', label='relative mins')
          props = dict(boxstyle='round', facecolor='white', linewidth=0.1)
          for m in minima:
              if np.exp(ykde[m]) != 0 :
                 plt.text(xkde[m], np.exp(ykde[m]), str(round(xkde[m],4))+','+str(round(np.exp(ykde[m]),4)), fontsize=5, bbox=props)

       plt.legend()

    plt.xlabel(xlabel+units)
    plt.savefig(xlabel+'.pdf')

    return (n, bins, patches)
#-----------------------------------------------------------------------------------------------------------------------------------


def print_halo_info(halo, group_dict):
    print()
    print('halo #', halo)
    print('# dm,gas,stars particles =', group_dict['GroupLenType'][halo][1],group_dict['GroupLenType'][halo][0],group_dict['GroupLenType'][halo][4])
    print('Group M 200 [Msun/h] =', group_dict['Group_M_Crit200'][halo]*1e10)
    print('Group M 500 [Msun/h] =', group_dict['Group_M_Crit500'][halo]*1e10)
    print('Group R 200 [kpc/h]  =', group_dict['Group_R_Crit200'][halo])
    print('Group R 500 [kpc/h]  =', group_dict['Group_R_Crit500'][halo])
    print( 'Group Pos   [kpc/h]  =', group_dict['GroupPos'][halo])
    print()
#-----------------------------------------------------------------------------------------------------------------------------------

def get_squared_distances(dist, Lbox):
    """ OK
        Check particle distances and corrects them according to bondary conditions given Lbox
        Note that Illustris has [0,Lbox] coordinates
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
             dist[:,0] = map(lambda dx: dx+Lbox if dx<-Lbox_half else dx, dist[:,0]) # map(lambda dx: dx-Lbox_half if dx>Lbox_half else dx, dist[:,0])
          if len(dist[:,1][periodic_filt_1[:,1]])!=0:
             dist[:,1] = map(lambda dx: dx+Lbox if dx<-Lbox_half else dx, dist[:,1]) # map(lambda dx: dx-Lbox_half if dx>Lbox_half else dx, dist[:,1])
          if len(dist[:,2][periodic_filt_1[:,2]])!=0:
             dist[:,2] = map(lambda dx: dx+Lbox if dx<-Lbox_half else dx, dist[:,2]) # map(lambda dx: dx-Lbox_half if dx>Lbox_half else dx, dist[:,2])
       if len_filt_2!=0:
          if len(dist[:,0][periodic_filt_2[:,0]])!=0:
             dist[:,0] = map(lambda dx: dx-Lbox if dx>Lbox_half else dx,  dist[:,0]) # map(lambda dx: dx+Lbox_half if dx<=Lbox_half else dx, dist[:,0])
          if len(dist[:,1][periodic_filt_2[:,1]])!=0:
             dist[:,1] = map(lambda dx: dx-Lbox if dx>Lbox_half else dx,  dist[:,1]) # map(lambda dx: dx+Lbox_half if dx<=Lbox_half else dx, dist[:,1])
          if len(dist[:,2][periodic_filt_2[:,2]])!=0:
             dist[:,2] = map(lambda dx: dx-Lbox if dx>Lbox_half else dx,  dist[:,2]) # map(lambda dx: dx+Lbox_half if dx<=Lbox_half else dx, dist[:,2])
       dist2 = np.sum(dist**2.0,axis=1)

    return dist2 # kpc/h
#-----------------------------------------------------------------------------------------------------------------------------------

def get_particles(halo, min_part, max_part, ptype, simPath, nsnap):

    ''' OK

    halo: index of group in FOF file
    min_part: index of first particle of halo
    max_part: index of last particle of halo
    ptype: DM, GAS or STAR
    simPath: where the Illustris snapshot files are
    nsnap: number of snapshot (135 -> z=0)

    Returns positions and, if not DM, masses of particles in the FOF group
    (DM particles all have same mass)
    '''

    # CHANGE IT BACK TO len, TEMPORARY WHILE IT RE-DOWNLOADS LAST FILE
    n_hdf5_files = len(glob.glob(simPath+'.*.hdf5'))

    if ptype=='DM':
       field = 'PartType1'
    elif ptype=='GAS':
       field = 'PartType0'
    elif ptype=='STAR':
       field = 'PartType4'
    elif ptype=='BH':
       field = 'PartType5'
    else:
       print('Particle type not recognised!')

    # Looks for list of indecis for fist and last particle of each FOF group

    partfname = 'file_n'+ptype+'part_'+str(n_hdf5_files)+'.npy'

    if os.path.isfile(partfname): # if list already exist, read it
       print('Reading file ', partfname)
       file_ndmpart_cum = np.cumsum(np.load(partfname))

    else:                         # else, create it
       print('Building list of file particle lenghts')
       file_ndmpart = []
       for i in range(len(glob.glob(simPath+'.*.hdf5'))):
           fname = simPath+'snapdir_'+str("%03d" % nsnap)+'/snap_'+str(nsnap)+'.'+str(i)+'.hdf5'
           f = h5py.File(fname)
           print(fname, len(f[field]['ParticleIDs'].value))
           file_ndmpart.append(len(f[field]['ParticleIDs'].value))
       np.save(partfname, file_ndmpart)
       file_ndmpart_cum = np.cumsum(file_ndmpart)

    first_file = min([ii for ii,b in enumerate(min_part<file_ndmpart_cum) if b])
    last_file  = min([ii for ii,b in enumerate(max_part<file_ndmpart_cum) if b])

    # Get data from snapshot
    print('Looking for particles in snapshots')

    for i in range(first_file, last_file+1):

        f = h5py.File(simPath+'snap_'+str(nsnap)+'.'+str(i)+'.hdf5', 'r')

        index_first_particle_for_file = int(max(min_part, file_ndmpart_cum[i-1]) - file_ndmpart_cum[i-1])  # when subtracting for whatever reason it becomes a float
        index_last_particle_for_file  = int(min(max_part, file_ndmpart_cum[i]-1) - file_ndmpart_cum[i-1]+1)

        pcoords = f[field]['Coordinates'].value[index_first_particle_for_file:index_last_particle_for_file]

        if ptype!='DM':
           pmasses = f[field]['Masses'].value[index_first_particle_for_file:index_last_particle_for_file]

        if i==first_file:
           pos = pcoords # kpc/h
           if ptype!='DM':
              mas = pmasses # Msum/h
        else:
           pos = np.vstack([pos, pcoords])
           if ptype!='DM':
              mas = np.hstack([mas, pmasses])

    print('# particles from summing all particles found', np.shape(pos))
    print()

    if ptype!='DM':
       return pos, mas
    else:
       return pos

#-----------------------------------------------------------------------------------------------------------------------------------
def fnc(radius, dist2, delta_goal, pmasses):
    infilt = (dist2<=radius*radius)
    return np.sum(pmasses[infilt]) / (const*radius**3) - delta_goal * rho_crit

def fnc_DM(radius, dist2, delta_goal, mdm):
    infilt = (dist2<=radius*radius)
    return np.sum(infilt)*mdm / (const*radius**3) - delta_goal * rho_crit

#-----------------------------------------------------------------------------------------------------------------------------------

def find_radius(ptype, dist2, delta_goal, rini, mdm=0, pmasses=0):
    '''
       Uses bisection to find radius at which density = delta*crit_density
       mdm : DM particle mass in case of DM-only sim
       pmasses : particle masses in case of HYDRO sim
       Note: f(1) and f(rini) must have different signs
       no check on increasing rini because rini is taken from Illustris halo finder
       (eg if looking for R1000 we start from R500)
    '''
    r = rini

    if ptype=='DM':
       while fnc_DM(r, dist2, delta_goal, mdm) > 0:
            r += r/3.
    else:
       while fnc(r, dist2, delta_goal, pmasses) > 0:
            r += r/3.

    if ptype=='DM':
       return optimize.bisect(fnc_DM, 1, r, args=(dist2, delta_goal, mdm))
    else:
       return optimize.bisect(fnc, 1, r, args=(dist2, delta_goal, pmasses))
#-----------------------------------------------------------------------------------------------------------------------------------