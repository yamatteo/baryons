import multiprocessing as mp
import pandas as pd
import my_utils_illustris as myil

def calc_per_halo(ID, path, simname, snap=99):
    '''
       Calculates gas and stellar mass given radius per groupID
       Returns list with ID, Mgas200, Mstar200, Mgas500, Mstar500, Mgas1000, Mstar1000, Mgas2500, Mstar2500
       (made for pool parallel tool)
    '''
    row = []

    row.append(ID)

    halo = myil.classGroup.Group(path, snap, simname, ID)

    try:
        Mgas200  = halo.baryons_within_R(radius=halo.R200, ptype='GAS')
        Mstar200 = halo.baryons_within_R(radius=halo.R200, ptype='STARS')

        Mgas500  = halo.baryons_within_R(radius=halo.R500, ptype='GAS')
        Mstar500 = halo.baryons_within_R(radius=halo.R500, ptype='STARS')

        Mgas1000  = halo.baryons_within_R(radius=halo.R1000, ptype='GAS')
        Mstar1000 = halo.baryons_within_R(radius=halo.R1000, ptype='STARS')

        Mgas2500  = halo.baryons_within_R(radius=halo.R2500, ptype='GAS')
        Mstar2500 = halo.baryons_within_R(radius=halo.R2500, ptype='STARS')

    except Exception:
        Mgas200, Mstar200, Mgas500, Mstar500, Mgas1000, Mstar1000, Mgas2500, Mstar2500 = (0, 0, 0, 0, 0, 0, 0, 0)

    row.append(Mgas200)
    row.append(Mstar200)

    row.append(Mgas500)
    row.append(Mstar500)

    row.append(Mgas1000)
    row.append(Mstar1000)

    row.append(Mgas2500)
    row.append(Mstar2500)

    return row


