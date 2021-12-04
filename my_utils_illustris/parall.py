import multiprocessing as mp
import pandas as pd
import my_utils_illustris as myil

def calc_per_halo(ID, path, simname, snap=99):
    '''
       Calculates M1000 and M2500 given groupID
       Returns list with ID, M1000, R1000, M2500, R2500
       (made for pool parallel tool)
    '''
    row = []

    row.append(ID)

    halo = myil.classGroup.Group(path, snap, simname, ID)

    try:
        r1000, m1000 = halo.M_delta(delta_goal=1000, start_radius=halo.R500)
        r2500, m2500 = halo.M_delta(delta_goal=2500, start_radius=r1000)    

    except Exception:
        r1000, m1000, r2500, m2500 = (0, 0, 0, 0)

    row.append(m1000)
    row.append(r1000)

    row.append(m2500)
    row.append(r2500)

    return row


