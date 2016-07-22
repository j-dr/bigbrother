from __future__ import print_function, division
import numpy as np

def ra2rad(mapunit, mapkey):
    phi = np.deg2rad(mapunit[mapkey])

    return phi

def dec2rad(mapunit, mapkey):
    theta = -mapunit[mapkey]+90.
    theta = np.deg2rad(theta)

    return theta

def flux2mag(mapunit, mapkey, zp=30.0):
    return zp - 2.5*np.log10(mapunit[mapkey])

def bccmag2mag(mapunit, mapkey, Q=3.16):
    """
    Note, any catalog using this conversion should define
    redshift as a necessary map key
    """
    return mapunit[mapkey] - Q * (1 / 1 + mapunit['redshift'] - 1 / 1.1)

def fabermag2mag(mapunit, mapkey, Q=0.866):
    """
    Note, any catalog using this conversion should define
    redshift as a necessary map key
    """
    return mapunit[mapkey] - Q * (np.log10(mapunit['redshift']) + 1)
