from __future__ import print_function, division
import numpy as np

def ra2rad(phi):
    phi = np.deg2rad(phi)

    return phi

def dec2rad(theta):
    theta = -theta+90.
    theta = np.deg2rad(theta)

    return theta


def flux2mag(flux, zp=30.0):
    return zp - 2.5*np.log10(flux)

    
    
