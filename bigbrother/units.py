
import numpy as np
import healpy as hp

def ra2rad(mapunit, mapkey):
    phi = np.deg2rad(mapunit[mapkey])
    return phi

def pid2binary(mapunit, mapkey):
    return (mapunit[mapkey]==-1).astype(np.int)

def dec2rad(mapunit, mapkey):
    theta = -mapunit[mapkey]+90.
    theta = np.deg2rad(theta)

    return theta

def ra2mra(mapunit, mapkey):
    return 360 - mapunit[mapkey]

def mra2ra(mapunit, mapkey):
    return 360 - mapunit[mapkey]

def flux2mag(mapunit, mapkey, zp=30.0):
    print('zp : {0}'.format(zp))
    return zp - 2.5*np.log10(mapunit[mapkey])

def magh2mag(mapunit, mapkey, h=0.7):
    return mapunit[mapkey] + 5.0 * np.log10(h)

def mag2magh(mapunit, mapkey, h=0.7):
    return mapunit[mapkey] - 5.0 * np.log10(h)

def fluxerr2magerr(mapunit, mapkey):
    return (2.5/np.log(10.)) * mapunit[mapkey] / mapunit['appmag']

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

def mag2luminosity(mapunit, mapkey, msun=4.679):
    return 10 ** ((mapunit[mapkey] - msun) / -2.5)
    

def mpchra2ra(mapunit, mapkey):
    theta, phi = hp.vec2ang(mapunit[mapkey])

    return phi * 180. / np.pi

def mpchra2rad(mapunit, mapkey):
    
    theta, phi = hp.vec2ang(mapunit[mapkey])

    return phi 

def mpchdec2dec(mapunit, mapkey):
    theta, phi = hp.vec2ang(mapunit[mapkey])

    return -theta  * 180. / np.pi + 90.

def mpchdec2rad(mapunit, mapkey):
    theta, phi = hp.vec2ang(mapunit[mapkey])

    return theta
