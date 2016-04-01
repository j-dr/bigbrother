from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
from astropy.cosmology import FlatLambdaCDM
from .galaxy import GalaxyCatalog, BCCCatalog, S82PhotCatalog, S82SpecCatalog, DESGoldCatalog
from .halo import HaloCatalog, BCCHaloCatalog
import numpy as np
import healpy as hp
import helpers
import fitsio
import time

TZERO = None
def tprint(info):
    global TZERO
    if TZERO is None:
        TZERO = time.time()

    print('[%8ds] %s' % (time.time()-TZERO,info))


class Ministry:
    """
    A class which owns all the other catalog data 
    """
    
    def __init__(self, omega_m, omega_l, h, minz, maxz, area=0.0,
                 boxsize=None):
        """
        Initialize a ministry object
        
        Arguments
        ---------
        omega_m : float
            Matter density parameter now
        omega_l : float
            Lambda density parameter now
        h : float
            Dimensionless hubble constant
        minz : float
            Minimum redshift
        maxz : float
            Maximum redshift
        area : float, optional
            The area spanned by all catalogs held
        """

        self.omega_m = omega_m
        self.omega_l = omega_l
        self.h = h
        self.cosmo = FlatLambdaCDM(H0=100*h, Om0=omega_m)
        self.minz = minz
        self.maxz = maxz
        if minz!=maxz:
            self.lightcone = True
        else:
            self.lightcone = False
            self.boxsize = boxsize

        self.area = area
        self.volume = self.calculate_volume(area,self.minz,self.maxz)


    def calculate_volume(self,area,minz,maxz):
        if self.lightcone:
            rmin = self.cosmo.comoving_distance(minz)*self.h
            rmax = self.cosmo.comoving_distance(maxz)*self.h
            return (area/41253)*(4/3*np.pi)*(rmax**3-rmin**3)
        else:
            return (self.boxsize*self.h)**3
        
        
    def setGalaxyCatalog(self, catalog_type, filestruct, fieldmap=None,
                         zbins=None, maskfile=None,
                         goodpix=1):
        """
        Fill in the galaxy catalog information
        """

        if catalog_type == "BCC":
            self.galaxycatalog = BCCCatalog(self, filestruct,  zbins=zbins, 
                                            fieldmap=fieldmap, maskfile=maskfile,
                                            goodpix=goodpix)
        elif catalog_type == "S82Phot":
            self.galaxycatalog = S82PhotCatalog(self, None)
        elif catalog_type == "S82Spec":
            self.galaxycatalog = S82SpecCatalog(self, None)
        elif catalog_type == "DESGold":
            self.galaxycatalog = DESGoldCatalog(self, filestruct, maskfile=maskfile,                                                goodpix=goodpix)

    def setHaloCatalog(self, catalog_type, filestruct):
        """
        Fill in the halo catalog information
        """

        if catalog_type == "BCC":
            self.halocatalog = BCCHaloCatalog(self, filestruct, zbins=zbins, 
                                              fieldmap=fieldmap, maskfile=maskfile, 
                                              goodpix=goodpix)


    def validate(self, metrics=None, verbose=False):
        """
        Run all validation metrics by iterating over only the files we
        need at a given time, mapping catalogs to relevant statistics
        which are reduced at the end of the iteration into observables 
        that we care about
        """

        #For now, just focus on galaxy observables
        #catalogs that need to be in memory at a particular
        #moment get more complicated when calculating galaxy-halo 
        #relation statistics
        #self.galaxycatalog.configureMetrics(metrics)
        mappables = self.galaxycatalog.genMappable(metrics)

        #probably want Ministry to have map method
        #which knows how to combine different 
        #types of catalogs
        for f in mappables:
            if verbose:
                tprint('    {0}'.format(f))
            self.galaxycatalog.map(f)

        self.galaxycatalog.reduce()
