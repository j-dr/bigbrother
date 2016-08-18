from __future__ import print_function, division
import numpy as np


class Jackknife:

    def __init__(self, jtype='healpix', nside=8, nneighbors=1, radius=None,
                nregions=None):

        self.jtype      = jtype
        self.nside      = nside
        self.nneighbors = nneighbors
        self.radius     = radius
        self.nregions   = nregions


    def healpixJackknifeRegions(self):
        pass

    def subboxJackknifeRegions(self):

        pass

    def combineJackknife(self):
        pass
