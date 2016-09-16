from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
from .healpix_utils import PixMetric
from astropy.cosmology import FlatLambdaCDM
import units
import numpy as np
import healpy as hp
import fitsio
import time
import sys

class BaseCatalog:
    """
    Base class for catalog type
    """

    _valid_reader_types = ['fits', 'rockstar', 'ascii']

    def __init__(self, ministry, filestruct, fieldmap=None,
                 unitmap=None,  filters=None, goodpix=None,
                 reader=None, area=None, jtype=None, nbox=None,
                 filenside=None, groupnside=None, nest=True,
                 maskfile=None):

        self.ministry = ministry
        self.filestruct = filestruct
        self.fieldmap = fieldmap
        self.unitmap  = unitmap
        if filters is not None:
            self.filters = filters
        else:
            self.filters = []
        self.parseFileStruct(filestruct)
        self.maskfile = maskfile
        self.mask = None

        if area is None:
            self.area = 0.0
        else:
            self.area = area

        #jackknife information
        self.jtype = jtype

        #for healpix type jackknifing
        self.filenside = filenside
        self.nest = nest

        if groupnside is None:
            self.groupnside = 4
        else:
            self.groupnside = groupnside

        #for subbox type jackknifing
        self.nbox = nbox

        #for mask type jackknifing
        if goodpix is None:
            self.goodpix = 1
        else:
            self.goodpix = goodpix

        self.necessaries = []

        if reader in BaseCatalog._valid_reader_types:
            self.reader = reader
        elif reader is None:
            self.reader = 'fits'
        else:
            raise(ValueError("Invalid reader type {0} specified".format(reader)))

    @abstractmethod
    def parseFileStruct(self, filestruct):
        """
        Given a filestruct object, create map from parameters
        we require to filepaths for easy access
        """

    @abstractmethod
    def getFilePixels(self,nside):
        """
        Get the healpix cells occupied by galaxies
        in each file. Assumes files have already been
        sorted correctly by parseFileStruct
        """
        
    def groupFiles(self):
        """
        Group files together spatially. Healpix grouping implemented,
        subbox grouping still needs to be done.
        """

        fpix = self.getFilePixels(self.groupnside)
        upix = np.unique(np.array([p for sublist in fpix for p in sublist]))
        fgrps = []

        for p in upix:
            fgrps.append([i for i in range(len(fpix)) if p in fpix[i]])

        print('upix: {0}'.format(upix))
        print('fgrps: {0}'.format(fgrps))

        return upix, fgrps

    def readFITSMappable(self, mappable, fieldmap):
        """
        For the file held in mappable, read in the fields defined in
        fieldmap. File must be in FITS format.
        """

        mapunit = {}
        ft      = mappable.dtype
        fname   = mappable.name

        for f in fieldmap.keys():
            fields = []
            for val in fieldmap[ft].values():
                if hasattr(val, '__iter__'):
                    fields.extend(val)
                else:
                    fields.extend([val])

        data = fitsio.read(fname, columns=fields)
        for mapkey in fieldmap[ft].keys():
            mapunit[mapkey] = data[fieldmap[ft][mapkey]]
            if hasattr(fieldmap[ft][mapkey], '__iter__'):
                dt = mapunit[mapkey].dtype[0]
                ne = len(mapunit[mapkey])
                nf = len(fieldmap[ft][mapkey])
                mapunit[mapkey] = mapunit[mapkey].view(dt).reshape((ne,nf))

        return mapunit

    def maskMappable(self, mapunit, mappable):

        tp = np.zeros((len(mapunit[mapunit.keys()[0]]),2))

        if mappable.jtype == 'healpix':
            print('Masking {0} using healpix {1}'.format(mappable.name, mappable.grp))
            for i, key in enumerate(['azim_ang', 'polar_ang']):
                try:
                    conversion = getattr(self, '{0}2{1}'.format(self.unitmap[key],'rad'))
                except:
                    conversion = getattr(units, '{0}2{1}'.format(self.unitmap[key],'rad'))

                tp[:,i] = conversion(mapunit, key)

            pix = hp.ang2pix(self.groupnside, tp[:,1], tp[:,0], nest=self.nest)
            pidx = pix==mappable.grp

            mu = {}
            for k in mapunit.keys():
                mu[k] = mapunit[k][pidx]

            mapunit = mu
            return mapunit

        elif mappable.jtype is None:
            return mapunit
        else:
            raise NotImplementedError

    def readMappable(self, mappable, fieldmap):
        """
        Default reader is FITS reader
        """

        if self.reader=='fits':
            mapunit = self.readFITSMappable(mappable, fieldmap)

        return self.maskMappable(mapunit, mappable)


    def setFieldMap(self, fieldmap):
        self.fieldmap = fieldmap

    def convert(self, mapunit, metrics):
        """
        Convert a map unit from the units given in the catalog
        to those required in metrics
        """
        beenconverted = []
        for m in metrics:
            if self.ctype not in m.catalog_type:
                continue
            for key in m.unitmap:

                if key in beenconverted: continue
                if key not in mapunit.keys(): continue

                try:
                    if self.unitmap[key]==m.unitmap[key]:
                        continue
                except KeyError as e:
                    print(e)
                    print("Catalog unitmap: {0}".format(self.unitmap))
                    print("Metric unitmap: {0}".format(m.unitmap))                    

                try:
                    conversion = getattr(self, '{0}2{1}'.format(self.unitmap[key],m.unitmap[key]))
                except:
                    conversion = getattr(units, '{0}2{1}'.format(self.unitmap[key],m.unitmap[key]))

                mapunit[key] = conversion(mapunit, key)
                beenconverted.append(key)

        return mapunit

    def filter(self, mapunit, fieldmap):

        idx = None

        for i, key in enumerate(self.filters):
            filt = getattr(self, 'filter{0}'.format(key))

            if key.lower() not in mapunit.keys():
                continue

            if i==0:
                idx = filt(mapunit)
            else:
                idxi = filt(mapunit)
                idx = idx&idxi

        if idx is not None:
            for key in mapunit.keys():
                if key in fieldmap.keys():
                    mapunit[key] = mapunit[key][idx]

        return mapunit

    def getArea(self, jackknife=False):

        arm = np.array([True if m.__class__.__name__=="Area" else False
                  for m in self.ministry.metrics])
        am = any(arm)

        if am:
            idx, = np.where(arm==True)[0]

        if not jackknife:
            if not am:
                return self.ministry.area
            else:
                return self.ministry.metrics[idx].area
        else:
            if not am:
                return self.ministry.area
            else:
                return self.ministry.metrics[idx].jarea


class PlaceHolder(BaseCatalog):

    def __init__(self, ministry, filestruct, **kwargs):

        self.ctype = 'placeholder'
        BaseCatalog.__init__(self, ministry, filestruct, **kwargs)

