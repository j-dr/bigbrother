from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
from .healpix_utils import PixMetric
from astropy.cosmology import FlatLambdaCDM
import units
import numpy as np
import healpy as hp
import fitsio
import time

class BaseCatalog:
    """
    Base class for catalog type
    """

    _valid_reader_types = ['fits', 'rockstar', 'ascii']

    def __init__(self, ministry, filestruct, fieldmap=None,
                 unitmap=None, filenside=None, groupnside=None,
                 nest=True, maskfile=None, filters=None, goodpix=None,
                 reader=None, area=None):

        self.ministry = ministry
        self.filestruct = filestruct
        self.fieldmap = fieldmap
        self.unitmap  = unitmap
        self.filters = filters
        self.parseFileStruct(filestruct)
        self.maskfile = maskfile
        self.mask = None
        if area is None:
            self.area = 0.0
        else:
            self.area = area

        self.filenside = filenside
        self.nest = nest

        if groupnside is None:
            self.groupnside = 4

        if goodpix is None:
            self.goodpix = 1
        else:
            self.goodpix = goodpix

        self.necessaries = []
        self.filters = []

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

    def getFilePixels(self, nside):
        """
        Get the healpix cells occupied by galaxies
        in each file. Assumes files have already been
        sorted correctly by parseFileStruct
        """
        fpix = []

        #BCC catalogs have pixels in filenames
        if ('BCC' in self.__class__.__name__) &
          (self.filenside is not None) & (self.filenside<=self.groupnside):
            fk = self.filestruct.keys()

            for f in self.filestruct[fk[0]]:
                p = int(f.split('.')[-1])

                if (self.filenside == self.groupnside):
                    fpix.append([p])
                else:
                    if not self.nest:
                        p = hp.ring2nest(self.filenside, p)

                    base = p >> self.filenside
                    hosubpix = pix & ( ( 1 << ( self.filenside ) ) - 1 );
                    losubpix = hosubpix / ( 1 << ( self.filenside - self.groupnside) ) );
                    p  = base * ( 1 << ( 2 * self.groupnside ) ) + losubpix;

                    fpix.append([p])

        else:
            pmetric = PixMetric(self.ministry, self.groupnside)
            mg = self.ministry.genMetricGroups([pmetric])
            ms = mg[0][1]
            fm = mg[0][0]

            for mappable in self.ministry.genMappable(fm):
                mapunit = self.ministry.readMappable(mappable, fm)
                mapunit = self.ministry.treeToDict(mapunit)
                mapunit = self.convert(mapunit, ms)
                fpix.append(pmetric(mapunit))

        return fpix

    def groupFiles(self):

        fpix = self.getfilePixels(self.groupnside)


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

    def readMappable(self, mappable, fieldmap):
        """
        Default reader is FITS reader
        """

        if self.reader=='fits':
            return self.readFITSMappable(mappable, fieldmap)

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

                elif self.unitmap[key]==m.unitmap[key]:
                    continue

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

class PlaceHolder(BaseCatalog):

    def __init__(self, ministry, filestruct, fieldmap=None,
                 nside=None, zbins=None, maskfile=None,
                 filters=None, unitmap=None, goodpix=None,
                 reader=None):

        self.ctype = 'placeholder'
        BaseCatalog.__init__(self, ministry, filestruct,
                                fieldmap=fieldmap, nside=nside,
                                maskfile=maskfile, filters=filters,
                                unitmap=unitmap, reader=reader)
