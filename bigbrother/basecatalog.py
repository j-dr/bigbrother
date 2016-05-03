from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
from .helpers import PixMetric
from astropy.cosmology import FlatLambdaCDM
import units
import numpy as np
import healpy as hp
import helpers
import fitsio
import time

class BaseCatalog:
    """
    Base class for catalog type
    """
    
    def __init__(self, ministry, filestruct, nside=8, maskfile=None, goodpix=1):
        self.ministry = ministry
        self.filestruct = filestruct
        self.parseFileStruct(filestruct)
        self.maskfile = maskfile
        self.mask = None
        self.area = 0.0
        self.nside = nside
        self.goodpix = goodpix

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
        
        pmetric = PixMetric(self.ministry, nside)
        mg = self.ministry.genMetricGroups([pmetric])
        ms = mg[0][1]
        fm = mg[0][0]

        for mappable in self.ministry.genMappable(fm):
            mapunit = self.ministry.readMappable(mappable, fm)
            mapunit = self.ministry.treeToDict(mapunit)
            mapunit = self.convert(mapunit, ms)
            fpix.append(pmetric(mapunit))

        return fpix
            
    def genMappable(self, metrics):
        """
        Given a set of metrics, generate a list of mappables
        which can be fed into map functions
        """
        mappables = []
        mapkeys = []
        fieldmap = {}
        usedfiletypes = []

        if metrics!=None:
            self.metrics = metrics

        if hasattr(self, 'necessaries'):
            mapkeys.extend(self.necessaries)

        for m in self.metrics:
            mapkeys.extend(m.mapkeys)

        mapkeys = np.unique(mapkeys)

        #for each type of data necessary for 
        #the metrics we want to calculate,
        #determine the file type it's located
        #in and the field 
        for mapkey in mapkeys:
            try:
                fileinfo = self.fieldmap[mapkey]
            except KeyError as e:
                print('No key {0}, continuing...'.format(e))
                continue

            for field in fileinfo.keys():
                valid = False
                filetypes = fileinfo[field]
                for ft in filetypes:
                    if ft in self.filetypes:
                        #if already have one field
                        #make a list of fields
                        if ft not in fieldmap.keys():
                            fieldmap[ft] = {}
                        if mapkey in fieldmap[ft].keys():
                            if hasattr(fieldmap[ft][mapkey],'__iter__'):
                                fieldmap[ft][mapkey].append(field)
                            else:
                                fieldmap[ft][mapkey] = [fieldmap[ft][mapkey],field]
                        else:
                            fieldmap[ft][mapkey] = field

                        valid = True
                        if ft not in usedfiletypes:
                            usedfiletypes.append(ft)
                        break

                if not valid:
                    raise Exception("Filetypes {0} for mapkey {1} are not available!".format(filetypes, mapkey))
                
        self.mapkeys = mapkeys

        #Create mappables out of filestruct and fieldmaps
        for i in range(len(self.filestruct[self.filetypes[0]])):
            mappable = {}
            for ft in usedfiletypes:
                mappable[self.filestruct[ft][i]] = fieldmap[ft]
            mappables.append(mappable)

        return mappables

    def readFITSMappable(self, mappable, fieldmap):
        """
        For the file help in mappable, read in the fields defined in
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

#        if sortbyz:
#            if 'redshift' not in mapunit.keys():
#                raise ValueError('There is no reshift field, cannot sort by redshift!')
#
#            zidx = mapunit['redshift'].argsort()
#            for mapkey in fieldmap[ft].keys():
#                mapunit[mapkey] = mapunit[mapkey][zidx]

        return mapunit

    def readMappable(self, mappable, fieldmap):
        """
        Default reader is FITS reader
        """
        
        return self.readFITSMappable(mappable, fieldmap)

    @abstractmethod
    def map(self, mappable):
        """
        Do some operations on a mappable unit of the catalog
        """
        
    def reduce(self):
        """
        Reduce the information produced by the map operations
        """
        for m in self.metrics:
            m.reduce()

    def setFieldMap(self, fieldmap):
        self.fieldmap = fieldmap

    def convert(self, mapunit, metrics):
        """
        Convert a map unit from the units given in the catalog
        to those required in metrics
        """
        
        for m in metrics:
            for key in m.unitmap:
                if key not in mapunit.keys(): continue
                try:
                    conversion = getattr(self, '{0}2{1}'.format(self.unitmap[key],m.unitmap[key]))
                except:
                    conversion = getattr(units, '{0}2{1}'.format(self.unitmap[key],m.unitmap[key]))

                mapunit[key] = conversion(mapunit[key])

        return mapunit

    def filter(self, mapunit, fieldmap):
        
        for i, key in enumerate(self.filters):
            filt = getattr(self, 'filter{0}'.format(key))
            
            if i==0:
                idx = filt(mapunit)
            else:
                idxi = filt(mapunit)
                idx = idx&idxi
                
        for key in mapunit.keys():
            if key in fieldmap.keys():
                mapunit[key] = mapunit[key][idx]

        return mapunit
        
                                     
        
