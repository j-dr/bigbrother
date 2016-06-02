from __future__ import print_function, division
from collections import OrderedDict
from .massmetric import SimpleHOD, MassFunction, OccMass
from .basecatalog import BaseCatalog
import numpy as np
import healpy as hp
import helpers


class HaloCatalog(BaseCatalog):
    """
    Base class for halo catalogs
    """

    def __init__(self, ministry, filestruct, fieldmap=None, 
                 nside=8, zbins=None, maskfile=None,
                 filters=None, unitmap=None, goodpix=1):

        self.ctype = 'halocatalog'
        BaseCatalog.__init__(self, ministry, filestruct, fieldmap=None, 
                             nside=8, maskfile=None, filters=None, 
                             unitmap=None, goodpix=1)

    
    def calculateArea(self, pixels, nside):
        """
        Calculate the area in the given pixels provided a mask
        that is pixelated with an nside greater than that of 
        the catalog
        """

        area = np.zeros(len(pixels))
        if self.mask is None:
            self.mask, self.maskhdr = hp.read_map(self.maskfile,h=True)
            self.maskhdr = dict(self.maskhdr)
        
        #our map should be finer than our file pixelation
        assert(nside<self.maskhdr['NSIDE']) 

        udmap = hp.ud_grade(np.arange(12*nside**2),self.maskhdr['NSIDE'])
        pixarea = hp.nside2pixarea(self.maskhdr['NSIDE'],degrees=True)
        for i,p in enumerate(pixels):
            pm, = np.where(udmap==p)
            area[i] = pixarea*len(self.mask[pm][self.mask[pm]>=self.goodpix])
            
        return area

    def getArea(self):
        
        if self.mask is None:
            return self.ministry.area
        else:
            return self.area

    def unitConversion(self, mapunit):

        midx = mapunit['mass']!=0.0

        for mapkey in mapunit.keys():
            mapunit[mapkey] = mapunit[mapkey][midx]
            if mapkey=='mass':
                mapunit[mapkey] = np.log10(mapunit[mapkey][midx])

        return mapunit


class BCCHaloCatalog(HaloCatalog):
    """
    Class to handle BCC Halo catalogs
    """

    def __init__(self, ministry, filestruct, unitmap=None, fieldmap=None, 
                 nside=8, zbins=None, maskfile=None, filters=None,
                 goodpix=1):

        if unitmap is None:
            unitmap = {'mass':'msunh'}

        HaloCatalog.__init__(self, ministry, filestruct, unitmap=unitmap, 
                             maskfile=maskfile, filters=filters, 
                             goodpix=goodpix)
        self.ministry = ministry
        self.metrics = [MassFunction(self.ministry, zbins=zbins),
                        OccMass(self.ministry, zbins=zbins)]

        self.nside = nside

        if fieldmap is None:
            self.fieldmap = {'mass':OrderedDict([('MVIR',['truth'])]),
                             'occ':OrderedDict([('N19', ['truth'])]),
                             'redshift':OrderedDict([('Z',['truth'])])}
            self.hasz = True
        else:
            self.fieldmap = fieldmap
            if 'redshift' in fieldmap.keys():
                self.sortbyz = True
            else:
                self.sortbyz = False

        self.unitmap = {'mass':'msunh'}

    def parseFileStruct(self, filestruct):
        """
        Given a filestruct object, namely a list of truth 
        and/or obs files, map fields in these files 
        to generalized observables which our map functions
        know how to deal with
        """
        self.filestruct = filestruct
        filetypes = self.filestruct.keys()
        self.filetypes = filetypes        

        if len(filestruct.keys())>1:
            opix =  np.array([int(t.split('/')[-1].split('.')[-2]) for t
                              in self.filestruct[filetypes[0]]])
            oidx = opix.argsort()

            for ft in filetypes:
                assert(len(filestruct[ft])==len(filestruct[filetypes[0]]))
                pix = np.array([int(t.split('/')[-1].split('.')[-2]) for t
                    in self.filestruct[ft]])
                idx = pix.argsort()
                assert(pix[idx]==opix[oidx])
                
                if len(idx)==1:
                    self.filestruct[ft] = [self.filestruct[ft][idx]]
                else:
                    self.filestruct[ft] = self.filestruct[ft][idx]

    def pixelVal(self,mappable):
        """
        Get the healpix cell value of this mappble using the fact
        that BCC files contain their healpix values
        """
        fts = mappable.keys()
        f1 = fts[0]
        pix = int(f1.split('.')[-2])
        
        return pix

    def map(self, mappable):
        """
        Do some operations on a mappable unit of the catalog
        """

        mapunit = self.readFITSMappable(mappable, sortbyz=self.sortbyz)
        mapunit = self.unitConversion(mapunit)

        for m in self.metrics:
            m.map(mapunit)
