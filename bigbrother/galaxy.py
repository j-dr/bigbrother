from __future__ import print_function, division
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
from astropy.cosmology import FlatLambdaCDM
import healpy as hp
import numpy as np
import helpers
import fitsio
import time

from .basecatalog     import BaseCatalog
from .magnitudemetric import LuminosityFunction, MagCounts, ColorColor, LcenMass, ColorMagnitude, FQuenched, FQuenchedLum
from .corrmetric      import GalaxyRadialProfileBCC


class GalaxyCatalog(BaseCatalog):
    """
    Base class for galaxy catalogs
    """

    def __init__(self, ministry, filestruct, fieldmap=None,
                 nside=8, zbins=None, maskfile=None,
                 filters=None, unitmap=None, goodpix=1,
                 reader='fits'):

        self.ctype = 'galaxycatalog'
        BaseCatalog.__init__(self, ministry, filestruct,
                                fieldmap=None, nside=8,
                                maskfile=None, filters=None,
                                unitmap=None, goodpix=1,
                                reader=reader)

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

        #our map should be at least as fine as our file pixelation
        assert(nside<=self.maskhdr['NSIDE'])

        udmap = hp.ud_grade(np.arange(12*nside**2),self.maskhdr['NSIDE'])
        pixarea = hp.nside2pixarea(self.maskhdr['NSIDE'],degrees=True)
        for i,p in enumerate(pixels):
            pm, = np.where(udmap==p)
            area[i] = pixarea*len(self.mask[pm][self.mask[pm]>=self.goodpix])

        return area

    def parseFileStruct(self, filestruct):
        """
        Given a filestruct object, namely a list of truth
        and/or obs files, map fields in these files
        to generalized observables which our map functions
        know how to deal with
        """
        self.filestruct = filestruct
        self.filetypes = self.filestruct.keys()

    def getArea(self):

        arm = np.array([True if m.__class__.__name__=="Area" else False
                  for m in self.ministry.metrics])
        am = any(arm)
        if am:
            idx, = np.where(arm==True)

        if (self.mask is None) & (not am):
            return self.ministry.area
        elif am:
            return self.ministry.metrics[idx].area
        else:
            return self.calculateArea()

    def readMappable(self, mappable, fieldmap):

        return self.readFITSMappable(mappable, fieldmap)

    def filterAppmag(self, mapunit, bands=None, badval=99.):

        if bands is None:
            bands = range(mapunit['appmag'].shape[1])

        for i, b in enumerate(bands):
            if i==0:
                idx = mapunit['appmag'][:,b]!=badval
            else:
                idxi = mapunit['appmag'][:,b]!=badval
                idx = idx&idxi

        return idx

class BCCCatalog(GalaxyCatalog):
    """
    BCC style ADDGALS catalog
    """

    def __init__(self, ministry, filestruct, fieldmap=None,
                 nside=8, zbins=None, maskfile=None,
                 filters=None, unitmap=None, goodpix=1):
        GalaxyCatalog.__init__(self, ministry, filestruct, maskfile=maskfile, goodpix=goodpix)
        self.min = ministry
        self.metrics = [LuminosityFunction(self.ministry, zbins=zbins),
                        MagCounts(self.ministry, zbins=zbins),
                        LuminosityFunction(self.ministry, zbins=zbins, central_only=True),
                        LcenMass(self.ministry, zbins=zbins),
                        ColorMagnitude(self.ministry, zbins=zbins, usebands=[0,1]),
                        ColorMagnitude(self.ministry, zbins=zbins,                                            usebands=[0,1],
                          central_only=True),
                        ColorColor(self.ministry, zbins=zbins, usebands=[0,1,2]),
                        FQuenched(self.ministry, zbins=np.linspace(0,2.0,30)),
                        FQuenchedLum(self.ministry, zbins=zbins),
                        GalaxyRadialProfileBCC(self.ministry, zbins=zbins)]

        self.nside = nside
        if filters is None:
            self.filters = ['Appmag']
        else:
            self.filters = filters

        self.unitmap = {'luminosity':'mag', 'appmag':'mag', 'halomass':'msunh'}

        if fieldmap is None:
            self.fieldmap = {'luminosity':OrderedDict([('AMAG',['truth'])]),
                             'appmag':OrderedDict([('MAG_G',['obs']), ('MAG_R',['obs']),
                                                   ('MAG_I',['obs']), ('MAG_Z',['obs']),
                                                   ('MAG_Y',['obs'])]),
                             'redshift':OrderedDict([('Z',['truth'])])}
            self.sortbyz = True
        else:
            self.fieldmap = fieldmap
            if 'redshift' in fieldmap.keys():
                self.sortbyz = True
            else:
                self.sortbyz = False

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
            lfs = [len(filestruct[ft]) for ft in filestruct.keys()]
            mlfs = min(lfs)
            oft = lfs.index(mlfs)
            opix =  np.array([int(t.split('/')[-1].split('.')[-2]) for t
                              in self.filestruct[filetypes[oft]]])
            oidx = opix.argsort()

            for ft in filetypes:
                pix = np.array([int(t.split('/')[-1].split('.')[-2]) for t
                    in self.filestruct[ft]])
                idx = np.in1d(pix, opix)
                self.filestruct[ft] = self.filestruct[ft][idx]
                pix = pix[idx]
                idx = pix.argsort()
                assert((pix[idx]==opix[oidx]).all())

                if len(idx)==1:
                    self.filestruct[ft] = [self.filestruct[ft][idx]]
                else:
                    self.filestruct[ft] = self.filestruct[ft][idx]

                print(len(self.filestruct[ft]))

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

        if self.maskfile!=None:
            pix = self.pixelVal(mappable)
            a = self.calculateArea([pix],self.nside)
            self.area += a[0]

        for m in self.metrics:
            m.map(mapunit)


class S82SpecCatalog(GalaxyCatalog):
    """
    SDSS DR6 stripe82 photometric galaxy catalog (for mag/count, color comparisons)
    """

    def __init__(self, ministry, filestruct, fieldmap=None,
                 unitmap=None, filters=None, nside=8):
        GalaxyCatalog.__init__(self, ministry, filestruct,
                               unitmap=unitmap, filters=filters)
        self.ministry = ministry
        self.parseFileStruct(None)
        self.metrics = [LuminosityFunction(self.ministry)]
        self.fieldmap = {'luminosity':OrderedDict([('AMAG',['spec'])]),
                         'redshift':OrderedDict([('Z',['spec'])])}


    def parseFileStruct(self, filestruct):
        """
        Only have one file so this is trivial
        """
        self.filestruct = {'spec':['/nfs/slac/g/ki/ki01/mbusha/data/sdss/dr6/cooper/combined_dr6_cooper.fit']}
        self.filetypes = self.filestruct.keys()

    def map(self, mappable):
        """
        Do some operations on a mappable unit of the catalog
        """

        mapunit = self.readFITSMappable(mappable)
        for m in self.metrics:
            m.map(mapunit)

    def reduce(self):
        """
        Reduce the information produced by the map operations
        """
        for m in self.metrics:
            m.reduce()

class S82PhotCatalog(GalaxyCatalog):
    """
    SDSS DR6 stripe82 photometric galaxy catalog (for mag/count, color comparisons)
    """

    def __init__(self, ministry, filestruct, fieldmap=None, unitmap=None,
                 filters=None, nside=8):
        GalaxyCatalog.__init__(self, ministry, filestruct, filters=filters,
                               unitmap=unitmap)
        self.ministry = ministry
        self.parseFileStruct(None)
        self.metrics = [MagCounts(self.ministry), ColorColor(self.ministry)]
        self.fieldmap = {'appmag':OrderedDict([('G',['phot']), ('R',['phot']),
                                   ('I',['phot']), ('Z',['phot'])]),
                         'redshift':OrderedDict([('PHOTOZCC2',['phot'])])}

    def parseFileStruct(self, filestruct):
        """
        Only have one file so this is trivial
        """
        self.filestruct = {'phot':['/nfs/slac/g/ki/ki01/mbusha/data/sdss/dr6/umich/DR6_Input_catalog_ellipticity_stripe82.fit']}
        self.filetypes = self.filestruct.keys()


    def map(self, mappable):
        """
        Do some operations on a mappable unit of the catalog
        """

        mapunit = self.readFITSMappable(mappable)
        for m in self.metrics:
            m.map(mapunit)

    def reduce(self):
        """
        Reduce the information produced by the map operations
        """
        for m in self.metrics:
            m.reduce()


class DESGoldCatalog(GalaxyCatalog):
    """
    DES Gold catalog in the style of Y1A1.
    """

    def __init__(self, ministry, filestruct, fieldmap=None,
                 unitmap=None, filters=None, nside=8,
                 maskfile=None, goodpix=1):

        GalaxyCatalog.__init__(self, ministry, filestruct,
                               maskfile=maskfile,goodpix=goodpix,
                               filters=filters, unitmap=unitmap)
        self.necessaries = ['modest']
        self.ministry = ministry
        self.parseFileStruct(filestruct)
        self.nside = nside
        self.metrics = [MagCounts(self.ministry, zbins=None),
                        ColorColor(self.ministry, zbins=None)]
        if fieldmap==None:
            self.fieldmap = {'appmag':OrderedDict([('FLUX_AUTO_G',['auto']),
                                                   ('FLUX_AUTO_R',['auto']),
                                                   ('FLUX_AUTO_I',['auto']),
                                                   ('FLUX_AUTO_Z',['auto'])]),
                             'modest':OrderedDict([('MODEST_CLASS',['basic'])]),
                             'polar_ang':OrderedDict([('DEC',['basic'])]),
                             'azim_ang':OrderedDict([('RA',['basic'])])}
        else:
            self.fieldmap = fieldmap

        if unitmap is None:
            self.unitmap = {'appmag':'flux', 'polar_ang':'dec', 'azim_ang':'ra'}
        else:
            self.unitmap = unitmap

        if filters is None:
            self.filters = ['Modest']
        else:
            self.filters = filters

    def parseFileStruct(self, filestruct):

        self.filestruct = filestruct
        filetypes = self.filestruct.keys()
        self.filetypes = filetypes

        if len(filestruct.keys())>1:
            opix =  np.array([int(t.split('_')[-2].split('pix')[-1]) for t
                              in self.filestruct[filetypes[0]]])
            oidx = opix.argsort()

            for ft in filetypes:
                assert(len(filestruct[ft])==len(filestruct[filetypes[0]]))
                pix = np.array([int(t.split('_')[-2].split('pix')[-1]) for t
                    in self.filestruct[ft]])
                idx = pix.argsort()
                assert((pix[idx]==opix[oidx]).all())

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
        pix = int(f1.split('_')[-2].split('pix')[-1])

        return pix


    def map(self, mappable):
        """
        Do some operations on a mappable unit of the catalog
        """

        mapunit = self.readFITSMappable(mappable)
        mapunit = self.unitConversion(mapunit)
        mapunit = self.filterModest(mapunit)

        if self.maskfile!=None:
            pix = self.pixelVal(mappable)
            a = self.calculateArea([pix],self.nside)
            self.area += a[0]

        for m in self.metrics:
            m.map(mapunit)

    def unitConversion(self, mapunit):

        for mapkey in mapunit.keys():
            if mapkey=='appmag':
                mapunit[mapkey] = 30.0 - 2.5*np.log10(mapunit[mapkey])

        return mapunit

    def filterModest(self, mapunit):

        return mapunit['modest']==1

    def reduce(self):
        """
        Reduce the information produced by the map operations
        """
        for m in self.metrics:
            m.reduce()
