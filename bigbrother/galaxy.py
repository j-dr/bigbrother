from __future__ import print_function, division
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
from astropy.cosmology import FlatLambdaCDM
import healpy as hp
import numpy as np
import fitsio
import time

from .magnitudemetric import LuminosityFunction, MagCounts, ColorColor, LcenMass, ColorMagnitude, FQuenched, FQuenchedLum, ColorDist
from .lineofsight     import DNDz
from .massmetric      import Richness
from .healpix_utils   import Area
from .corrmetric      import GalaxyRadialProfileBCC
from .basecatalog     import BaseCatalog

class GalaxyCatalog(BaseCatalog):
    """
    Base class for galaxy catalogs
    """

    def __init__(self, ministry, filestruct, zbins=None, **kwargs):


        self.ctype = 'galaxycatalog'
        self.zbins = zbins
        BaseCatalog.__init__(self, ministry, filestruct, **kwargs)


    def calculateMaskArea(self, pixels, nside):
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

    def getFilePixels(self, nside):
        """
        Get the healpix cells occupied by galaxies
        in each file. Assumes files have already been
        sorted correctly by parseFileStruct
        """
        fpix = []

        #BCC catalogs have pixels in filenames
        if (('BCC' in self.__class__.__name__) &
          (self.filenside is not None) & (self.filenside>=self.groupnside)):
            print('BCC')
            fk = self.filestruct.keys()

            for f in self.filestruct[fk[0]]:
                p = int(f.split('.')[-2])

                if (self.filenside == self.groupnside):
                    fpix.append([p])
                else:
                    if not self.nest:
                        print('not nest')
                        while p > 12*self.filenside**2:
                            p = p - 1000
                        p = hp.ring2nest(self.filenside, p)

                    o1 = int(np.log2(self.filenside))
                    o2 = int(np.log2(self.groupnside))

                    base = int(p >> 2*o1)
                    hosubpix = int(p & ( ( 1 << ( 2 * o1 ) ) - 1 ))
                    losubpix = int(hosubpix // ( 1 << 2 * ( o1 - o2) ))
                    p  = int(base * ( 1 << ( 2 * o2 ) ) + losubpix)

                    fpix.append([p])

        else:
            ct = ['galaxycatalog']

            pmetric = PixMetric(self.ministry, self.groupnside, catalog_type=ct)
            mg = self.ministry.genMetricGroups([pmetric])
            ms = mg[0][1]
            fm = mg[0][0]

            for mappable in self.ministry.genMappable(fm):
                mapunit = self.ministry.readMappable(mappable, fm)
                mapunit = self.ministry.treeToDict(mapunit)
                mapunit = self.convert(mapunit, ms)
                fpix.append(pmetric(mapunit))

        return fpix


    def getArea(self, jackknife=False):

        arm = np.array([True if m.__class__.__name__=="Area" else False
                  for m in self.ministry.metrics])
        am = any(arm)
        if am:
            idx, = np.where(arm==True)[0]

        if not jackknife:
            if (self.mask is None) & (not am):
                return self.ministry.area
            elif am:
                return self.ministry.metrics[idx].area
            else:
                return self.calculateMaskArea()
        else:
            return self.ministry.metrics[idx].jarea

    def readMappable(self, mappable, fieldmap):

        if self.reader=='fits':
            mapunit =  self.readFITSMappable(mappable, fieldmap)
        elif self.reader=='ascii':
            mapunit = self.readAsciiMappable(mappable, fieldmap)
        else:
            raise(ValueError("Reader {0} is not supported for galaxy catalogs".format(self.reader)))

        return self.maskMappable(mapunit, mappable)

    def readAsciiMappable(self, mappable, fieldmap):

        mapunit = {}
        ft      = mappable.dtype
        fname   = mappable.name

        fields = []
        for val in fieldmap[ft].values():
            if hasattr(val, '__iter__'):
                fields.extend(val)
            else:
                fields.extend([val])

        data = np.genfromtxt(fname, usecols=fields)
        data = data.reshape((len(data), len(fields)))

        for mapkey in fieldmap[ft].keys():
            mapunit[mapkey] = data[:,fields.index(fieldmap[ft][mapkey])]
            if hasattr(fieldmap[ft][mapkey], '__iter__'):
                dt = mapunit[mapkey].dtype[0]
                ne = len(mapunit[mapkey])
                nf = len(fieldmap[ft][mapkey])
                mapunit[mapkey] = mapunit[mapkey].view(dt).reshape((ne,nf))

        return mapunit


    def filterAppmag(self, mapunit, bands=None, badval=99.):
        print('Filtering appmag')
        if bands is None:
            if len(mapunit['appmag'].shape)>1:
                bands = range(mapunit['appmag'].shape[1])
            else:
                bands = [0]
                mapunit['appmag'] = np.atleast_2d(mapunit['appmag']).T

        for i, b in enumerate(bands):
            if i==0:
                idx = (mapunit['appmag'][:,b]!=badval) & (np.isfinite(mapunit['appmag'][:,b])) & (~np.isnan(mapunit['appmag'][:,b]))
            else:
                idxi = (mapunit['appmag'][:,b]!=badval) & (np.isfinite(mapunit['appmag'][:,b])) & (~np.isnan(mapunit['appmag'][:,b]))
                idx = idx&idxi

        return idx

class BCCCatalog(GalaxyCatalog):
    """
    BCC style ADDGALS catalog
    """

    def __init__(self, ministry, filestruct, **kwargs):

        GalaxyCatalog.__init__(self, ministry, filestruct, **kwargs)
        self.metrics = [Area(self.ministry, jtype=self.jtype),
                        LuminosityFunction(self.ministry,
                                            zbins=self.zbins,
                                            tag="AllLF",
                                            jtype=self.jtype),
                        MagCounts(self.ministry,
                                    zbins=self.zbins,
                                    tag="BinZ",
                                    jtype=self.jtype),
                        MagCounts(self.ministry, zbins=None,
                                    tag="AllZ",
                                    jtype=self.jtype),
                        LuminosityFunction(self.ministry,
                                            zbins=self.zbins,
                                            central_only=True,
                                            tag="CentralLF",
                                            jtype=self.jtype),
                        LcenMass(self.ministry,
                                  zbins=self.zbins,
                                  jtype=self.jtype),
                        ColorDist(self.ministry,
                                    zbins=self.zbins,
                                    jtype=self.jtype,
                                    tag="CDZBin"),
                        ColorDist(self.ministry,
                                    zbins=self.zbins,
                                    appmag=True,
                                    jtype=self.jtype,
                                    tag="CDAppZBin"),
                        ColorMagnitude(self.ministry,
                                        zbins=self.zbins,
                                        usebands=[0,1],
                                        tag="AllCMBinZ",
                                        jtype=self.jtype),
                        ColorMagnitude(self.ministry,
                                        zbins=self.zbins,
                                        usebands=[0,1],
                                        tag="AllAppCMBinZ",
                                        jtype=self.jtype,
                                        appmag=True),
                        ColorMagnitude(self.ministry,
                                        zbins=self.zbins,
                                        usebands=[0,1],
                                        central_only=True,
                                        tag="CentralCMBinZ",
                                        jtype=self.jtype),
                        ColorColor(self.ministry,
                                    zbins=self.zbins,
                                    usebands=[0,1,2],
                                    tag="CCBinZ",
                                    jtype=self.jtype),
                        ColorColor(self.ministry,
                                    zbins=self.zbins,
                                    usebands=[0,1,2],
                                    tag="CCAppBinZ",
                                    jtype=self.jtype,
                                    appmag=True),
                        ColorColor(self.ministry,
                                    zbins=np.linspace(0.0, 0.2, 5),
                                    usebands=[0,1,2],
                                    tag="CCSDSSZ",
                                    jtype=self.jtype),
                        ColorColor(self.ministry,
                                    zbins=None,
                                    usebands=[0,1,2],
                                    tag="CCAllZ",
                                    jtype=self.jtype),
                        FQuenched(self.ministry,
                                    zbins=np.linspace(0,2.0,30),
                                    jtype=self.jtype,
                                    tag='FQAmag'),
                        FQuenched(self.ministry,
                                    zbins=np.linspace(0,2.0,30),
                                    jtype=self.jtype,
                                    appmag=True,
                                    tag='FQAppmag'),
                        FQuenchedLum(self.ministry,
                                      zbins=self.zbins,
                                      jtype=self.jtype),
                        GalaxyRadialProfileBCC(self.ministry,
                                                zbins=self.zbins,
                                                jtype=self.jtype),
                        Richness(self.ministry, zbins=self.zbins,
                                  jtype=self.jtype),
                        DNDz(self.ministry,
                              magbins=[20, 21, 22, 23])]

        if len(self.filters) == 0:
            self.filters = ['Appmag']

        if self.unitmap is None:
            self.unitmap = {'luminosity':'mag', 'appmag':'mag', 'halomass':'msunh',
                            'azim_ang':'ra', 'polar_ang':'dec'}

        if self.fieldmap is None:
            self.fieldmap = {'luminosity':OrderedDict([('AMAG',['gtruth'])]),
                             'appmag':OrderedDict([('MAG_G',['obs']), ('MAG_R',['obs']),
                                                   ('MAG_I',['obs']), ('MAG_Z',['obs']),
                                                   ('MAG_Y',['obs'])]),
                             'redshift':OrderedDict([('Z',['gtruth'])])}
            self.sortbyz = True
        else:
            if 'redshift' in self.fieldmap.keys():
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

    def pixelVal(self,mappable):
        """
        Get the healpix cell value of this mappble using the fact
        that BCC files contain their healpix values
        """
        fts = mappable.keys()
        f1 = fts[0]
        pix = int(f1.split('.')[-2])

        return pix

class S82SpecCatalog(GalaxyCatalog):
    """
    SDSS DR6 stripe82 photometric galaxy catalog (for mag/count, color comparisons)
    """

    def __init__(self, ministry, filestruct, **kwargs):
        GalaxyCatalog.__init__(self, ministry, filestruct, **kwargs)

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

class S82PhotCatalog(GalaxyCatalog):
    """
    SDSS DR6 stripe82 photometric galaxy catalog (for mag/count, color comparisons)
    """

    def __init__(self, ministry, filestruct, **kwargs):

        GalaxyCatalog.__init__(self, ministry, filestruct, **kwargs)

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


class DESGoldCatalog(GalaxyCatalog):
    """
    DES Gold catalog in the style of Y1A1.
    """

    def __init__(self, ministry, filestruct, **kwargs):


        GalaxyCatalog.__init__(self, ministry, filestruct, goodpix=1, **kwargs)

        self.necessaries = ['modest']
        self.parseFileStruct(filestruct)
        self.metrics = [Area(self.ministry, jtype=self.jtype),
                        MagCounts(self.ministry, zbins=self.zbins, tag="BinZ",jtype=self.jtype),
                        MagCounts(self.ministry, zbins=None, tag="AllZ", jtype=self.jtype),
                        ColorDist(self.ministry,
                                    zbins=self.zbins,
                                    appmag=True,
                                    jtype=self.jtype,
                                    tag="CDAppZBin"),
                        ColorMagnitude(self.ministry, zbins=self.zbins, usebands=[0,1], tag="AllCMBinZ", jtype=self.jtype, appmag=True),
                        ColorColor(self.ministry, zbins=self.zbins, usebands=[0,1,2], tag="BinZ", jtype=self.jtype, appmag=True),
                        ColorColor(self.ministry, zbins=None,
                         usebands=[0,1,2], tag="AllZ", jtype=self.jtype, appmag=True),
                        DNDz(self.ministry, magbins=[20, 21, 22, 23])]



        if self.fieldmap==None:
            self.fieldmap = {'appmag':OrderedDict([('FLUX_AUTO_G',['auto']),
                                                   ('FLUX_AUTO_R',['auto']),
                                                   ('FLUX_AUTO_I',['auto']),
                                                   ('FLUX_AUTO_Z',['auto'])]),
                             'modest':OrderedDict([('MODEST_CLASS',['basic'])]),
                             'polar_ang':OrderedDict([('DEC',['basic'])]),
                             'azim_ang':OrderedDict([('RA',['basic'])]),
                             'redshift':OrderedDict([('BPZ_MC',['photoz'])])}

        if self.unitmap is None:
            self.unitmap = {'appmag':'flux', 'polar_ang':'dec', 'azim_ang':'ra'}

        if len(self.filters) == 0:
            self.filters = ['Modest', 'Appmag', 'Photoz']

    def parseFileStruct(self, filestruct):

        self.filestruct = filestruct
        filetypes = self.filestruct.keys()
        self.filetypes = filetypes

        nfiles = np.array([len(self.filestruct[ft]) for ft in filetypes])
        midx = np.argmin(nfiles)
        dn = nfiles[1:]-nfiles[:-1]

        if len(dn)!=0 and not (dn==0).all():
            print('File types do not all have the same number of files!')
            print('Number of files per filetype : {0}'.format(zip(filetypes, nfiles)))

        if len(filestruct.keys())>1:
            opix =  np.array([int(t.split('_')[-2].split('pix')[-1]) for t
                              in self.filestruct[filetypes[midx]]])
            oidx = opix.argsort()

            for ft in filetypes:
                pix = np.array([int(t.split('_')[-2].split('pix')[-1]) for t
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


    def pixelVal(self,mappable):
        """
        Get the healpix cell value of this mappble using the fact
        that BCC files contain their healpix values
        """
        fts = mappable.keys()
        f1 = fts[0]
        pix = int(f1.split('_')[-2].split('pix')[-1])

        return pix


    def unitConversion(self, mapunit):

        for mapkey in mapunit.keys():
            if mapkey=='appmag':
                mapunit[mapkey] = 30.0 - 2.5*np.log10(mapunit[mapkey])

        return mapunit

    def filterModest(self, mapunit):

        return mapunit['modest']==1

    def filterPhotoz(self, mapunit):

        return mapunit['redshift']>0
