from __future__ import print_function, division
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
from astropy.cosmology import FlatLambdaCDM
import healpy as hp
import numpy as np
import fitsio
import time

from .magnitudemetric import LuminosityFunction, MagCounts, ColorColor, LcenMass, ColorMagnitude, FQuenched, FQuenchedLum, ColorDist, FRed
from .lineofsight     import DNDz
from .massmetric      import Richness, GalHOD, GalCLF
from .healpix_utils   import Area, PixMetric, HealpixMap
from .corrmetric      import GalaxyRadialProfileBCC
from .basecatalog     import BaseCatalog

class GalaxyCatalog(BaseCatalog):
    """
    Base class for galaxy catalogs
    """

    def __init__(self, ministry, filestruct, zbins=None, zp=None, Q=None,**kwargs):


        self.ctype = 'galaxycatalog'
        self.zbins = zbins
        self.zp = zp
        self.Q  = Q
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
            fk = self.filestruct.keys()

            for f in self.filestruct[fk[0]]:
                p = int(f.split('.')[-2])

                if (self.filenside == self.groupnside):
                    fpix.append([p])
                else:
                    if not self.nest:
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

            pmetric = PixMetric(self.ministry, self.groupnside,
                                  catalog_type=ct,nest=self.nest)
            mg = self.ministry.genMetricGroups([pmetric])
            ms = mg[0][1]
            fm = mg[0][0]

            mappables = self.ministry.genMappables(mg[0])

            if self.ministry.parallel:
                from mpi4py import MPI

                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                size = comm.Get_size()

                mappables = mappables[rank::size]

            for i, mappable in enumerate(mappables):

                mapunit = self.ministry.readMappable(mappable, fm)
                print('converting before getting file pixels')
                if (not hasattr(ms,'__iter__')) and ('only' in ms.aschema):
                    mapunit = self.ministry.scListToDict(mapunit)
                    mapunit = self.ministry.convert(mapunit, ms)
                    mapunit = self.ministry.filter(mapunit)

                elif 'only' in ms[0].aschema:
                    mapunit = self.ministry.scListToDict(mapunit)
                    mapunit = self.ministry.convert(mapunit, ms)
                    mapunit = self.ministry.filter(mapunit)

                if ((ms[0].aschema == 'galaxygalaxy')
                  | (ms[0].aschema == 'halohalo')):
                    mapunit = self.ministry.dcListToDict(mapunit)
                    mapunit = self.ministry.convert(mapunit, ms)
                    mapunit = self.ministry.filter(mapunit)

                fpix.append(pmetric.map(mapunit))

                del mapunit

            if self.ministry.parallel:
                gfpix = comm.allgather(fpix)
                fpix = []
                for fp in gfpix:
                    fpix.extend(fp)

        return fpix


    def getArea(self, jackknife=False):

        arm = np.array([True if m.__class__.__name__=="Area" else False
                  for m in self.ministry.metrics])
        am = any(arm)
        if am:
            idx, = np.where(arm==True)[0]

        if not jackknife:
            if (self.mask is None) & (not am):
                return np.array([self.ministry.area])
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

        return mapunit

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

    def filter10sigma(self, mapunit, bands=None, badval=99.):
        if bands is None:
            if len(mapunit['appmag_err'].shape)>1:
                bands = range(mapunit['appmag_err'].shape[1])
            else:
                bands = [0]
                mapunit['appmag_err'] = np.atleast_2d(mapunit['appmag_err']).T

        for i, b in enumerate(bands):
            if i==0:
                idx = (mapunit['appmag_err'][:,b]<0.1) & (np.isfinite(mapunit['appmag_err'][:,b])) & (~np.isnan(mapunit['appmag_err'][:,b]))
            else:
                idxi = (mapunit['appmag_err'][:,b]<0.1) & (np.isfinite(mapunit['appmag_err'][:,b])) & (~np.isnan(mapunit['appmag_err'][:,b]))
                idx = idx&idxi

        return idx

    def filterStar(self, mapunit):
        print('Filtering stars')

        return mapunit['pstar']<0.2

    def filterLssflag(self, mapunit):
        print("Filtering LSS")
        
        return (mapunit['lssflag']==1) | (mapunit['lssflag']==3)

    def filterWlflag(self, mapunit):
        print("Filtering WL")
        
        return (mapunit['wlflag']==2) | (mapunit['wlflag']==3)



class BCCCatalog(GalaxyCatalog):
    """
    BCC style ADDGALS catalog
    """

    def __init__(self, ministry, filestruct, ctfile=None,**kwargs):

        filters = kwargs.pop('filters', ['Appmag'])

        GalaxyCatalog.__init__(self, ministry, filestruct, filters=filters,
                               **kwargs)

        self.ctfile = ctfile
        self.metrics = [Area(self.ministry, jtype=self.jtype),
                        LuminosityFunction(self.ministry,
                                            zbins=self.zbins,
                                            tag="AllLF",
                                            jtype=self.jtype),
                        LuminosityFunction(self.ministry,
                                            zbins=np.linspace(0.0,1.0,11),
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
                        LcenMass(self.ministry,
                                  zbins=np.linspace(0.15,0.9,6),
                                  jtype=self.jtype),
                        ColorDist(self.ministry,
                                    jtype=self.jtype,
                                    tag="CDZBin",
                                    pdf=True,
                                    usebands=[[0,1],[0,2],[1,2]]),
                        ColorDist(self.ministry,
                                    appmag=True,
                                    jtype=self.jtype,
                                    tag="CDAppZBin",
                                    pdf=True,
                                    usebands=[[0,1],[0,2],[1,2]]),
                        ColorDist(self.ministry,
                                    appmag=True,
                                    zbins=np.array([0.0,2.0]),
                                    jtype=self.jtype,
                                    tag="CDAppAll",
                                    pdf=True,
                                    usebands=[[0,1],[0,2],[1,2]]),
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
                        ColorMagnitude(self.ministry,
                                       zbins=np.array([0.0,2.0]),
                                       usebands=[0,1],
                                       tag="AllCMApp",
                                       jtype=self.jtype,
                                       appmag=True),                        
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
                                    magcuts=[21.5],
                                    tag="CCSDSSZ",
                                    jtype=self.jtype,
                                    appmag=True),
                        ColorColor(self.ministry,
                                    zbins=None,
                                    usebands=[0,1,2],
                                    tag="CCAppAllZ",
                                    jtype=self.jtype,
                                    appmag=True),
                        FQuenched(self.ministry,
                                    zbins=np.linspace(0,2.0,30),
                                    jtype=self.jtype,
                                    appmag=False,
                                    tag='FQAmag'),
                        FQuenched(self.ministry,
                                    zbins=np.linspace(0,2.0,30),
                                    jtype=self.jtype,
                                    appmag=True,
                                    tag='FQAppmag'),
                        FQuenchedLum(self.ministry,
                                      zbins=np.array([0.0,0.3,0.6,0.8,1.0,1.2]),
                                      cbins=np.linspace(0.0,1.2,60),
                                      magbins=np.linspace(-22,-18,20),
                                      jtype=self.jtype,
                                      cinds=[0,1],
                                      splitcolor=0.8),
                        FRed(self.ministry,
                             zbins=np.linspace(0,2.0,30),
                             jtype=self.jtype,
                             ctfile=self.ctfile),
                        GalaxyRadialProfileBCC(self.ministry,
                                                zbins=np.linspace(0.15,0.9,6),
                                                jtype=self.jtype,
                                                mcutind=1),
                        Richness(self.ministry,
                                  zbins=self.zbins,
                                  jtype=self.jtype),
                        GalCLF(self.ministry,
                               zbins=self.zbins,
                               jtype=self.jtype,
                               tag='galclf'),
                        GalCLF(self.ministry,
                               zbins=np.array([0.1,0.3,0.5,0.7,0.9]),
                               massbins=np.logspace(13.5,15,4),
                               jtype=self.jtype,
                               tag='galrmpclf'),
                        GalHOD(self.ministry,
                               zbins=np.linspace(0.0, 0.6, 7),
                               jtype=self.jtype,
                               magcuts=[-22.51, -21.756, -21.09, -20.35,-19.79],
                               cutband=1,
                               upper_limit=True,
                               tag='lowz_galhod'),
                        GalHOD(self.ministry,
                               zbins=np.linspace(0.15,0.9,6),
                               jtype=self.jtype,
                               magcuts=[-22.51, -21.756, -21.09, -20.35,-19.79],
                               cutband=1,
                               upper_limit=True,
                               tag='lensbin_galhod'),
                        DNDz(self.ministry,
                              magbins=[20, 21, 22, 23],
                              jtype=self.jtype),
                        HealpixMap(self.ministry)]

        if self.unitmap is None:
            self.unitmap = {'luminosity':'mag', 'appmag':'mag', 'halomass':'msunh',
                            'azim_ang':'ra', 'polar_ang':'dec', 'redshift':'z',
                            'rhalo':'mpch', 'r200':'mpch'}

        if self.fieldmap is None:
            self.fieldmap = {'luminosity':OrderedDict([('AMAG',['gtruth'])]),
                             'appmag':OrderedDict([('MAG_G',['gobs']),
                                                   ('MAG_R',['gobs']),
                                                   ('MAG_I',['gobs']),
                                                   ('MAG_Z',['gobs']),
                                                   ('MAG_Y',['gobs'])]),
                             'redshift':OrderedDict([('Z',['gtruth'])]),
                             'azim_ang':OrderedDict([('RA', ['gobs'])]),
                             'polar_ang':OrderedDict([('DEC', ['gobs'])]),
                             'central':OrderedDict([('CENTRAL', ['gtruth'])]),
                             'halomass':OrderedDict([('M200', ['gtruth'])]),
                             'rhalo':OrderedDict([('RHALO', ['gtruth'])]),
                             'r200':OrderedDict([('R200', ['gtruth'])]),
                             'haloid':OrderedDict([('HALOID', ['gtruth'])]),
                             'density':OrderedDict([('SIGMA5', ['gtruth'])])}
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

        filters = kwargs.pop('filters', ['Modest', 'Appmag', 'Photoz', 'Badregion'])
        necessaries = kwargs.pop('necessaries', ['modest', 'badregion'])

        GalaxyCatalog.__init__(self, ministry, filestruct, goodpix=1, filters=filters,
                               necessaries=necessaries, **kwargs)

        self.parseFileStruct(filestruct)
        self.metrics = [Area(self.ministry, jtype=self.jtype),
                        MagCounts(self.ministry, zbins=self.zbins, tag="BinZ",jtype=self.jtype),
                        MagCounts(self.ministry, zbins=None, tag="AllZ", jtype=self.jtype),
                        ColorDist(self.ministry,
                                    appmag=True,
                                    jtype=self.jtype,
                                    tag="CDAppZBin",
                                    pdf=True,
                                    usebands=[[0,1],[0,2],[1,2]]),
                        ColorDist(self.ministry,
                                    appmag=True,
                                    zbins=np.array([0.0,2.0]),
                                    jtype=self.jtype,
                                    tag="CDAppAll",
                                    pdf=True,
                                    usebands=[[0,1],[0,2],[1,2]]),
                        ColorMagnitude(self.ministry, zbins=self.zbins, usebands=[0,1], tag="AllCMAppBinZ", jtype=self.jtype, appmag=True),
                        ColorMagnitude(self.ministry, zbins=np.array([0.0,2.0]), usebands=[0,1], tag="AllCMApp", jtype=self.jtype, appmag=True),
                        ColorColor(self.ministry, zbins=self.zbins, usebands=[0,1,2], tag="CCAppBinZ", jtype=self.jtype, appmag=True),
                        ColorColor(self.ministry, zbins=None,
                         usebands=[0,1,2], tag="CCAppAllZ", jtype=self.jtype, appmag=True),
                        DNDz(self.ministry, magbins=[20, 21, 22, 23], jtype=self.jtype)]



        if self.fieldmap==None:
            self.fieldmap = {'appmag':OrderedDict([('FLUX_AUTO_G',['auto']),
                                                   ('FLUX_AUTO_R',['auto']),
                                                   ('FLUX_AUTO_I',['auto']),
                                                   ('FLUX_AUTO_Z',['auto'])]),
                             'modest':OrderedDict([('MODEST_CLASS',['basic'])]),
                             'badregion':OrderedDict([('BADREGION',['badregion'])]),
                             'polar_ang':OrderedDict([('DEC',['basic'])]),
                             'azim_ang':OrderedDict([('RA',['basic'])]),
                             'redshift':OrderedDict([('BPZ_MC',['photoz'])])}

        if self.unitmap is None:
            self.unitmap = {'appmag':'flux', 'polar_ang':'dec', 'azim_ang':'ra', 'redshift':'z'}


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
        print('Filtering modest')

        return mapunit['modest']==1

    def filterRedshift(self, mapunit):
        print('Filtering redshift >0')

        return mapunit['redshift']>0

    def filterBadregion(self, mapunit):
        print('Filtering badregion')

        print('No badregions: {0}'.format((mapunit['badregion']==0).all()))

        return mapunit['badregion']==0
