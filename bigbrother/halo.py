from __future__ import print_function, division
from collections import OrderedDict
from .massmetric import SimpleHOD, MassFunction, OccMass
from .basecatalog     import BaseCatalog
from helpers import SimulationAnalysis
import numpy as np
import healpy as hp

class HaloCatalog(BaseCatalog):
    """
    Base class for halo catalogs
    """

    def __init__(self, ministry, filestruct, zbins=None, **kwargs):

        self.ctype = 'halocatalog'
        self.zbins = zbins
        BaseCatalog.__init__(self, ministry, filestruct, **kwargs)


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
            ct = ['halocatalog']

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


    def unitConversion(self, mapunit):

        midx = mapunit['halomass']!=0.0

        for mapkey in mapunit.keys():
            mapunit[mapkey] = mapunit[mapkey][midx]
            if mapkey=='halomass':
                mapunit[mapkey] = np.log10(mapunit[mapkey][midx])

        return mapunit

    def readMappable(self, mappable, fieldmap):
        """
        Takes a mappable object and a fieldmap as inputs
        and returns a mapunit containing the data required
        by the fieldmap.
        """
        if self.reader=='fits':
            mapunit = self.readFITSMappable(mappable, fieldmap)
        elif self.reader=='rockstar':
            mapunit =  self.readRockstarMappable(mappable, fieldmap)

        return self.maskMappable(mapunit, mappable)


    def readRockstarMappable(self, mappable, fieldmap):
        """
        Takes in a mappable object, and a
        """

        #Fill in reader code here

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

	data = SimulationAnalysis.readHlist(fname, fields)

	for mapkey in fieldmap[ft].keys():
            mapunit[mapkey] = data[fieldmap[ft][mapkey]]
            if hasattr(fieldmap[ft][mapkey], '__iter__'):
                dt = mapunit[mapkey].dtype[0]
                ne = len(mapunit[mapkey])
                nf = len(fieldmap[ft][mapkey])
                mapunit[mapkey] = mapunit[mapkey].view(dt).reshape((ne,nf))

	return mapunit

class BCCHaloCatalog(HaloCatalog):
    """
    Class to handle BCC Halo catalogs
    """

    def __init__(self, ministry, filestruct, **kwargs):

        if 'unitmap' not in kwargs.keys():
            kwargs['unitmap'] = {'halomass':'msunh'}

        HaloCatalog.__init__(self, ministry, filestruct, **kwargs)

        self.metrics = [MassFunction(self.ministry, zbins=self.zbins,
                                      lightcone=True,jtype=self.jtype),
                        OccMass(self.ministry, zbins=self.zbins,
                                      lightcone=True,jtype=self.jtype)]
        if self.fieldmap is None:
            self.fieldmap = {'halomass':OrderedDict([('MVIR',['htruth'])]),
                             'occ':OrderedDict([('N19', ['htruth'])]),
                             'redshift':OrderedDict([('Z',['htruth'])])}
            self.hasz = True
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
