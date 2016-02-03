from __future__ import print_function, division
from collections import OrderedDict
from metrics import LuminosityFunction, MagCounts, ColorColor
from abc import ABCMeta, abstractmethod
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import healpy as hp
import fitsio
import time

TZERO = None
def tprint(info):
    global TZERO
    if TZERO is None:
        TZERO = time.time()

    print('[%8ds] %s' % (time.time()-TZERO,info))


class Simulation:
    """
    A class which owns all the other catalog data 
    """
    
    def __init__(self, omega_m, omega_l, h, minz, maxz, area=0.0):
        
        self.omega_m = omega_m
        self.omega_l = omega_l
        self.h = h
        self.cosmo = FlatLambdaCDM(H0=100*h, Om0=omega_m)
        self.minz = minz
        self.maxz = maxz
        self.area = area
        self.calculate_volume(area)


    def calculate_volume(self,area):
        rmin = self.cosmo.comoving_distance(self.minz)
        rmax = self.cosmo.comoving_distance(self.maxz)
        self.volume = (self.area/41253)*(4/3*np.pi)*(rmax-rmin)**3
        
    def setGalaxyCatalog(self, catalog_type, filestruct, fieldmap=None,
                         input_LF=None, zbins=None, maskfile=None,
                         goodpix=1):
        """
        Fill in the galaxy catalog information for this simulation
        """

        if catalog_type == "BCC":
            self.galaxycatalog = BCCCatalog(self, filestruct, input_LF=input_LF,
                                            zbins=zbins, fieldmap=fieldmap,
                                            maskfile=maskfile, goodpix=goodpix)
        elif catalog_type == "S82Phot":
            self.galaxycatalog = S82PhotCatalog(self, None)
        elif catalog_type == "S82Spec":
            self.galaxycatalog = S82SpecCatalog(self, None)
        elif catalog_type == "DESGold":
            self.galaxycatalog = DESGoldCatalog(self, filestruct, maskfile=maskfile,
                                                goodpix=goodpix)

    def setHaloCatalog(self, catalog_type, filestruct):
        """
        Fill in the halo catalog information for this simulation
        """

    def setNBody(self, simulation_type):
        """
        Fill in NBody parameters and particles
        """

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

        #probably want Simulation to have map method
        #which knows how to combine different 
        #types of catalogs
        for f in mappables:
            if verbose:
                tprint('    {0}'.format(f))
            self.galaxycatalog.map(f)

        self.galaxycatalog.reduce()

        
class GalaxyCatalog:
    """
    Base class for galaxy catalogs
    """
    
    __metaclass__ = ABCMeta

    
    def __init__(self, filestruct, maskfile=None, goodpix=1):
        self.filestruct = filestruct
        self.maskfile = maskfile
        self.mask = None
        self.area = 0.0
        self.goodpix = goodpix

    @abstractmethod
    def parseFileStruct(self, filestruct):
        """
        Given a filestruct object, create map from parameters 
        we require to filepaths for easy access
        """

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

    def readFITSMappable(self, mappable, sortbyz=True):
        """
        For each element in the mappable, read in the fi
        specified in the fields dictionary and return
        a map from field names that the map function recognizes
        to the relevant data
        """

        mapunit = {}
        for f in mappable.keys():

            fieldmap = mappable[f]
            fields = []
            for val in fieldmap.values():
                if hasattr(val, '__iter__'):
                    fields.extend(val)
                else:
                    fields.extend([val])
                    
            data = fitsio.read(f, columns=fields)
                
            for mapkey in fieldmap.keys():
                mapunit[mapkey] = data[fieldmap[mapkey]]
                if hasattr(fieldmap[mapkey], '__iter__'):
                    dt = mapunit[mapkey].dtype[0]
                    ne = len(mapunit[mapkey])
                    nf = len(fieldmap[mapkey])
                    mapunit[mapkey] = mapunit[mapkey].view(dt).reshape((ne,nf))

        if sortbyz:
            if 'redshift' not in mapunit.keys():
                raise ValueError('There is no reshift field, cannot sort by redshift!')

            zidx = mapunit['redshift'].argsort()
            for mapkey in fieldmap.keys():
                mapunit[mapkey] = mapunit[mapkey][zidx]

        return mapunit

        
    @abstractmethod
    def map(self, mappable):
        """
        Do some operations on a mappable unit of the catalog
        """
        
    @abstractmethod
    def reduce(self):
        """
        Reduce the information produced by the map operations
        """

    def setFieldMap(self, fieldmap):
        self.fieldmap = fieldmap

    def calculateArea(self, pixels, nside):
        """
        Calculate the area in the given pixels provided a mask
        that is pixelated with an nside greater than that of 
        the catalog
        """

        area = np.zeros(len(pixels))
        if self.mask==None:
            self.mask, self.maskhdr = hp.read_map(self.maskfile,h=True)
            self.maskhdr = dict(self.maskhdr)
        
        #our map should be finer than our file pixelation
        assert(nside<self.maskhdr['NSIDE']) 

        udmap = hp.ud_grade(np.arange(12*nside**2),self.maskhdr['NSIDE'])
        pixarea = hp.nside2pixarea(self.maskhdr['NSIDE'],degrees=True)
        for i,p in enumerate(pixels):
            pm, = np.where(udmap==p)
            print('ngood: {0}'.format(len(self.mask[pm][self.mask[pm]>=self.goodpix])))
            print('nold: {0}'.format(len(udmap[pm][udmap[pm]>=self.goodpix])))
            print('mask: {0}'.format(self.mask))
            print('goodpix: {0}'.format(self.goodpix))
            area[i] = pixarea*len(self.mask[pm][self.mask[pm]>=self.goodpix])
            
        return area

    def getArea(self):
        
        if self.mask==None:
            return self.sim.area
        else:
            return self.area


class BCCCatalog(GalaxyCatalog):
    """
    BCC style ADDGALS catalog
    """

    def __init__(self, simulation, filestruct, fieldmap=None, 
                 input_LF=None, nside=8, zbins=None, maskfile=None,
                 goodpix=1):
        GalaxyCatalog.__init__(self, filestruct, maskfile=maskfile, goodpix=goodpix)
        self.sim = simulation
        self.input_LF = input_LF
        self.filestruct = filestruct
        self.parseFileStruct(filestruct)
        self.metrics = [LuminosityFunction(self.sim, zbins=zbins), 
                        MagCounts(self.sim, zbins=zbins), 
                        ColorColor(self.sim, zbins=zbins)]
        self.nside = nside

        if fieldmap==None:
            self.fieldmap = {'luminosity':OrderedDict([('AMAG',['truth'])]),
                             'appmag':OrderedDict([('MAG_G',['obs']), ('MAG_R',['obs']),
                                                   ('MAG_I',['obs']), ('MAG_Z',['obs']),
                                                   ('MAG_Y',['obs'])]),
                             'redshift':OrderedDict([('Z',['truth'])])}
            self.hasz = True
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

        if self.maskfile!=None:
            pix = self.pixelVal(mappable)
            a = self.calculateArea([pix],self.nside)
            self.area += a[0]

        for m in self.metrics:
            m.map(mapunit)
        
    def reduce(self):
        """
        Reduce the information produced by the map operations
        """
        for m in self.metrics:
            m.reduce()



class S82SpecCatalog(GalaxyCatalog):
    """
    SDSS DR6 stripe82 photometric galaxy catalog (for mag/count, color comparisons)
    """

    def __init__(self, simulation, filestruct, input_LF=None, nside=8):
        GalaxyCatalog.__init__(self, filestruct)
        self.sim = simulation
        self.input_LF = input_LF
        self.parseFileStruct(None)
        self.metrics = [LuminosityFunction(self.sim)]
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

    def __init__(self, simulation, filestruct, input_LF=None, nside=8):
        GalaxyCatalog.__init__(self, filestruct)
        self.sim = simulation
        self.input_LF = input_LF
        self.parseFileStruct(None)
        self.metrics = [MagCounts(self.sim), ColorColor(self.sim)]
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

    def __init__(self, simulation, filestruct, nside=8, maskfile=None, goodpix=1):
        GalaxyCatalog.__init__(self, filestruct,maskfile=maskfile,goodpix=goodpix)
        self.necessaries = ['modest']
        self.sim = simulation
        self.parseFileStruct(filestruct)
        self.nside = nside
        self.metrics = [MagCounts(self.sim, zbins=None), 
                        ColorColor(self.sim, zbins=None)] 
        self.fieldmap = {'appmag':OrderedDict([('FLUX_AUTO_G',['auto']), 
                                               ('FLUX_AUTO_R',['auto']),
                                               ('FLUX_AUTO_I',['auto']), 
                                               ('FLUX_AUTO_Z',['auto'])]),
                         'modest':OrderedDict([('MODEST_CLASS',['basic'])])}

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
        """                                                                                    Do some operations on a mappable unit of the catalog                                   """

        mapunit = self.readFITSMappable(mappable, sortbyz=False)
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

        midx = mapunit['modest']==1

        for mapkey in mapunit.keys():
            mapunit[mapkey] = mapunit[mapkey][midx]

        return mapunit
        

    def reduce(self):
        """                                                                                                                                                                          
        Reduce the information produced by the map operations                                                                                                                        
        """
        for m in self.metrics:
            m.reduce()

