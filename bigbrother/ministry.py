from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
from astropy.cosmology import FlatLambdaCDM
from .galaxy import GalaxyCatalog, BCCCatalog, S82PhotCatalog, S82SpecCatalog, DESGoldCatalog
from .halo import HaloCatalog, BCCHaloCatalog
import numpy as np
import healpy as hp
import helpers
import fitsio
import time

TZERO = None
def tprint(info):
    global TZERO
    if TZERO is None:
        TZERO = time.time()

    print('[%8ds] %s' % (time.time()-TZERO,info))


class Ministry:
    """
    A class which owns all the other catalog data 
    """
    
    def __init__(self, omega_m, omega_l, h, minz, maxz, area=0.0,
                 boxsize=None):
        """
        Initialize a ministry object
        
        Arguments
        ---------
        omega_m : float
            Matter density parameter now
        omega_l : float
            Lambda density parameter now
        h : float
            Dimensionless hubble constant
        minz : float
            Minimum redshift
        maxz : float
            Maximum redshift
        area : float, optional
            The area spanned by all catalogs held
        """

        self.omega_m = omega_m
        self.omega_l = omega_l
        self.h = h
        self.cosmo = FlatLambdaCDM(H0=100*h, Om0=omega_m)
        self.minz = minz
        self.maxz = maxz
        if minz!=maxz:
            self.lightcone = True
        else:
            self.lightcone = False
            self.boxsize = boxsize

        self.area = area
        self.volume = self.calculate_volume(area,self.minz,self.maxz)


    def calculate_volume(self,area,minz,maxz):
        if self.lightcone:
            rmin = self.cosmo.comoving_distance(minz)*self.h
            rmax = self.cosmo.comoving_distance(maxz)*self.h
            return (area/41253)*(4/3*np.pi)*(rmax**3-rmin**3)
        else:
            return (self.boxsize*self.h)**3
        
        
    def setGalaxyCatalog(self, catalog_type, filestruct, fieldmap=None,
                         zbins=None, maskfile=None,
                         goodpix=1):
        """
        Fill in the galaxy catalog information
        """

        if catalog_type == "BCC":
            self.galaxycatalog = BCCCatalog(self, filestruct,  zbins=zbins, 
                                            fieldmap=fieldmap, maskfile=maskfile,
                                            goodpix=goodpix)
        elif catalog_type == "S82Phot":
            self.galaxycatalog = S82PhotCatalog(self, None)
        elif catalog_type == "S82Spec":
            self.galaxycatalog = S82SpecCatalog(self, None)
        elif catalog_type == "DESGold":
            self.galaxycatalog = DESGoldCatalog(self, filestruct, maskfile=maskfile, 
                                                goodpix=goodpix)

    def setHaloCatalog(self, catalog_type, filestruct, fieldmap=None,
                       zbins=None, maskfile=None, goodpix=1):
        """
        Fill in the halo catalog information
        """

        if catalog_type == "BCC":
            self.halocatalog = BCCHaloCatalog(self, filestruct, zbins=zbins, 
                                              fieldmap=fieldmap, maskfile=maskfile, 
                                              goodpix=goodpix)

    def getMetricDependencies(self, metric):
        
        fieldmap = {}
        valid = {}
        for ctype in metric.catalog_type:
            if not hasattr(self, ctype):
                raise ValueError("This Ministry does not have"
                                 "a catalog of type {0} as required"
                                 "by {1}".format(ctype, metric.__class__.__name__))
            else:
                cat = getattr(self, ctype)

            #go through metric dependencies, checking if 
            #this catalog satisfies them. If it does, create 
            #the map from the metric dependency to the catalog
            #fields as specified by the catalog's field map

            for mapkey in metric.mapkeys:
                if mapkey not in valid.keys():
                    valid[mapkey] = False

                if mapkey in cat.fieldmap.keys():
                    fileinfo = cat.fieldmap[mapkey]
                else:
                    continue

                for field in fileinfo.keys():
                    filetypes = fileinfo[field]
                    for ft in filetypes:
                        if ft in cat.filetypes:
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
                                
                            valid[mapkey] = True

        notavail = []
        for key in valid.keys():
            if not valid[key]:
                notavail.append(key)
        
        if len(notavail)>0:
            raise Exception("Mapkeys {0} are not available!".format(notavail))

        return fieldmap


    def compFieldMaps(self, fm1, fm2):
        
        ft1 = set(fm1.keys())
        ft2 = set(fm2.keys())

        if ft1.issubset(ft2):
            temp = ft1
            ft1  = ft2
            ft2  = temp
        elif not ft2.issubset(ft1):
            return False

        for ft in ft2:
            mk1 = set(fm1[ft].keys())
            mk2 = set(fm2[ft].keys())

            if mk1.issubset(mk2) or mk1.issubset(mk2):
                imk = mk1.intersection(mk2)
                for k in imk:
                    if fm1[ft][k]!=fm2[ft][k]:
                        return False

            elif len(mk1.union(mk2))==(max(len(mk1),len(mk2))+1):
                imk = mk1.intersection(mk2)
                for k in imk:
                    if fm1[ft][k]!=fm2[ft][k]:
                        return False

            else:
                return False

        return True
        
    def genMetricGroups(self, metrics, fieldmaps):
        
        fms = set(zip(fieldmaps, metrics))
        graph = {f:fms.difference(f) for f in fms}
        
        #while merges are still necessary, keep going
        while True:
            nomerge = True
            merge = []
            #iterate through nodes, figuring out 
            #which nodes to merge
            for i, node in enumerate(graph):
                for edge in graph[node]:
                    if self.compFieldMaps(node[0], edge[0]):
                        merge.append([node,edge])
                        nomerge=False
            if nomerge:
                break
            else:
                nfm = []
                merged = []
                for m in merge:
                    #check to avoid duplication
                    if m not in merged:
                        fms = fms.difference(set(m))
                        nfm.append(self.combineFieldMaps(*m))
                        merged.append(m)
                    
                fms = fms.union(set(nfm))
                graph = {f:fms.difference(f) for f in fms}

        return fms

    def combineFieldMaps(fm1, fm2):
        """
        Combine field maps associated with two different metrics.
        The field maps must be compatible in the sense that 
        comparing them using compFieldMaps returns True
        """
        #make sure compatible
        m1 = fm1[1]
        m2 = fm2[1]
        fm1 = fm1[0]
        fm2 = fm2[0]

        if not self.compFieldMaps(fm1, fm2):
            raise ValueError("Field maps are not compatible!")

        cfm = {}
        ft1 = set(fm1.keys())
        ft2 = set(fm2.keys())

        if ft1.issubset(ft2):
            temp = ft1
            ft1  = ft2
            ft2  = temp

        for ft in ft2:
            mk1 = set(fm1[ft].keys())
            mk2 = set(fm2[ft].keys())
            cfm[ft] = mk1.union(mk2)

        return [cfm, [m1, m2]]

        
    def genMetricGroups(self, metrics):
        """
        Given a set of metrics, generate a list of mappables
        which can be fed into map functions
        """
        mappables = []

        mfieldmaps = [self.getMetricDependencies(m) for m in metrics]
        mgroups = self.genMetricGroups(metrics, mfieldmaps)
        self.metric_groups = mgroups


    def associateFileStructs(self):
        """
        Given files structures for different catalogs,
        use some rule to be determined to group sets
        of files from the different catalogs
        """
        pass


    def genMappables(self, fieldmap):
        """
        Given a field map, generate a mappble which can be 
        passed to a metric
        """
        
        


    def validate(self, metrics=None, verbose=False):
        """
        Run all validation metrics by iterating over only the files we
        need at a given time, mapping catalogs to relevant statistics
        which are reduced at the end of the iteration into observables 
        that we care about
        """
        self.genMetricGroups()
        for mg in self.metric_groups:
            ms = mg[0]
            fm = mg[1]
            for mappable in self.genMappables(fm):
                for m in ms:
                    m.map(mappable)

            for m in ms:
                m.reduce()

