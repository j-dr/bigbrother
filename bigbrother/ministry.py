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
            self.galaxycatalog = DESGoldCatalog(self, filestruct, maskfile=maskfile,                                                goodpix=goodpix)

    def setHaloCatalog(self, catalog_type, filestruct):
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
        

    def genMetricGroups(self, fieldmaps):
        
        fms = set(fieldmaps)
        graph = {f:fms.difference(f) for f in fms}
        
        #while merges are still necessary, keep going
        while True:
            nomerge = True
            merge = []
            #iterate through nodes, figuring out 
            #which nodes to merge
            for node in graph:
                for edge in graph[node]:
                    if self.compFieldMaps(node, edge):
                        merge.append([node,edge])
                        nomerge=False

            if nomerge:
                break
            else:
                nfm = []
                for m in merge:
                    fms = fms.difference(m)
                    nfm.append(self.combineFieldMaps(m))
                    
                fms = fms.union(set(nfm))
                graph = {f:fms.difference(f) for f in fms}

        return fms

        
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

        #probably want Ministry to have map method
        #which knows how to combine different 
        #types of catalogs
        for f in mappables:
            if verbose:
                tprint('    {0}'.format(f))
            self.galaxycatalog.map(f)

        self.galaxycatalog.reduce()
