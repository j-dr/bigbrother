from __future__ import print_function, division
from collections import OrderedDict
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

class Mappable(object):
    """
    A tree which contains information on the order in which files should be read
    """
    
    def __init__(self, name=None, dtype=None, children=[], childtype=None):
        
        self.name = name
        self.dtype = dtype
        self.children = children
        self.data = None

        

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

        #get file types required
        ft1 = set(fm1.keys())
        ft2 = set(fm2.keys())

        #if first set is subset of second, switch order
        #if disjoint, not compatible
        if ft1.issubset(ft2):
            temp = ft1
            ft1  = ft2
            ft2  = temp
        elif not ft2.issubset(ft1):
            return False

        #iterate over filetypes in intersection
        for ft in ft2:
            mk1 = set(fm1[ft].keys())
            mk2 = set(fm2[ft].keys())

            #check if sets of keys contained in one another
            if mk1.issubset(mk2) or mk2.issubset(mk1):
                imk = mk1.intersection(mk2)
                #if fields that keys map to don't correspond
                #fms are not compatible
                for k in imk:
                    if fm1[ft][k]!=fm2[ft][k]:
                        return False

            #if only one addional element, also check for compatibility
            elif len(mk1.union(mk2))==(max(len(mk1),len(mk2))+1):
                imk = mk1.intersection(mk2)
                for k in imk:
                    if fm1[ft][k]!=fm2[ft][k]:
                        return False

            else:
                return False

        return True

    def compAssoc(self, mg0, mg1):
        """
        Check if metric groups have compatible file
        association schemas
        """
        if hasattr(mg0, '__iter__'):
            m0 = mg0[0]
        else:
            m0 = mg0
            
        if hasattr(mg1, '__iter__'):
            m1 = mg1[0]
        else:
            m1 = mg1

        if m0.aschema == m1.aschema:
            return True
        else:
            return False
        
    def genMetricGroups(self, metrics):
        """
        Given a list of metrics, group them together based
        on what types of data they require so that 
        groups of metrics can be run on the same mappables

        inputs
        ------
        metrics -- list
        A list of Metric objects to be grouped together
        
        outputs
        ------
        fms -- list
        First element of the list is fieldmaps for each group
        of metrics. The second element are the groups of metrics
        themselves, formatted in lists.
        """
        
        fieldmaps = [self.getMetricDependencies(m) for m in metrics]
        fms = zip(fieldmaps, metrics)
        nodes = range(len(fms))
        snodes = set(nodes)
        graph = {f:snodes.difference(set([f])) for f in snodes}
        
        i = 0
        while i < len(nodes)-1:
            #iterate through nodes, figuring out 
            #which nodes to merge
            node = nodes[i]
            nomerge = True
            for edge in graph[node]:
                if self.compFieldMaps(fms[node][0], fms[edge][0]) &  self.compAssoc(fms[node][1], fms[edge][1]):
                    nomerge=False
                    m = [node, edge]
                    mg0 = fms[node]
                    mg1 = fms[edge]
                    
                    #store new metric group in nfm to be added
                    #to graph later
                    nfm = self.combineFieldMaps(mg0, mg1)
                    
                    #pop the one with the lower index first so 
                    #we know where the second element is afterwards
                    fms.pop(min(m))
                    nodes.pop(min(m))
                    fms.pop(max(m)-1)
                    nodes.pop(max(m)-1)
                    
                    fms.append(nfm)
                    
                    #reconstruct graph with merged nodes 
                    nodes = range(len(fms))
                    snodes = set(nodes)
                    graph = {f:snodes.difference(set([f])) for f in snodes}
                    break

            if nomerge:
                i+=1

        return fms

    def combineFieldMaps(self, fm1, fm2):
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
            umk = mk1.union(mk2)
            f = []
            for k in umk:
                try:
                    f.append(fm1[ft][k])
                except:
                    f.append(fm2[ft][k])

            cfm[ft] = OrderedDict(zip(umk, f))

        if hasattr(m1,'__iter__'):
            m1.extend(list(m2))
            m = m1
        elif hasattr(m2,'__iter__'):
            m2.extend(list(m1))
            m = m2
        else:
            m = [m1,m2]

        return [cfm, m]


    def singleTypeMappable(self, fieldmap, fs):
        """
        If only working with one type of catalog,
        can assume that the length of all file types
        in file struct are the same
        """

        filetypes = fieldmap.keys()
        mappables = []


        #need to put filetypes with redshifts in
        #them first
        zft = []
        nzft = []
        for ft in filetypes:
            if 'redshift' in fieldmap[ft]:
                zft.append(ft)
            else:
                nzft.append(ft)
        
        filetypes = zft
        filetypes.extend(nzft)

        #Create mappables out of filestruct and fieldmaps
        for i in range(len(fs[filetypes[0]])):

            for i, ft in enumerate(filetypes):
                if i==0:
                    root = Mappable(fs[ft][i], ft)
                    last = root
                else:
                    node = Mappable(fs[ft][i], ft)
                    last.children.append(fs[ft])
                    last = node

            mappables.append(root)

        return mappables
         
    def galaxyGalaxyMappable(self, fieldmap):

        raise NotImplementedError

    def haloHaloMappable(self, fieldmap):

        raise NotImplementedError

    def haloGalaxyMappable(self, fieldmap, nside=8):

        #combine file structures for halo 
        #and galaxy catalogs
        fs  = self.galaxycatalog.filestruct
        hfs = self.halocatalog.filestruct
        fs.update(hfs)

        gft = [ft for ft in fieldmap.keys() if ft 
               in self.galaxycatalog.filestruct.keys()]
        hft = [ft for ft in fieldmap.keys() if ft 
               in self.halocatalog.filestruct.keys()]
        
        #get a filetype necessary for halo catalog
        hkey = hft[0]

        gfpix = self.galaxycatalog.getFilePixels(nside)
        hfpix = self.halocatalog.getFilePixels(nside)

        filetypes = fieldmap.keys()
        mappables = []

        #Create mappables out of filestruct and fieldmaps
        for i in range(len(hfs[hkey])):
            root = Mappable(hfs[hkey][i], hkey)
            last = root
            for ft in hft:
                if ft==hkey:    continue
                mp = Mappable(hfs[ft][i], ft)
                last.children.append(mp)

            hlast = last
            gidx = self.getIntersection('halogalaxy',hfpix[i], gfpix)

            for idx in gidx:
                last = hlast

                for ft in gft:
                    mp = Mappable(gfs[ft][idx], ft)
                    last.children.append(mp)
                    last = mp

            mappables.append(mappable)

        return mappables


    def getIntersection(self, aschema, p1, p2, nside=8, nest=True):
        """
        Given an association schema, determine which files intersect
        """
        
        idx = []

        if aschema=='halogalaxy':
            for p in p1:
                pix = [p]
                nbrs = hp.get_all_neighbours(nside, p, nest=nest)
                pix.append(nbrs)
                pix = np.array(pix)
                
                #iterate through pixel lists of secondary file
                #type. If any pixels in these files are 
                #neighbors of the primary pixel then we 
                #need to read this file
                for i, ip in enumerate(p2):
                    fidx = np.in1d(ip, pix)
                    if fidx.any():
                        idx.append(i)

        return np.array(set(idx))

    def genMappables(self, mgroup):
        """
        Given a fieldmap and an association
        schema create mappables for a metric
        group from file structures
        """

        fm = mgroup[0]
        m  = mgroup[1]

        if hasattr(m, '__iter__'):
            aschema = m[0].aschema
        else:
            aschema = m.aschema

        if aschema == 'galaxyonly':
            return self.singleTypeMappable(fm, self.galaxycatalog.filestruct)
        elif aschema == 'haloonly':
            return self.singleTypeMappable(fm, self.halocatalog.filestruct)
        elif aschema == 'galaxygalaxy':
            return self.galaxyGalaxyMappable(fm)
        elif aschema == 'halohalo':
            return self.haloHaloMappable(fm)
        elif aschema == 'halogalaxy':
            return self.haloGalaxyMappable(fm)


    def readMappable(self, mappable, fieldmap):

        if mappable.dtype[0]=='h':
            mappable.data = self.halocatalog.readMappable(mappable, fieldmap)
        elif mappable.dtype[0]=='g':
            mappable.data = self.galaxycatalog.readMappable(mappable, fieldmap)

        if len(mappable.children)>0:
            for child in mappable.children:
                self.readMappable(child, fieldmap)

        return mappable

    def treeToDict(self, mapunit):
        """
        Most general form of map unit is a tree. Most
        metrics don't require this. If the schema we
        are working with doesn't, turn the tree
        into the old dict structure
        """
        
        mu = {}
        
        while len(mapunit.children)>0:
            if len(mapunit.children)>1:
                raise ValueError("mapunit has more than one branch!")

            for key in mapunit.data.keys():
                if key in mu.keys():
                    shp0 = mu[key].shape
                    shp1 = mapunit.data[key].shape
                    
                    if os[0]!=ns[0]:
                        raise ValueError("Sizes of data for same mapkey {0} do not match!".format(key))

                    nshp = [shp0[0], shp0[1]+shp1[1]]
                    
                    d = np.ndarray(nshp)
                    d[:,:shp0[1]] = mu[key]
                    d[:,shp0[1]:] = mapunit.data[key]
                else:
                    mu[key] = mapunit.data[key]

            mapunit = mapunit.children[0]

        for key in mapunit.data.keys():
            if key in mu.keys():
                shp0 = mu[key].shape
                shp1 = mapunit.data[key].shape
                
                if os[0]!=ns[0]:
                    raise ValueError("Sizes of data for same mapkey {0} do not match!".format(key))
                
                nshp = [shp0[0], shp0[1]+shp1[1]]
                
                d = np.ndarray(nshp)
                d[:,:shp0[1]] = mu[key]
                d[:,shp0[1]:] = mapunit.data[key]
            else:
                mu[key] = mapunit.data[key]
            
        return mu

    def sortByZ(self, mappable, fieldmap, idx):
        """
        Sort a mappable by redshift for each galaxy type
        """
        
        if 'redshift' in fieldmap[mappable.dtype].keys():
            idx = mappable.data['redshift'].argsort()


        dk = mappable.data.keys()
        if len(idx)==len(mappable.data[dk[0]]):
            for k in dk:
                mappable.data[k] = mappable.data[k][idx]
                
        
        if len(mappable.children)>0:
            for child in mappable.children:
                self.sortByZ(child, fieldmap, idx)
        

    def validate(self, metrics=None, verbose=False):
        """
        Run all validation metrics by iterating over only the files we
        need at a given time, mapping catalogs to relevant statistics
        which are reduced at the end of the iteration into observables 
        that we care about
        """
        
        if metrics==None:
            metrics = self.metrics
        
        self.metric_groups = self.genMetricGroups(metrics)
        
        for mg in self.metric_groups:
            sbz = False
            ms  = mg[1]
            fm  = mg[0]
            for ft in fm.keys():
                if 'redshift' in fm[ft].keys():
                    sbz = True

            for mappable in self.genMappables(mg):
                mapunit = self.readMappable(mappable, fm)

                if sbz:
                    self.sortByZ(mapunit, fm, [])

                if 'only' in ms[0].aschema:
                    mapunit = self.treeToDict(mapunit)

                for m in ms:
                    print('*****{0}*****'.format(m.__class__.__name__))
                    m.map(mapunit)
                    
                del mapunit

            for m in ms:
                m.reduce()

