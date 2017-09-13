from __future__ import print_function, division
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
from astropy.cosmology import FlatLambdaCDM
from .basecatalog import PlaceHolder
from .galaxy import GalaxyCatalog, BCCCatalog, S82PhotCatalog, S82SpecCatalog, DESGoldCatalog
from .halo import HaloCatalog, BCCHaloCatalog
from .particle import LGadgetSnapshot

from copy import copy, deepcopy
import numpy as np
import healpy as hp
import fitsio
import units
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

    def __init__(self, name, dtype, children=None, childtype=None, jtype=None,
                  gnside=None, nbox=None, grp=None):

        self.name = name
        self.dtype = dtype
        if children==None:
            self.children = []
        else:
            self.children = children

        self.data = None
        self.jtype = jtype
        self.gnside = gnside
        self.nbox = nbox
        self.grp = grp

    def recursive_delete(self):

        print(self.children)
        for k in self.data.keys():
            del self.data[k]

        if len(self.children)==0:
            return
        else:
            for i in range(len(self.children)):
                self.children[i].recursive_delete()
        
class Ministry:
    """
    A class which owns all the other catalog data
    """

    _known_galaxy_catalog_types = ['BCC', 'S82Phot', 'S82Spec', 'DESGold', 'PlaceHolder']
    _known_halo_catalog_types   = ['BCC', 'PlaceHolder']
    _known_particle_catalog_types = ['LGadgetSnapshot']

    def __init__(self, omega_m, omega_l, h, minz, maxz, area=None,
                 boxsize=None, one_metric_group=False, parallel=False,
                 ministry_name=None, maskfile=None,maskcomp=None, maskval=None):
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

        self.ministry_name = ministry_name
        self.omega_m = omega_m
        self.omega_l = omega_l
        self.h = h
        self.cosmo = FlatLambdaCDM(H0=100*h, Om0=omega_m)
        self.minz = minz
        self.maxz = maxz
        self.one_metric_group = one_metric_group
        self.parallel = parallel
        self.galaxycatalog = None
        self.halocatalog = None
        self.particlecatalog = None
        self.ministry_name = ministry_name

        if area is None:
            self.area = 0.0
        else:
            self.area = area

        if minz!=maxz:
            self.lightcone = True
        else:
            self.lightcone = False
            self.boxsize = boxsize

        self.volume = self.calculate_volume(self.area,self.minz,self.maxz)
        self.maskfile = maskfile
        self.maskcomp = maskcomp
        self.maskval  = maskval
        self.mask = None
        

    def calculate_volume(self,area,minz,maxz):
        if self.lightcone:
            rmin = self.cosmo.comoving_distance(minz)*self.h
            rmax = self.cosmo.comoving_distance(maxz)*self.h
            return (area/41253)*(4/3*np.pi)*(rmax**3-rmin**3)
        else:
            return (self.boxsize*self.h)**3


    def setGalaxyCatalog(self, catalog_type, filestruct, **kwargs):

        """
        Fill in the galaxy catalog information
        """

        if catalog_type == "BCC":
            self.galaxycatalog = BCCCatalog(self, filestruct, **kwargs)
        elif catalog_type == "S82Phot":
            self.galaxycatalog = S82PhotCatalog(self, None)
        elif catalog_type == "S82Spec":
            self.galaxycatalog = S82SpecCatalog(self, None)
        elif catalog_type == "DESGold":
            self.galaxycatalog = DESGoldCatalog(self, filestruct, **kwargs)

        elif catalog_type == "PlaceHolder":
            self.galaxycatalog = PlaceHolder(self, None)

    def setHaloCatalog(self, catalog_type, filestruct, goodpix=1, **kwargs):
        """
        Fill in the halo catalog information
        """

        if catalog_type == "BCC":
            self.halocatalog = BCCHaloCatalog(self, filestruct, goodpix=goodpix, **kwargs)

        elif catalog_type == "PlaceHolder":
            self.galaxycatalog = PlaceHolder(self, None)

    def setParticleCatalog(self, catalog_type, filestruct, **kwargs):
        
        if catalog_type == "LGadgetSnapshot":
            self.particlecatalog = LGadgetSnapshot(self, filestruct, **kwargs)
        elif catalog_type == "PlaceHolder":
            self.particlecatalog = PlaceHolder(self, None)

    def getMetricDependencies(self, metric):

        fieldmap = {}
        valid = {}
        for ctype in metric.catalog_type:
            if (getattr(self, ctype) is None):
                raise ValueError("This Ministry does not have "
                                 "a catalog of type {0} as required "
                                 "by {1}".format(ctype, metric.__class__.__name__))
            else:
                cat = getattr(self, ctype)

            #go through metric dependencies, checking if
            #this catalog satisfies them. If it does, create
            #the map from the metric dependency to the catalog
            #fields as specified by the catalog's field map

            mk  = copy(metric.mapkeys)
            mk.extend(cat.necessaries)

            if cat.jtype == 'subbox':
                mk.extend(['px','py','pz'])
            elif cat.jtype == 'healpix':
                mk.extend(['polar_ang', 'azim_ang'])

            mk = np.unique(mk)

            for mapkey in mk:
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
            raise Exception("Mapkeys {0} are not available. Required by {1}!".format(notavail, metric.__class__.__name__))

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

    def compUnits(self, mg0, mg1):
        """
        Check if metrics have compatible unit
        requirements
        """
        #ugly loop, but small data structure so okay
        for m0 in mg0:
            for m1 in mg1:
                for k0 in m0.unitmap.keys():
                    for k1 in m1.unitmap.keys():
                        if k0 == k1:
                            if m0.unitmap[k0] != m1.unitmap[k1]:
                                return False
        return True

    def compJackknife(self, mg0, mg1):
        """
        Check if metrics have compatible jackknife
        requirements
        """
        for m0 in mg0:
            for m1 in mg1:
                if (m0.jtype is not None) and (m0.jtype != m1.jtype):
                    return False

        return True

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
        metrics = [[m] for m in metrics]
        fms = zip(fieldmaps, metrics)

        if self.one_metric_group:
            fm = fms.pop()
            while len(fms)>0:
                fm = self.combineFieldMaps(fm, fms.pop())

            return [fm]

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
                if (self.compFieldMaps(fms[node][0], fms[edge][0]) & self.compAssoc(fms[node][1], fms[edge][1]) & self.compUnits(fms[node][1], fms[edge][1]) &
                  self.compJackknife(fms[node][1], fms[edge][1])):
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

        #if not self.compFieldMaps(fm1[0], fm2[0]):
        #    raise ValueError("Field maps are not compatible!")

        cfm = {}
        ft1 = set(fm1[0].keys())
        ft2 = set(fm2[0].keys())

        if ft1.issubset(ft2):
            temp = fm1
            fm1  = fm2
            fm2  = temp

        ft1 = set(fm1[0].keys())
        ft2 = set(fm2[0].keys())
        m1 = fm1[1]
        m2 = fm2[1]
        fm1 = fm1[0]
        fm2 = fm2[0]

        ift = ft1-ft2

        for ft in ft2:
            if ft not in fm1.keys():
                mk1 = set([])
            else:
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

        for ft in ift:
            cfm[ft] = fm1[ft]

        if hasattr(m1,'__iter__'):
            m1.extend(list(m2))
            m = m1
        elif hasattr(m2,'__iter__'):
            m2.append(m1)
            m = m2
        else:
            m = [m1,m2]

        return [cfm, m]

    def singleTypeMappable(self, fieldmap, fs, ct, jtype, override=False):
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
        

        if (jtype is not None) & (not override):
            cat = getattr(self, ct)
            g, fgroups = cat.groupFiles()

            jt = jtype
            nb = cat.nbox
            gn = cat.groupnside

        elif not override:
            fgroups = [np.arange(len(fs[filetypes[0]]))]
            g       = [0]
            jt = None
            nb = 0
            gn = 0

        else:
            for i in range(len(fs[filetypes[0]])):

                for j, ft in enumerate(filetypes):
                    if j==0:
                        root = Mappable(fs[ft][i], ft)
                        last = root
                    else:
                        node = Mappable(fs[ft][i], ft)
                        last.children.append(node)
                        last = node
                    
                mappables.append(root)

            return mappables            

        #Create mappables out of filestruct and fieldmaps
        for i, fg in enumerate(fgroups):
            for fc, j in enumerate(fg):
                for k, ft in enumerate(filetypes):
                    if (fc==0) & (k==0):
                        root = Mappable(fs[ft][j], ft, jtype=jt,
                                      gnside=gn, nbox=nb, grp=g[i])
                        last = root
                    else:
                        node = Mappable(fs[ft][j], ft, jtype=jt,
                                      gnside=gn, nbox=nb, grp=g[i])
                        last.children.append(node)
                        last = node

            mappables.append(root)

        return mappables


    def galaxyGalaxyMappable(self, fieldmap):

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

        g, fgroups = self.galaxycatalog.groupFiles()
        jt = self.galaxycatalog.jtype
        nb = self.galaxycatalog.nbox
        gn = self.galaxycatalog.groupnside

        fs = self.galaxycatalog.filestruct

        #Create mappables out of filestruct and fieldmaps
        for i, fg in enumerate(fgroups):
            for fc, j in enumerate(fg):
                for k, ft in enumerate(filetypes):
                    if (fc==0) & (k==0):
                        root = Mappable(fs[ft][j], ft, jtype=jt,
                                      gnside=gn, nbox=nb, grp=g[i])
                        last = root
                    else:
                        node = Mappable(fs[ft][j], ft, jtype=jt,
                                      gnside=gn, nbox=nb, grp=g[i])
                        last.children.append(node)
                        last = node

            mappables.append(root)

        return mappables


    def haloHaloMappable(self, fieldmap):
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

        g, fgroups = self.halocatalog.groupFiles()
        jt = self.halocatalog.jtype
        nb = self.halocatalog.nbox
        gn = self.halocatalog.groupnside

        fs = self.halocatalog.filestruct

        #Create mappables out of filestruct and fieldmaps
        for i, fg in enumerate(fgroups):
            for fc, j in enumerate(fg):
                for k, ft in enumerate(filetypes):
                    if (fc==0) & (k==0):
                        root = Mappable(fs[ft][j], ft, jtype=jt,
                                      gnside=gn, nbox=nb, grp=g[i])
                        last = root
                    else:
                        node = Mappable(fs[ft][j], ft, jtype=jt,
                                      gnside=gn, nbox=nb, grp=g[i])
                        last.children.append(node)
                        last = node

            mappables.append(root)

        return mappables

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

    def particleParticleMappable(self, fieldmap):

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

        g, fgroups = self.particlecatalog.groupFiles()
        jt = self.particlecatalog.jtype
        nb = self.particlecatalog.nbox
        gn = self.particlecatalog.groupnside

        fs = self.particlecatalog.filestruct

        #Create mappables out of filestruct and fieldmaps
        for i, fg in enumerate(fgroups):
            for fc, j in enumerate(fg):
                for k, ft in enumerate(filetypes):
                    if (fc==0) & (k==0):
                        root = Mappable(fs[ft][j], ft, jtype=jt,
                                      gnside=gn, nbox=nb, grp=g[i])
                        last = root
                    else:
                        node = Mappable(fs[ft][j], ft, jtype=jt,
                                      gnside=gn, nbox=nb, grp=g[i])
                        last.children.append(node)
                        last = node

            mappables.append(root)

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

    def genMappables(self, mgroup, override=False):
        """
        Given a fieldmap and an association
        schema create mappables for a metric
        group from file structures
        """

        fm = mgroup[0]
        m  = mgroup[1]

        if hasattr(m, '__iter__'):
            aschema = m[0].aschema
            ct      = m[0].catalog_type[0]
        else:
            aschema = m.aschema
            ct      = m.catalog_type[0]

        if aschema == 'galaxyonly':
            return self.singleTypeMappable(fm, self.galaxycatalog.filestruct, ct, m[0].jtype, override=override)
        elif aschema == 'haloonly':
            return self.singleTypeMappable(fm, self.halocatalog.filestruct, ct, m[0].jtype, override=override)
        elif aschema == 'singleonly':
            if ct == 'galaxycatalog':
                return self.singleTypeMappable(fm, self.galaxycatalog.filestruct, ct, m[0].jtype, override=override)
            if ct == 'halocatalog':
                return self.singleTypeMappable(fm, self.halocatalog.filestruct, ct, m[0].jtype, override=override)
            if ct == 'particlecatalog':
                return self.singleTypeMappable(fm, self.particlecatalog.filestruct, ct, m[0].jtype, override=override)
        elif aschema == 'galaxygalaxy':
            return self.galaxyGalaxyMappable(fm)
        elif aschema == 'halohalo':
            return self.haloHaloMappable(fm)
        elif aschema == 'halogalaxy':
            return self.haloGalaxyMappable(fm)
        elif aschema == 'particleparticle':
            print('particleparticle')
            return self.particleParticleMappable(fm)


    def readMappable(self, mappable, fieldmap):
        print('Reading {}'.format(mappable.name))
        if (self.halocatalog is not None) and (mappable.dtype in self.halocatalog.filetypes):
            mappable.data = self.halocatalog.readMappable(mappable, fieldmap)
        elif (self.galaxycatalog is not None) and (mappable.dtype in self.galaxycatalog.filetypes):
            mappable.data = self.galaxycatalog.readMappable(mappable, fieldmap)
        elif (self.particlecatalog is not None) and (mappable.dtype in self.particlecatalog.filetypes):
            mappable.data = self.particlecatalog.readMappable(mappable, fieldmap)

        if len(mappable.children)>0:
            for child in mappable.children:
                self.readMappable(child, fieldmap)

        return mappable

    def dcListToDict(self, mapunit):
        """
        Association schemas with double catalog types (e.g. galaxygalaxy)
        result in mappables which are lists. This method compresses
        such a list into one dict, adding appending rows to columns when
        the same fields exist in multiple nodes of the list. Appropriate
        for combining small pieces of catalogs into larger ones.
        """

        mu = {}

        while len(mapunit.children)>0:
            if len(mapunit.children)>1:
                raise ValueError("mapunit has more than one branch!")

            for key in mapunit.data.keys():
                if key in mu.keys():
                    if len(mu[key].shape) == 1:
                        mu[key] = np.hstack([mu[key], mapunit.data[key]])
                    else:
                        mu[key] = np.vstack([mu[key], mapunit.data[key]])                    
                else:
                    mu[key] = mapunit.data[key]

            mapunit = mapunit.children[0]

        for key in mapunit.data.keys():
            if key in mu.keys():
                if len(mu[key].shape) == 1:
                    mu[key] = np.hstack([mu[key], mapunit.data[key]])
                else:
                    mu[key] = np.vstack([mu[key], mapunit.data[key]])                    
            else:
                mu[key] = mapunit.data[key]

        return mu


    def scListToDict(self, mapunit):
        """
        Association schemas with only one catalog type
        result in mappables which are lists. This method compresses
        such a list into one dict, adding new columns when
        the same field exists in multiple nodes of the list. Appropriate
        for combining different files for the same galaxies.
        """

        mu = {}

        while len(mapunit.children)>0:
            if len(mapunit.children)>1:
                raise ValueError("mapunit has more than one branch!")

            for key in mapunit.data.keys():
                if key in mu.keys():
                    if len(mu[key].shape)<2:
                        mu[key] = np.atleast_2d(mu[key]).T
                        mapunit.data[key] = np.atleast_2d(mapunit.data[key]).T

                    shp0 = mu[key].shape
                    shp1 = mapunit.data[key].shape

                    nshp = [shp0[0], shp0[1]+shp1[1]]

                    d = np.ndarray(nshp)
                    d[:,:shp0[1]] = mu[key]
                    d[:,shp0[1]:] = mapunit.data[key]

                    mu[key] = d
                else:
                    mu[key] = mapunit.data[key]

            mapunit = mapunit.children[0]

        for key in mapunit.data.keys():
            if key in mu.keys():
                if len(mu[key].shape)<2:
                    mu[key] = np.atleast_2d(mu[key]).T
                    mapunit.data[key] = np.atleast_2d(mapunit.data[key]).T

                shp0 = mu[key].shape
                shp1 = mapunit.data[key].shape

                nshp = [shp0[0], shp0[1]+shp1[1]]

                d = np.ndarray(nshp)
                d[:,:shp0[1]] = mu[key]
                d[:,shp0[1]:] = mapunit.data[key]
                mu[key] = d
            else:
                mu[key] = mapunit.data[key]

        return mu

    def sortMappableByZ(self, mappable, fieldmap, idx):
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
                self.sortMappableByZ(child, fieldmap, idx)

    def sortMapunitByZ(self, mapunit):

        dk = mapunit.keys()

        if 'redshift' in dk:
            idx = mapunit['redshift'].argsort()

        for k in dk:
            mapunit[k] = mapunit[k][idx]

        return mapunit

    def maskMappable(self, mapunit, mappable):

        if mappable.jtype == 'healpix':
            if mappable.gnside == 0:
                return mapunit

            tp = np.zeros((len(mapunit[mapunit.keys()[0]]),2))
            print('Masking {0} using healpix {1}'.format(mappable.name, mappable.grp))
            for i, key in enumerate(['azim_ang', 'polar_ang']):

                if self.galaxycatalog is not None:
                    if 'azim_ang' in self.galaxycatalog.unitmap.keys():
                        um = self.galaxycatalog.unitmap
                        nest = self.galaxycatalog.nest

                elif self.halocatalog is not None:
                    if 'azim_ang' in self.halocatalog.unitmap.keys():
                        um = self.halocatalog.unitmap
                        nest = self.halocatalog.nest

                conversion = getattr(units, '{0}2{1}'.format(um[key],'rad'))

                tp[:,i] = conversion(mapunit, key)

            pix = hp.ang2pix(mappable.gnside, tp[:,1], tp[:,0], nest=nest)
            pidx = pix==mappable.grp

            if self.maskfile is not None:
                if self.mask is None:
                    self.mask, self.mask_header = hp.read_map(self.maskfile)
                    self.mask_header = dict(self.mask_header)

                pix = hp.ang2pix(self.mask_header['NSIDE'], tp[:,1], tp[:,0],
                                  nest=self.mask_header['ORDERING']=='NEST')

                if self.maskcomp == 'lt':
                    pidx &= self.mask[pix]<self.maskval
                elif self.maskcomp == 'gt':
                    pidx &= self.mask[pix]>self.maskval
                elif self.maskcomp == 'gte':
                    pidx &= self.mask[pix]>=self.maskval
                elif self.maskcomp == 'lte':
                    pidx &= self.mask[pix]<=self.maskval
                elif self.maskcomp == 'eq':
                    pidx &= self.mask[pix]==self.maskval
                elif self.maskcomp == 'neq':
                    pidx &= self.mask[pix]!=self.maskval
                else:
                    raise('Comparison {} not supported'.format(self.maskcomp))

            mu = {}
            for k in mapunit.keys():
                mu[k] = mapunit[k][pidx]

            del mapunit

            mapunit = mu
            return mapunit

        elif mappable.jtype == 'angular-generic':
            tp = np.zeros((len(mapunit[mapunit.keys()[0]]),2))

            print('Masking {0} using angular-generic {1}'.format(mappable.name, mappable.grp))
            for i, key in enumerate(['azim_ang', 'polar_ang']):
                try:
                    conversion = getattr(self, '{0}2{1}'.format(self.unitmap[key],'rad'))
                except:
                    conversion = getattr(units, '{0}2{1}'.format(self.unitmap[key],'rad'))

                tp[:,i] = conversion(mapunit, key)

            if self.mask is None:
                self.mask, self.mask_header = hp.read_map(self.maskfile)
                self.mask_header = dict(self.mask_header)

            pix = hp.ang2pix(self.mask_header['NSIDE'], tp[:,1], tp[:,0],
                              nest=self.mask_header['ORDERING']=='NEST')

            if self.maskcomp == 'lt':
                pidx = self.mask[pix]<self.maskval
            elif self.maskcomp == 'gt':
                pidx = self.mask[pix]>self.maskval
            elif self.maskcomp == 'gte':
                pidx = self.mask[pix]>=self.maskval
            elif self.maskcomp == 'lte':
                pidx = self.mask[pix]<=self.maskval
            elif self.maskcomp == 'eq':
                pidx = self.mask[pix]==self.maskval
            elif self.maskcomp == 'neq':
                pidx = self.mask[pix]!=self.maskval
            else:
                raise('Comparison {} not supported'.format(self.maskcomp))

            mu = {}
            for k in mapunit.keys():
                mu[k] = mapunit[k][pidx]

            mapunit = mu
            return mapunit

        elif mappable.jtype == 'subbox':
            if mappable.nbox == 0:
                return mapunit

            tp = np.zeros((len(mapunit[mapunit.keys()[0]]),3))
            if self.galaxycatalog is not None:
                if 'px' in self.galaxycatalog.unitmap.keys():
                    um = self.galaxycatalog.unitmap

            elif self.halocatalog is not None:
                if 'px' in self.halocatalog.unitmap.keys():
                    um = self.halocatalog.unitmap

            elif self.particlecatalog is not None:
                if 'px' in self.particlecatalog.unitmap.keys():
                    um = self.particlecatalog.unitmap

            print('Masking {0} using subbox {1}'.format(mappable.name, mappable.grp))

            for i, key in enumerate(['px', 'py', 'pz']):
                if um[key] != 'mpch':
                    try:
                        conversion = getattr(self, '{0}2{1}'.format(um[key],'mpch'))
                    except:
                        conversion = getattr(units, '{0}2{1}'.format(um[key],'mpch'))

                    tp[:,i] = conversion(mapunit, key)
                else:
                    tp[:,i] = mapunit[key]

            xi = (mappable.nbox * tp[:,0]) // self.boxsize
            yi = (mappable.nbox * tp[:,1]) // self.boxsize
            zi = (mappable.nbox * tp[:,2]) // self.boxsize

            bidx = xi * mappable.nbox**2 + yi * mappable.nbox + zi

            pidx = bidx==mappable.grp

            mu = {}
            for k in mapunit.keys():
                mu[k] = mapunit[k][pidx]

            mapunit = mu
            return mapunit

        elif mappable.jtype is None:
            return mapunit
        else:
            raise NotImplementedError

    def convert(self, mapunit, metrics):

        if (self.galaxycatalog is not None):
            mapunit = self.galaxycatalog.convert(mapunit, metrics)

        if (self.halocatalog is not None):
            mapunit = self.halocatalog.convert(mapunit, metrics)

        return mapunit

    def filter(self, mapunit):

        if (self.galaxycatalog is not None):
            mapunit = self.galaxycatalog.filter(mapunit, self.galaxycatalog.fieldmap)
        if (self.halocatalog is not None):
            mapunit = self.halocatalog.filter(mapunit, self.halocatalog.fieldmap)

        return mapunit

    def validate(self, nmap=None, metrics=None, verbose=False, parallel=False):
        """
        Run all validation metrics by iterating over only the files we
        need at a given time, mapping catalogs to relevant statistics
        which are reduced at the end of the iteration into observables
        that we care about
        """

        if metrics==None:
            try:
                metrics = self.metrics
            except AttributeError as e:
                self.metrics = []
                if (self.galaxycatalog is not None):
                    self.metrics.extend(self.galaxycatalog.metrics)
                if (self.halocatalog is not None):
                    self.metrics.extend(self.halocatalog.metrics)

                metrics = self.metrics

        #get rid of metrics that don't need to be mapped
        metrics = [m for m in metrics if not m.nomap]

        self.metric_groups = self.genMetricGroups(metrics)

        #metric group w/ Area in it should be first
        areaidx = None
        for mi, mg in enumerate(self.metric_groups):
            ms = mg[1]
            for mj, m in enumerate(ms):
                if m.__class__.__name__ == 'Area':
                    areaidx = mi
                    maidx   = mj

                    nms = []
                    nms.append(m)
                    nms.extend(ms[:mj])
                    nms.extend(ms[mj+1:])
                    self.metric_groups[mi] = [mg[0],nms]
                    break

        if areaidx is not None:
            mgs = []
            mgs.append(self.metric_groups[areaidx])
            mgs.extend(self.metric_groups[:areaidx])
            mgs.extend(self.metric_groups[areaidx+1:])
            self.metric_groups = mgs

        if parallel:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()
        else:
            rank = None
            comm = None

        for mg in self.metric_groups:
            sbz = False
            ms  = mg[1]
            fm  = mg[0]
            for ft in fm.keys():
                if 'redshift' in fm[ft].keys():
                    sbz = True

            mappables = self.genMappables(mg)

            if nmap is not None:
                mappables = mappables[:nmap]

            self.njacktot = len(mappables)

            if parallel:
                mappables = mappables[rank::size]

            self.njack = len(mappables)

            for i, mappable in enumerate(mappables):

                mapunit = self.readMappable(mappable, fm)

#                if (sbz & (ms[0].aschema != 'galaxygalaxy')
#                  & (ms[0].aschema != 'halohalo')):
#                    self.sortMappableByZ(mapunit, fm, [])

                if (not hasattr(ms,'__iter__')):
                    aschema = ms.aschema 
                else:
                    aschema = ms[0].aschema

                if ('only' in aschema) & (mappable.jtype is None):
                    mapunit = self.dcListToDict(mapunit)
                    mapunit = self.maskMappable(mapunit, mappable)
                    mapunit = self.convert(mapunit, ms)
                    mapunit = self.filter(mapunit)
                    if sbz:
                        mapunit = self.sortMapunitByZ(mapunit)

                elif ('only' in aschema) & (mappable.jtype is not None):
                    mapunit = self.dcListToDict(mapunit)
                    mapunit = self.maskMappable(mapunit, mappable)
                    mapunit = self.convert(mapunit, ms)
                    mapunit = self.filter(mapunit)
                    if sbz:
                        mapunit = self.sortMapunitByZ(mapunit)

                elif sbz & ((aschema == 'galaxygalaxy')
                  | (aschema == 'halohalo')
                  | (aschema == 'particleparticle')):
                    mapunit = self.dcListToDict(mapunit)
                    mapunit = self.maskMappable(mapunit, mappable)
                    mapunit = self.convert(mapunit, ms)
                    mapunit = self.filter(mapunit)
                    if sbz:
                        mapunit = self.sortMapunitByZ(mapunit)
                elif ((aschema == 'galaxygalaxy')
                  | (aschema == 'halohalo')
                  | (aschema == 'particleparticle')):
                    mapunit = self.dcListToDict(mapunit)
                    mapunit = self.maskMappable(mapunit, mappable)
                    mapunit = self.convert(mapunit, ms)
                    mapunit = self.filter(mapunit)

                for m in ms:
                    print('*****{0}*****'.format(m.__class__.__name__))
                    m.map(mapunit)

                for k in mapunit.keys():
                    del mapunit[k]

                del mapunit
                mappable.recursive_delete()

        #need to reduce area first

        for mg in self.metric_groups:
            ms = mg[1]

            for m in ms:
                m.reduce(rank=rank,comm=comm)
