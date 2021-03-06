from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
from .healpix_utils import PixMetric, SubBoxMetric
from astropy.cosmology import FlatLambdaCDM
from . import units
import numpy as np
import healpy as hp
import fitsio
import time
import sys

class BaseCatalog:
    """
    Base class for catalog type
    """

    _valid_reader_types = ['fits', 'rockstar', 'ascii', 'lgadget', 'addgalstxt']

    def __init__(self, ministry, filestruct, fieldmap=None,
                 unitmap=None,  filters=None, goodpix=None,
                 reader=None, area=None, jtype=None, nbox=None,
                 filenside=None, groupnside=None, nest=True,
                 maskfile=None, maskcomp=None, maskval=None,
                 necessaries=None, jackknife_area=None,
                 polar_ang_key='polar_ang', azim_ang_key='azim_ang',
                 px_key='px', py_key='py', pz_key='pz', 
                 azim_ang_lims=None, polar_ang_lims=None,
                 redshift_lims=None):

        self.ministry = ministry
        self.filestruct = filestruct
        self.fieldmap = fieldmap
        self.unitmap  = unitmap
        self.polar_ang_key = polar_ang_key
        self.azim_ang_key = azim_ang_key

        self.px_key = px_key
        self.py_key = py_key
        self.pz_key = pz_key

        self.polar_ang_lims = polar_ang_lims
        self.azim_ang_lims  = azim_ang_lims
        self.redshift_lims = redshift_lims

        if filters is not None:
            self.filters = filters
        else:
            self.filters = []
        self.parseFileStruct(filestruct)
        self.maskfile = maskfile
        self.maskcomp = maskcomp
        self.maskval  = maskval
        self.mask = None

        if area is None:
            self.area = 0.0
        else:
            self.area = area

        self.jackknife_area = jackknife_area

        #jackknife information
        self.jtype = jtype

        #for healpix type jackknifing
        self.filenside = filenside
        self.nest = nest

        if groupnside is None:
            self.groupnside = 4
        else:
            self.groupnside = groupnside

        #for subbox type jackknifing
        self.nbox = nbox

        #for mask type jackknifing
        if goodpix is None:
            self.goodpix = 1
        else:
            self.goodpix = goodpix

        if necessaries is None:
            self.necessaries = []
        else:
            self.necessaries = necessaries

        if reader in BaseCatalog._valid_reader_types:
            self.reader = reader
        elif reader is None:
            self.reader = 'fits'
        else:
            raise(ValueError("Invalid reader type {0} specified".format(reader)))

        self.addFilterKeysToNecessaries()

    @abstractmethod
    def parseFileStruct(self, filestruct):
        """
        Given a filestruct object, create map from parameters
        we require to filepaths for easy access
        """

    def getFilePixels(self, nside):
        """
        Get the healpix cells occupied by galaxies
        in each file. Assumes files have already been
        sorted correctly by parseFileStruct
        """
        fpix = []
        print('Test')
        #BCC catalogs have pixels in filenames
        if (self.filenside is not None) & (self.filenside>=self.groupnside):
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

            pmetric = PixMetric(self.ministry, self.groupnside,
                                catalog_type=ct, nest=self.nest,
                                polar_ang_key=self.polar_ang_key,
                                azim_ang_key=self.azim_ang_key)

            mg = self.ministry.genMetricGroups([pmetric])
            ms = mg[0][1]
            fm = mg[0][0]

            mappables = self.ministry.genMappables(mg[0], override=True)

            if self.ministry.parallel:
                from mpi4py import MPI

                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                size = comm.Get_size()

                mappables = mappables[rank::size]
                print('{} Grouping files, num mappables: {}'.format(rank, len(mappables)))

            for i, mappable in enumerate(mappables):

                mapunit = self.ministry.readMappable(mappable, fm)

                if (not hasattr(ms,'__iter__')) and ('only' in ms.aschema):
                    mapunit = self.ministry.scListToDict(mapunit)
                    mapunit = self.ministry.convert(mapunit, ms)
                    mapunit = self.ministry.filter(mapunit)

                elif 'only' in ms[0].aschema:
                    mapunit = self.ministry.scListToDict(mapunit)
                    mapunit = self.ministry.convert(mapunit, ms)
                    mapunit = self.ministry.filter(mapunit)

                if ((ms[0].aschema == 'galaxygalaxy')
                  | (ms[0].aschema == 'halohalo')
                  | (ms[0].aschema == 'particleparticle')):
                    mapunit = self.ministry.dcListToDict(mapunit)
                    mapunit = self.ministry.convert(mapunit, ms)
                    mapunit = self.ministry.filter(mapunit)

                fpix.append(pmetric.map(mapunit))

                del mapunit
                mappable.recursive_delete()

            if self.ministry.parallel:
                gfpix = comm.allgather(fpix)
                fpix = []
                for fp in gfpix:
                    fpix.extend(fp)

        return fpix

    def addFilterKeysToNecessaries(self):

        self.necessaries.extend([f.lower() for f in self.filters])
        self.necessaries = np.unique(self.necessaries)

    def groupFiles(self):
        """
        Group files together spatially. Healpix and cubic subboxes
        implemented
        """

        if self.jtype == 'healpix':
            print('Grouping files according to nside={}'.format(self.groupnside))

            fpix = self.getFilePixels(self.groupnside)
            upix = np.unique(np.array([p for sublist in fpix for p in sublist]))
            fgrps = []

            for p in upix:
                fgrps.append([i for i in range(len(fpix)) if p in fpix[i]])

            return np.array(upix), fgrps

        elif self.jtype == 'angular-generic':

            raise(NotImplementedError)

        elif self.jtype == 'subbox':
            print('Grouping files into n={} subboxes'.format(self.nbox**3))

            fsbox = self.getFileSubBoxes(self.nbox, [self.ctype])

            ubox = np.unique(np.array([b for sublist in fsbox for b in sublist]))
            fgrps = []

            for b in ubox:
                fgrps.append([i for i in range(len(fsbox)) if b in fsbox[i]])

            return ubox, fgrps

    def readFITSMappable(self, mappable, fieldmap):
        """
        For the file held in mappable, read in the fields defined in
        fieldmap. File must be in FITS format.
        """

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

        data = fitsio.read(fname, columns=fields)

        if hasattr(self, 'downsample_factor'):
            if self.downsample_factor is not None:
                idx  = np.random.choice(np.arange(len(data)), size=len(data)//self.downsample_factor)
                data = data[idx]

        for mapkey in fieldmap[ft].keys():

            if hasattr(fieldmap[ft][mapkey], '__iter__'):
                idx = np.array([data.dtype.names.index(fieldmap[ft][mapkey][i]) 
                                for i in range(len(fieldmap[ft][mapkey]))])
                dt = data[fieldmap[ft][mapkey]].dtype[0]
                mapunit[mapkey] = data.view((dt, len(fields)))[:,idx]
            else:
                mapunit[mapkey] = data[fieldmap[ft][mapkey]]

        return mapunit

    def getFileSubBoxes(self, nbox, ct):

        fbox = []

        if 'BCC' in self.__class__.__name__:
            raise ValueError('BCC catalogs are not in boxes!')
        else:

            bmetric = SubBoxMetric(self.ministry, nbox,
                                  catalog_type=ct, px_key=self.px_key,
                                  py_key=self.py_key, pz_key=self.pz_key)
            mg = self.ministry.genMetricGroups([bmetric])
            ms = mg[0][1]
            fm = mg[0][0]

            mappables = self.ministry.genMappables(mg[0], override=True)

            if self.ministry.parallel:
                from mpi4py import MPI

                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                size = comm.Get_size()
                
                print('Number of tasks: {}'.format(size))

                mappables = mappables[rank::size]

            for i, mappable in enumerate(mappables):

                if nbox == 0:
                    fbox.append([0])

                else:
                    mapunit = self.ministry.readMappable(mappable, fm)

                    if (not hasattr(ms,'__iter__')) and ('only' in ms.aschema):
                        mapunit = self.ministry.scListToDict(mapunit)
                        mapunit = self.ministry.convert(mapunit, ms)
                        mapunit = self.ministry.filter(mapunit)

                    elif 'only' in ms[0].aschema:
                        mapunit = self.ministry.scListToDict(mapunit)
                        mapunit = self.ministry.convert(mapunit, ms)
                        mapunit = self.ministry.filter(mapunit)

                    elif ((ms[0].aschema == 'galaxygalaxy')
                          | (ms[0].aschema == 'halohalo')
                          | (ms[0].aschema == 'particlecatalog')):
                        mapunit = self.ministry.dcListToDict(mapunit)
                        mapunit = self.ministry.convert(mapunit, ms)
                        mapunit = self.ministry.filter(mapunit)

                    fbox.append(bmetric.map(mapunit))

                    del mapunit
                    mappable.recursive_delete()

            if self.ministry.parallel:
                gfbox = comm.allgather(fbox)
                fbox = []
                for fb in gfbox:
                    if fb is None: continue
                    fbox.extend(fb)

        return fbox

    def maskMappable(self, mapunit, mappable):


        if mappable.jtype == 'healpix':
            tp = np.zeros((len(mapunit[mapunit.keys()[0]]),2))

            print('Masking {0} using healpix {1}'.format(mappable.name, mappable.grp))
            for i, key in enumerate(['azim_ang', 'polar_ang']):
                try:
                    conversion = getattr(self, '{0}2{1}'.format(self.unitmap[key],'rad'))
                except:
                    conversion = getattr(units, '{0}2{1}'.format(self.unitmap[key],'rad'))

                tp[:,i] = conversion(mapunit, key)

            pix = hp.ang2pix(self.groupnside, tp[:,1], tp[:,0], nest=self.nest)
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
            tp = np.zeros((len(mapunit[mapunit.keys()[0]]),3))

            print('Masking {0} using subbox {1}'.format(mappable.name, mappable.grp))
            for i, key in enumerate(['x', 'y', 'z']):
                if self.unitmap[key] != 'mpch':
                    try:
                        conversion = getattr(self, '{0}2{1}'.format(self.unitmap[key],'mpch'))
                    except:
                        conversion = getattr(units, '{0}2{1}'.format(self.unitmap[key],'mpch'))

                    tp[:,i] = conversion(mapunit, key)

            xi = (self.nbox * tp[:,0]) // self.ministry.boxsize
            yi = (self.nbox * tp[:,1]) // self.ministry.boxsize
            zi = (self.nbox * tp[:,2]) // self.ministry.boxsize

            bidx = xi * self.nbox**2 + yi * self.nbox + zi

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

    def readMappable(self, mappable, fieldmap):
        """
        Default reader is FITS reader
        """

        if self.reader=='fits':
            mapunit = self.readFITSMappable(mappable, fieldmap)

        return self.maskMappable(mapunit, mappable)


    def setFieldMap(self, fieldmap):
        self.fieldmap = fieldmap

    def convert(self, mapunit, metrics):
        """
        Convert a map unit from the units given in the catalog
        to those required in metrics
        """
        beenconverted = []
        for m in metrics:
            if self.ctype not in m.catalog_type:
                continue
            for key in m.unitmap:

                if key in beenconverted: continue
                if key not in self.fieldmap.keys(): continue
                if key not in mapunit.keys(): continue

                try:
                    if self.unitmap[key]==m.unitmap[key]:
                        continue
                except KeyError as e:
                    print(e)
                    print("Catalog unitmap: {0}".format(self.unitmap))
                    print("Metric unitmap: {0}".format(m.unitmap))

                try:
                    conversion = getattr(self, '{0}2{1}'.format(self.unitmap[key],m.unitmap[key]))
                except:
                    conversion = getattr(units, '{0}2{1}'.format(self.unitmap[key],m.unitmap[key]))

                if '{0}2{1}'.format(self.unitmap[key],m.unitmap[key]) == 'flux2mag':
                    if hasattr(self, 'zp'):
                        if self.zp is not None:
                            mapunit[key] = conversion(mapunit, key, zp=self.zp)
                        else:
                            mapunit[key] = conversion(mapunit, key)
                    else:
                        mapunit[key] = conversion(mapunit, key)
                elif ('{0}2{1}'.format(self.unitmap[key], m.unitmap[key]) == 'magh2mag') | ('{0}2{1}'.format(self.unitmap[key], m.unitmap[key]) == 'mag2magh'):
                   mapunit[key] = conversion(mapunit, key, h=self.ministry.h)
                elif (('{0}2{1}'.format(self.unitmap[key], m.unitmap[key]) == 'bccmag2mag')
                      | ('{0}2{1}'.format(self.unitmap[key], m.unitmap[key]) == 'fabermag2mag')):
                    mapunit[key] = conversion(mapunit, key, Q=self.Q)
                else:
                    mapunit[key] = conversion(mapunit, key)

                beenconverted.append(key)

        return mapunit

    def filter(self, mapunit, fieldmap):

        idx = None

        for i, key in enumerate(self.filters):
            filt = getattr(self, 'filter{0}'.format(key))

            if key.lower() not in mapunit.keys():
                continue
#                raise(ValueError("Trying to filter on {}, but it is not in the mapunit!".format(key)))

            if i==0:
                idx = filt(mapunit)
            else:
                idxi = filt(mapunit)
                idx = idx&idxi

        if idx is not None:
            for key in mapunit.keys():
                if key in fieldmap.keys():
                    mapunit[key] = mapunit[key][idx]

        return mapunit

    def filterazim_ang(self, mapunit):
        print('Filtering azim ang')
        idx = ( self.azim_ang_lims[0] < mapunit['azim_ang'] ) & ( mapunit['azim_ang'] < self.azim_ang_lims[1])
        
        return idx

    def filterpolar_ang(self, mapunit):
        print('Filtering polar ang')
        idx = ( self.polar_ang_lims[0] < mapunit['polar_ang'] ) & ( mapunit['polar_ang'] < self.polar_ang_lims[1])

        return idx

    def filterredshift(self, mapunit):
        print('Filtering redshift')
        
        idx = (self.redshift_lims[0] < mapunit['redshift']) & (mapunit['redshift'] < self.redshift_lims[1])

        return idx

    def getArea(self, jackknife=False):

        arm = np.array([True if m.__class__.__name__=="Area" else False
                  for m in self.ministry.metrics])
        am = any(arm)

        if am:
            idx, = np.where(arm==True)[0]

        if not jackknife:
            if not am:
                return self.ministry.area
            else:
                return self.ministry.metrics[idx].area
        else:
            if not am:
                return self.ministry.area
            else:
                return self.ministry.metrics[idx].jarea


class PlaceHolder(BaseCatalog):

    def __init__(self, ministry, filestruct, **kwargs):

        self.ctype = 'placeholder'
        BaseCatalog.__init__(self, ministry, filestruct, **kwargs)
