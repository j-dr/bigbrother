from __future__ import print_function
from .metric import Metric, GMetric
from scipy.stats import mode
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import fitsio

from .metric import jackknifeMap

def sortHpixFileStruct(filestruct):

    if len(filestruct.keys())>1:
        opix =  np.array([int(t.split('/')[-1].split('.')[-2]) for t
                          in filestruct[filetypes[0]]])
        oidx = opix.argsort()

        for ft in filetypes:
            assert(len(filestruct[ft])==len(filestruct[filetypes[0]]))
            pix = np.array([int(t.split('/')[-1].split('.')[-2]) for t
                            in filestruct[ft]])
            idx = pix.argsort()
            assert(pix[idx]==opix[oidx])

            if len(idx)==1:
                filestruct[ft] = [filestruct[ft][idx]]
            else:
                filestruct[ft] = filestruct[ft][idx]

    return filestruct

class PixMetric(Metric):

    def __init__(self, ministry, nside, nest=False, tag=None, **kwargs):
        """
        Initialize a PixMetric object. Note, all metrics should define
        an attribute called mapkeys which specifies the types of data that they
        expect.

        Arguments
        ---------
        ministry : Ministry
            The ministry object that this metric is associated with.
        """
        Metric.__init__(self, ministry, tag=tag, **kwargs)

        self.nside = nside
        self.nest  = nest

        self.mapkeys = ['polar_ang', 'azim_ang']
        self.aschema = 'singleonly'
        self.unitmap = {'polar_ang':'rad', 'azim_ang':'rad'}


    def map(self, mapunit):

        pix = hp.ang2pix(self.nside, mapunit['polar_ang'], mapunit['azim_ang'], nest=self.nest)

        return np.unique(pix)

    def reduce(self, rank=None, comm=None):
        pass

    def visualize(self):
        pass

    def compare(self):
        pass

class SubBoxMetric(Metric):

    def __init__(self, ministry, nbox, tag=None, **kwargs):
        """
        Initialize a PixMetric object. Note, all metrics should define
        an attribute called mapkeys which specifies the types of data that they
        expect.

        Arguments
        ---------
        ministry : Ministry
            The ministry object that this metric is associated with.
        """
        Metric.__init__(self, ministry, tag=tag, **kwargs)

        self.nbox = nbox
        self.lbox = self.ministry.boxsize

        self.mapkeys = ['px', 'py', 'pz']
        self.aschema = 'singleonly'
        self.unitmap = {'px':'mpch', 'py':'mpch', 'pz':'mpch'}


    def map(self, mapunit):

        print('self.nbox: {0}'.format(self.nbox))
        print('self.lbox: {0}'.format(self.lbox))

        xi = (self.nbox * mapunit['px']) // self.lbox
        yi = (self.nbox * mapunit['py']) // self.lbox
        zi = (self.nbox * mapunit['pz']) // self.lbox

        #fix edge case
        xi[xi==self.nbox] = self.nbox-1
        yi[yi==self.nbox] = self.nbox-1
        zi[zi==self.nbox] = self.nbox-1

        bidx = xi * self.nbox**2 + yi * self.nbox + zi

        return np.unique(bidx)

    def reduce(self, rank=None, comm=None):
        pass

    def visualize(self):
        pass

    def compare(self):
        pass



class Area(Metric):

    def __init__(self, ministry, nside=2048, tag=None, maskfile=None,
                 nside_group=None, nside_mask=None, nest_group=False,
                 nest_mask=False, **kwargs):

        Metric.__init__(self, ministry, tag=tag, novis=True, **kwargs)

        self.nside = nside

        if self.catalog_type is None:
            self.catalog_type = ['galaxycatalog']

        if self.catalog_type == ['galaxycatalog']:
            self.aschema = 'galaxyonly'
        elif self.catalog_type == ['halocatalog']:
            self.aschema = 'haloonly'

        if self.aschema =='galaxyonly':
            self.mapkeys = ['polar_ang', 'azim_ang', 'appmag']
            self.unitmap = {'polar_ang':'rad', 'azim_ang':'rad', 'appmag':'mag'}
        else:
            self.mapkeys = ['polar_ang', 'azim_ang']
            self.unitmap = {'polar_ang':'rad', 'azim_ang':'rad'}

        if maskfile is not None:
            self.mask = fitsio.read(maskfile)
        else:
            self.mask = None

        self.nside_mask  = nside_mask
        self.nside_group = nside_group

        self.nest_mask = nest_mask
        self.nest_group = nest_group

        self.jarea = None


    @jackknifeMap
    def map(self, mapunit):

        if self.jarea is None:
            self.jarea = np.zeros(self.njack)

        if self.mask is None:
            pix = hp.ang2pix(self.nside, mapunit['polar_ang'], mapunit['azim_ang'],
                             nest=True)
            upix = np.unique(pix)
            area = hp.nside2pixarea(self.nside,degrees=True) * len(upix)
            self.jarea[self.jcount] += area
        else:
            pix = hp.ang2pix(self.nside_group, mapunit['polar_ang'], mapunit['azim_ang'],
                             nest=self.nest_group)
            upix = np.unique(pix)

            if len(upix)>1:
                print('Hmm, there is more than one group pixel in this mappable, using the most frequent')
                p = mode(upix)[0][0]
            else:
                p = pix[0]
            
            if self.nest_group:
                order_in = 'NESTED'
            else:
                order_in = 'RING'
            
            if self.nest_mask:
                order_out = 'NESTED'
            else:
                order_out = 'RING'

            ud_map = hp.ud_grade(np.arange(12*self.nside_group**2), self.nside_mask,
                                 order_in=order_in, order_out=order_out)
            mpix   = ud_map[self.mask['HPIX']]
            npix   = np.sum(self.mask['FRACGOOD'][mpix==p,2])

            pix_area = hp.nside2pixarea(self.nside_mask)

            self.jarea[self.jcount] += pix_area * npix
                

    def reduce(self, rank=None, comm=None):
        if rank is not None:
            garea = comm.gather(self.jarea, root=0)

            if rank == 0:
                gshape = [self.jarea.shape[i] for i in range(len(self.jarea.shape))]
                gshape[0] = self.njacktot

                self.jarea = np.zeros(gshape)
                jc = 0

                for g in garea:
                    if g is None: continue
                    nj = g.shape[0]
                    self.jarea[jc:jc+nj] = g

                    jc += nj

                self.jarea, self.area, self.vararea = self.jackknife(self.jarea)

        else:
            self.jarea, self.area, self.vararea = self.jackknife(self.jarea)


    def visualize(self):
        pass

    def compare(self):
        pass


class HealpixMap(Metric):

    def __init__(self, ministry, nside=64, cuts=None, tag=None, **kwargs):

        Metric.__init__(self, ministry, tag=None, **kwargs)

        self.nside = nside
        self.cuts  = cuts

        if cuts is None:
            self.mapkeys = ['polar_ang', 'azim_ang']
            self.ncuts = 1
            self.cutkey = None
        else:
            self.mapkeys = ['polar_ang', 'azim_ang']
            self.cutkey = self.cuts.keys()[0]
            self.cuts = self.cuts[self.cutkeys]
            self.mapkeys.append(self.cutkey)
            self.ncuts = len(cuts[self.cutkey])

        self.aschema      = 'singleonly'
        if self.catalog_type is None:
            self.catalog_type = ['galaxycatalog']
            
        self.unitmap      = {'polar_ang':'rad', 'azim_ang':'rad'}
        self.pbins        = np.arange(12*nside**2+1)
        self.hmap         = np.zeros((12*nside**2, self.ncuts))

    def map(self, mapunit):

        pix = hp.ang2pix(self.nside, mapunit['polar_ang'], mapunit['azim_ang'])

        if self.cuts is None:
            c, e = np.histogram(pix, bins=self.pbins)
            self.hmap[:,0] += c
        else:
            for i, c in enumerate(self.cuts):
                cidx, = np.where(mapunit[self.cutkey]>c)
                c, e = np.histogram(pix[cidx], bins=self.pbins)
                self.hmap[:,i] += c

    def reduce(self, rank=None, comm=None):
        if rank is not None:
            ghmap = None
            ghmap = comm.gather(self.hmap, root=0)

            if rank==0:
                self.hmap = np.zeros_like(ghmap[0])
                for g in ghmap:
                    if g is None: continue
                    self.hmap += g

    def visualize(self, plotname=None, compare=False):
        hp.mollview(self.hmap[:,0])
        f = plt.gcf()
        ax = plt.gca()

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax

    def compare(self, othermetric, plotname=None):
        pass
