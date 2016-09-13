from __future__ import print_function
from .metric import Metric, GMetric
import numpy as np
import healpy as hp

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

    def __init__(self, ministry, nside, tag=None, **kwargs):
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

        self.mapkeys = ['polar_ang', 'azim_ang']
        self.aschema = 'singleonly'
        self.unitmap = {'polar_ang':'rad', 'azim_ang':'rad'}


    def map(self, mapunit):

        pix = hp.ang2pix(self.nside, mapunit['polar_ang'], mapunit['azim_ang'])
        
        return np.unique(pix)

    def reduce(self, rank=None, comm=None):
        pass

    def visualize(self):
        pass

    def compare(self):
        pass


class Area(Metric):

    def __init__(self, ministry, nside=256, tag=None, **kwargs):

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

        self.jarea = None

    @jackknifeMap
    def map(self, mapunit):

        if self.jarea is None:
            self.jarea = np.zeros(self.njack)

        pix = hp.ang2pix(self.nside, mapunit['polar_ang'], mapunit['azim_ang'],
                         nest=True)
        upix = np.unique(pix)
        area = hp.nside2pixarea(self.nside,degrees=True) * len(upix)
        self.jarea[self.jcount] += area

    def reduce(self, rank=None, comm=None):
        if rank is not None:
            garea = comm.gather(self.jarea, root=0)

            gshape = [self.jarea.shape[i] for i in range(len(self.jarea.shape))]
            gshape[0] = self.njacktot

            if rank == 0:
                self.jarea = np.zeros(gshape)
                jc = 0

                for g in garea:
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
            comm.Reduce(self.hmap, hmap, root=0)
            self.hmap = hmap


    def visualize(self, plotname=None, compare=False):
        hp.mollview(self.hmap)
        f = plt.gcf()
        ax = plt.gca()

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax

    def compare(self, othermetric, plotname=None):
        pass
