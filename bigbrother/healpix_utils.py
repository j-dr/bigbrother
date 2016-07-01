from __future__ import print_function
from .metric import Metric, GMetric
import numpy as np
import healpy as hp


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

    def __init__(self, ministry, nside, tag=None):
        """
        Initialize a PixMetric object. Note, all metrics should define
        an attribute called mapkeys which specifies the types of data that they
        expect.

        Arguments
        ---------
        ministry : Ministry
            The ministry object that this metric is associated with.
        """
        Metric.__init__(self, ministry, tag=tag)

        self.nside = nside

        self.mapkeys = ['polar_ang', 'azim_ang']
        self.aschema = 'singleonly'
        self.unitmap = {'polar_ang':'rad', 'azim_ang':'rad'}


    def map(self, mapunit):

        pix = hp.ang2pix(self.nside, mapunit['polar_ang'], mapunit['azim_ang'])

        return pix

    def reduce(self):
        pass

    def visualize(self):
        pass

    def compare(self):
        pass


class Area(Metric):

    def __init__(self, ministry, nside=64, tag=None):

        Metric.__init__(self, ministry, tag=tag)

        self.nside = nside

        self.mapkeys = ['polar_ang', 'azim_ang']
        self.aschema = 'galaxyonly'
        self.catalog_type = ['galaxycatalog']
        self.unitmap = {'polar_ang':'rad', 'azim_ang':'rad'}
        self.area = 0.0

    def map(self, mapunit):

        pix = hp.ang2pix(self.nside, mapunit['polar_ang'], mapunit['azim_ang'])
        upix = np.unique(pix)
        area = hp.nside2pixarea(self.nside,degrees=True) * len(upix)
        self.area += area

    def reduce(self):
        pass

    def visualize(self):
        pass

    def compare(self):
        pass


class HealpixMap(Metric):

    def __init__(self, ministry, nside=64, cuts=None, tag=None):

        Metric.__init__(self, ministry, tag=None)

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

    def reduce(self):
        pass

    def visualize(self, plotname=None, compare=False):
        hp.mollview(self.hmap)
        f = plt.gcf()
        ax = plt.gca()

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax

    def compare(self, othermetric, plotname=None):
        pass
