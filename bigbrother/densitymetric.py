from __future__ import print_function, division
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pylab as plt
import numpy as np

from .metric import Metric


class DensityMagnitudePDF(Metric):
    """
    Calculate the joint probability density of environmental
    density and magnitude.
    """

    def __init__(self, ministry, zbins=None, densbins=None,
                  magbins=None, catalog_type=None,
                  tag=None, central_only=False, normed=False):

        Metric.__init__(self, ministry, catalog_type=catalog_type, tag=tag)

        if zbins is None:
            self.zbins = np.linspace(ministry.minz, ministry.maxz, 6)
        else:
            self.zbins = zbins

        if densbins is None:
            self.densbins = np.logspace(0.1, 10, 61)
        else:
            self.densbins = densbins

        if magbins is None:
            self.magbins = np.linspace(-24, -19, 61)
        else:
            self.magbins = magbins

        self.nzbins = len(self.zbins) - b1
        self.nmagbins = len(self.magbins) - 1
        self.ndensbins = len(self.densbins) - 1
        self.central_only = central_only

        if self.central_only:
            self.mapkeys = ['luminosity', 'density', 'redshift', 'central']
        else
            self.mapkeys = ['luminosity', 'density', 'redshift']

        self.aschema = 'galaxyonly'
        self.unitmap = {'luminosity':'mag', 'density':'mpch'}

    def map(self, mapunit):

        if not hasattr(self, 'densmagcounts'):
            self.densmagcounts = np.zeros((self.ndensbins, self.nmagbins,
                                            self.nzbins))

        for i, z in enumarate(self.zbins[:-1]):
            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

            c, e0, e1 = np.histogram2d(mapunit['density'],
                                        mapunit['luminosity']),
                                        bins=[self.densbins, self.magbins],
                                        normed=self.normed)

            self.densmagcounts += c

    def reduce(self):
        area = self.ministry.galaxycatalog.getArea()

        if self.normed:
            dd = self.densbins[1:] - self.densbins[:-1]
            dm = self.magbins[1:] - self.magbins[:-1]
            dddm = np.outer(dd, dm)
            self.densmagpdf = self.densmagcounts / area / dddm
        else:
            self.densmagpdf = self.densmagcounts / area

    def visualize(self, compare=False, plotname=None, f=None, ax=None,
                  usez=None, **kwargs):

        mdens = (self.densbins[1:]+self.densbins[:-1])/2
        mmag = (self.magbins[1:]+self.magbins[:-1])/2

        X, Y = np.meshgrid(mdens, mmag)

        if usez is None:
            usezs = range(self.densmagpdf.shape[2])

        if f is None:
            f, ax = plt.subplots(self.nzbins,
                                   sharex=True, sharey=True, figsize=(8,8))
            ax = np.array(ax)
            newaxes = True
        else:
            newaxes = False

        for i in range(self.nzbins):
            l1 = ax[i].contour(X, Y, self.densmagpdf[:,:,i].T, 10,
                                    **kwargs)

        if newaxes:
            sax = f.add_subplot(111)
            sax.patch.set_alpha(0.0)
            sax.patch.set_facecolor('none')
            sax.spines['top'].set_color('none')
            sax.spines['bottom'].set_color('none')
            sax.spines['left'].set_color('none')
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

            if xlabel in kwargs:
                sax.set_xlabel(kwargs['xlabel'])
            else:
                sax.set_xlabel(r'Magnitude')

            if xlabel in kwargs:
                sax.set_ylabel(kwargs['ylabel'])
            else:
                sax.set_ylabel(r'Magnitude')

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)


        return f, ax, l1

    def compare(self, othermetrics, plotname=None, usez=None,
                  labels=None, **kwargs):

        tocompare = [self]
        tocompare.extend(othermetrics)

        if usez!=None:
            if not hasattr(usez[0], '__iter__'):
                usez = [usez]*len(tocompare)
            else:
                assert(len(usez)==len(tocompare))
        else:
            usez = [None]*len(tocompare)

        if labels is None:
            labels = [None] * len(tocompare)

        lines = []

        for i, m in enumerate(tocompare):
            if i==0:
                f, ax, l = m.visualize(usez=usez[i], compare=True,
                                    **kwargs)
            else:
                f, ax, l = m.visualize(usez=usez[i], compare=True,
                                        ax=ax, f=f, **kwargs)

            lines.append(l)


        if labels[0]!=None:
            f.legend(lines, labels)

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax
