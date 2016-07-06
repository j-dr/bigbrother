from __future__ import print_function, division
#if __name__=='__main__':
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import treecorr as tc
import numpy as np

from metric import Metric


class DNDz(Metric):

    def __init__(self, ministry, magbins=None, catalog_type=['galaxycatalog'],
                    tag=None, appmag=True, lower_limit=True, cutband=None):
        """
        Angular Number density of objects as a function of redshift.
        """

        Metric.__init__(self, ministry, tag=tag)

        self.catalog_type = catalog_type

        if zbins is None:
            self.zbins = np.linspace(0, 2.0, 61)
        else:
            self.zbins = zbins
            self.zbins = np.array(self.zbins)

        self.nzbins = len(self.zbins)-1

        if appmag:
            self.mkey = 'appmag'
            defmbins = np.array([18, 19, 20, 21])
        else:
            self.mkey = 'luminosity'
            defmbins = np.array([-24, -23, -22, -21])

        if cutband is None:
            self.cutband = 0
        else:
            self.cutband = cutband

        if magbins is None:
            self.magbins = defmbins
        else:
            self.magbins = magbins

        self.lower_limit = lower_limit

        if self.lower_limit:
            self.nmagbins = len(self.magbins)
        else:
            self.nmagbins = len(self.magbins) - 1

        self.aschema = 'galaxyonly'

        self.mapkeys = [self.mkey, 'redshift']
        self.unitmap = {self.mkey :'mag'}

    def map(self, mapunit):

        if not hasattr(self, 'dndz'):
            self.dndz = np.zeros(self.nzbins, self.nmagbins)

        for i in range(self.nmagbins):
            if self.lower_limit:
                if len(mapunit[self.mkey].shape)>1:
                    idx = mapunit[self.mkey][:,self.cutband]>self.magbins[i]
                else:
                    idx = mapunit[self.mkey]>self.magbins[i]
            else:
                if i==self.nmagbins: continue

                if len(mapunit[self.mkey].shape)>1:
                    idx = (self.magbins[i]<mapunit[self.mkey][:,self.cutband]) & (mapunit[self.mkey][:,self.cutband]<self.magbins[i+1])
                else:
                    idx = (self.magbins[i]<mapunit[self.mkey]) & (mapunit[self.mkey]<self.magbins[i+1])

            self.dndz[:,i] = np.histogram(mapunit['redshift'][idx],
                                            bins=self.zbins)

    def reduce(self):
        area = self.ministry.galaxycatalog.getArea()
        self.dndz = self.dndz/area

    def visualize(self, plotname=None, xlim=None, ylim=None, fylim=None,
                  f=None, ax=None, xlabel=None,ylabel=None,compare=False,
                  usecuts=None, **kwargs):

        if usecuts is None:
            usecuts = range(self.nmagbins)

        if f is None:
            f, ax = plt.subplots(len(usecuts), sharex=True, sharey=True,
                                   figsize=(15,15))
            newaxes = True
        else:
            newaxes = False

        if newaxes:
            sax = f.add_subplot(111)
            sax.patch.set_alpha(0.0)
            sax.patch.set_facecolor('none')
            sax.spines['top'].set_color('none')
            sax.spines['bottom'].set_color('none')
            sax.spines['left'].set_color('none')
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            sax.set_xlabel(r'$\frac{dN}{dZ}\, [deg^{-2}]$')
            sax.set_ylabel(r'$z$')

        for i, c in enumerate(usecuts):
            l1 = ax[i].step(self.zbins, self.dndz[:,i])

        plt.tight_layout()

        if (plotname is not None) and (not compare):
            plt.savefig(plotname)

        return f, ax, l1

    def compare(self, othermetrics, plotname=None, usecuts=None, **kwargs):

        tocompare = [self]
        tocompare.extend(othermetrics)

        if usecuts is not None:
            if not hasattr(usecuts[0], '__iter__'):
                usecuts = [usecuts]*len(tocompare)
            else:
                assert(len(usecuts)==len(tocompare))
        else:
            usecuts = [None]*len(tocompare)

        for i, m in enumerate(tocompare):
            if usecuts[i] is not None:
                assert(len(usecuts[0])==len(usecuts[i]))
            if i==0:
                f, ax = m.visualize(usecuts=usecuts[i], compare=True,
                                    **kwargs)
            else:
                f, ax = m.visualize(usecuts=usecuts[i], compare=True,
                                    f=f, ax=ax, **kwargs)

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax


#class TabulatedDNDz(Metric)
