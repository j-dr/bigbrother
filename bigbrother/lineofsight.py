from __future__ import print_function, division
#if __name__=='__main__':
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import numpy as np

from metric import Metric


class DNDz(Metric):

    def __init__(self, ministry, zbins=None, magbins=None,
                  catalog_type=['galaxycatalog'], tag=None, appmag=True,
                  lower_limit=True, cutband=None, normed=False):
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

        self.lower_limit = lower_limit

        if magbins is None:
            self.magbins = None
            self.nmagbins = 0
            self.nomags = True
        else:
            self.nomags = False
            self.magbins = magbins
            if self.lower_limit:
                self.nmagbins = len(self.magbins)
            else:
                self.nmagbins = len(self.magbins) - 1

        self.normed = normed

        self.aschema = 'galaxyonly'

        if self.nmagbins > 0:
            self.mapkeys = [self.mkey, 'redshift']
            self.unitmap = {self.mkey :'mag'}
        else:
            self.mapkeys = ['redshift']
            self.unitmap = {}

    def map(self, mapunit):

        if self.nmagbins>0:
            if not hasattr(self, 'dndz'):
                self.dndz = np.zeros((self.nzbins, self.nmagbins))

            for i in range(self.nmagbins):
                if self.lower_limit:
                    if len(mapunit[self.mkey].shape)>1:
                        idx = mapunit[self.mkey][:,self.cutband]<self.magbins[i]
                    else:
                        idx = mapunit[self.mkey]<self.magbins[i]
                else:
                    if i==self.nmagbins: continue

                    if len(mapunit[self.mkey].shape)>1:
                        idx = (self.magbins[i]<mapunit[self.mkey][:,self.cutband]) & (mapunit[self.mkey][:,self.cutband]<self.magbins[i+1])
                    else:
                        idx = (self.magbins[i]<mapunit[self.mkey]) & (mapunit[self.mkey]<self.magbins[i+1])

                c, e = np.histogram(mapunit['redshift'][idx],
                                      bins=self.zbins)
                self.dndz[:,i] += c
        else:
            if not hasattr(self, 'dndz'):
                self.dndz = np.zeros((self.nzbins,1))

            c, e = np.histogram(mapunit['redshift'],
                                  bins=self.zbins)
            self.dndz[:,0] += c


    def reduce(self):
        area = self.ministry.galaxycatalog.getArea()
        if self.normed:
            dz = self.zbins[1:]-self.zbins[:-1]
            self.dndz = self.dndz/area/dz
        else:
            self.dndz = self.dndz/area

    def visualize(self, plotname=None, xlim=None, ylim=None, fylim=None,
                  f=None, ax=None, xlabel=None,ylabel=None,compare=False,
                  usecuts=None, onepanel=False, **kwargs):

        if not hasattr(self, 'zmean'):
            self.zmean = (self.zbins[:-1]+self.zbins[1:])/2

        if (usecuts is None) & (not self.nomags):
            usecuts = range(self.nmagbins)
        elif self.nomags:
            usecuts = [0]

        if f is None:
            if not onepanel:
                f, ax = plt.subplots(len(usecuts), sharex=True, sharey=True,
                                    figsize=(15,15))
            else:
                f, ax = plt.subplots(1, figsize=(15,15))

            ax = np.atleast_1d(ax)
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
            sax.set_ylabel(r'$\frac{dN}{dZ}\, [deg^{-2}]$')
            sax.set_xlabel(r'$z$')

        for i, c in enumerate(usecuts):
            if onepanel:
                l1 = ax[0].step(self.zbins, np.hstack([self.dndz[:,c],
                                self.dndz[-1,c]]), where='post')
            else:
                l1 = ax[i].step(self.zbins, np.hstack([self.dndz[:,c],
                                self.dndz[-1,c]]), where='post')

        #plt.tight_layout()

        if (plotname is not None) and (not compare):
            plt.savefig(plotname)

        return f, ax, l1

    def compare(self, othermetrics, plotname=None, usecuts=None, labels=None,
                  **kwargs):

        tocompare = [self]
        tocompare.extend(othermetrics)

        lines = []

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
                f, ax, l1 = m.visualize(usecuts=usecuts[i], compare=True,
                                    **kwargs)
            else:
                f, ax, l1 = m.visualize(usecuts=usecuts[i], compare=True,
                                    f=f, ax=ax, **kwargs)
            lines.extend(l1)

        if labels!=None:
            f.legend(lines, labels, 'best')

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax


class TabulatedDNDz(DNDz):

    def __init__(self, *args, **kwargs):

        if 'fname' in kwargs:
            self.fname = kwargs.pop('fname')
        else:
            raise(ValueError("Please supply a path to the tabulated luminosity function using the fname kwarg!"))

        if 'ncuts' in kwargs:
            self.ncuts = kwargs.pop('ncuts')
        else:
            self.ncuts = 1

        if 'lowzcol' in kwargs:
            self.lowzcol = int(kwargs.pop('lowzcol'))
        else:
            self.lowzcol = 0

        if 'highzcol' in kwargs:
            self.highzcol = int(kwargs.pop('highzcol'))
        else:
            self.highzcol = 0

        if 'ycol' in kwargs:
            self.ycol = int(kwargs.pop('ycol'))
        else:
            self.ycol = 1

        DNDz.__init__(self,*args,**kwargs)

        #don't need to map this guy
        self.nomap = True
        self.loadDNDz()

    def loadDNDz(self):
        tab = np.genfromtxt(self.fname)
        self.dndz = np.zeros((tab.shape[0], self.ncuts))

        lowz = tab[:,self.lowzcol]
        highz = tab[:,self.highzcol]
        self.zbins = np.zeros(len(lowz)+1)
        self.zbins[:-1] = lowz
        self.zbins[-1] = highz[-1]
        for i in range(self.ncuts):
            self.dndz[:,i] = tab[:,i+self.ycol]
