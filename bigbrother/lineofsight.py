from __future__ import print_function, division
#if __name__=='__main__':
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import numpy as np

from metric import Metric
from .selection import Selector

class DNDz(Metric):

    def __init__(self, ministry, zbins=None, magbins=None,
                  catalog_type=['galaxycatalog'], tag=None, appmag=True,
                  lower_limit=True, cutband=None, normed=True):
        """
        Angular Number density of objects as a function of redshift.

        inputs
        --------
        ministry - Ministry

        keywords
        ------
        zbins - np.array
          An array containing the edges of the redshift bins to use for dn/dz
        lower_limit - boolean
          Whether or not the magnitudes in magbins should be interpreted
          as bin edges or lower limits of cuts (i.e. mag<magbins[i])
        magbins - np.array
          A list of magnitudes to use as bin edges or lower limits of cuts
        catalog_type - list
          A list of catalog types, ususally not used
        tag - string
          A name for the metric to be used when making comparisons using bb-compare
        appmag - boolean
          True if we want to use apparent magnitude for cuts, false for absolute magnitudes
        cutband - int
          The index of the column of a vector of magnitudes to use
        normed - boolean
          Whether the metric integrates to N/deg^2 or not. Usually want True.
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

        #Make selection dict here
        selection_dict = {'mag':{'selection_type':'binned1d',
                                  'mapkeys':'appmag',
                                  'bins':np.array([19.45])}}
        self.selector = Selector(self.selection_dict)

    def map(self, mapunit):
        """
        Map function for dn/dz. Extracts relevant information from a mapunit. Usually only called by a Ministry object, not manually.
        """

        #This will eventually need to be replaced with the outputs
        #if selector.mapArray()
        if not hasattr(self, 'dndz'):
            self.dndz = np.zeros((self.nzbins,self.nmagbins))

        for idx, aidx in self.selector.generateSelections(mapunit):
            c, e = np.histogram(mapunit[self.mkey][idx], bins=self.zbins)
            self.dndz[:,aidx] += c

    def reduce(self):
        """
        Converts extracted redshift information into a (normalized) dn/dz output and stores it as an attribute of the metric.
        """
        area = self.ministry.galaxycatalog.getArea()
        if self.normed:
            dz = self.zbins[1:]-self.zbins[:-1]
            self.dndz = self.dndz/area/dz
        else:
            self.dndz = self.dndz/area

    def visualize(self, plotname=None, xlim=None, ylim=None, fylim=None,
                  f=None, ax=None, xlabel=None,ylabel=None,compare=False,
                  usecuts=None, onepanel=False, **kwargs):
        """
        Plot dn/dz for an individual instance.

        keywords
        -------
        plotname - string
          File name to save plot to
        xlim - list
          x-axis limits (needs to be of length 2, e.g. [xmin, xmax])
        ylim - list
          Same as xlim but for y-axis
        fylim - list
          Fractional deviation subplot y limits. Same format as for xlim.
        f - Figure
          A matplotlib Figure object to use
        ax - Axes
          A matplotlib Axes object to plot on
        xlabel - string
          Label of the x-axis
        ylabel - string
          Label of the y-axis
        compare - boolean
          Whether or not this function is being called from the compare function.
        usecuts - list
          A list of indices of self.dndz to use. I.e. self.dndz[:,usecuts] will be plotted
        onepanel - boolean
          Whether all cuts should be plotted on a single panel
        """

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
        """
        Compare this DNDz metric to a list of other metrics

        inputs
        --------
        othermetrics - list
          A list of DNDz metrics (can also be TabulatedDNDz) to compare this metric to

        keywords
        --------
        plotname - string
          See visualize method
        usecuts - list
          See visualize method
        labels - list
          A list of labels to use for the legend. Order should be the label for this metric first, followed by labels in the order that metrics are given in othermetrics
        """

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

        if labels[0]!=None:
            f.legend(lines, labels, 'best')

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax


class TabulatedDNDz(DNDz):

    def __init__(self, fname, *args, **kwargs):
        """
        Create a DNDz object from a tabulated file

        inputs
        ------
        fname - string
          Name of the file containing the tabulated data

        keywords
        -------
        ncuts - string
          Number of cuts to use in the file. Defaults to 1.
        lowzcol - int
          The column containing the low z edge of the bins
        highzcol - int
          The column containing the high z edge of the bins
        ycol - int
          The column containing the dn/dz measurements
        """

        self.fname = fname

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
            self.highzcol = 1

        if 'ycol' in kwargs:
            self.ycol = int(kwargs.pop('ycol'))
        else:
            self.ycol = 2

        DNDz.__init__(self,*args,**kwargs)

        #don't need to map this guy
        self.nomap = True
        self.loadDNDz()

    def loadDNDz(self):
        """
        Loads the data from the file specified
        """

        tab = np.genfromtxt(self.fname)
        self.dndz = np.zeros((tab.shape[0], self.ncuts))

        lowz = tab[:,self.lowzcol]
        highz = tab[:,self.highzcol]
        self.zbins = np.zeros(len(lowz)+1)
        self.zbins[:-1] = lowz
        self.zbins[-1] = highz[-1]
        for i in range(self.ncuts):
            self.dndz[:,i] = tab[:,i+self.ycol]
