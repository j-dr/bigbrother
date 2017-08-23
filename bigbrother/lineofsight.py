from __future__ import print_function, division
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import numpy as np

from metric import Metric, jackknifeMap
from .selection import Selector

class DNDz(Metric):

    def __init__(self, ministry, zbins=None, magbins=None,
                  catalog_type=['galaxycatalog'], tag=None,
                  appmag=True, lower_limit=True, cutband=None,
                  normed=True, selection_dict=None, CMASS=False,
                  **kwargs):
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
        Metric.__init__(self, ministry, tag=tag, **kwargs)

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
            self.magbins = [None]
            self.nmagbins = 0
            self.nomags = True
        else:
            self.nomags = False
            self.magbins = magbins
            if self.lower_limit:
                self.nmagbins = len(self.magbins)
            else:
                self.nmagbins = len(self.magbins) - 1

        self.CMASS = CMASS

        self.normed = normed
        self.aschema = 'galaxyonly'

        if self.nmagbins > 0:
            self.mapkeys = [self.mkey, 'redshift']
            self.unitmap = {self.mkey :'mag'}
        else:
            self.mapkeys = ['redshift']
            self.unitmap = {}

        #Make selection dict here
        if (selection_dict is None) & lower_limit:
            selection_dict = {'mag':{'selection_type':'cut1d',
                                    'mapkeys':['appmag'],
                                    'bins':self.magbins,
                                    'selection_ind':self.cutband,
                                    'lower':True}}
        elif (selection_dict is None):
            selection_dict = {'mag':{'selection_type':'binned1d',
                                    'mapkeys':['appmag'],
                                    'bins':self.magbins,
                                    'selection_ind':self.cutband}}

        nmkeys = []
        for s in selection_dict:
            if 'mapkeys' in selection_dict[s]:
                ss = selection_dict[s]
                for m in ss['mapkeys']:
                    if m not in self.mapkeys:
                        self.mapkeys.append(m)
                    if m not in self.unitmap:
                        self.unitmap[m] = self.defaultUnits(m)

        if self.CMASS:
            if 'appmag' not in self.mapkeys:
                self.mapkeys.append('appmag')
                self.unitmap['appmag'] = 'mag'

        self.zcounts = None
        self.selector = Selector(selection_dict)

    @jackknifeMap
    def map(self, mapunit):
        """
        Map function for dn/dz. Extracts relevant information from a mapunit. Usually only called by a Ministry object, not manually.
        """
        #This will eventually need to be replaced with the outputs
        #if selector.mapArray()
        if self.zcounts is None:
            self.zcounts = np.zeros((self.njack, self.nzbins,self.nmagbins))

        if not self.CMASS:
            for idx, aidx in self.selector.generateSelections(mapunit):
                c, e = np.histogram(mapunit['redshift'][idx], bins=self.zbins)
                shp = [1 for i in range(len(aidx)+1)]
                shp[1] = len(c)
                self.zcounts[self.jcount,:,aidx] += c.reshape(shp)
        else:
            idx = self.selectCMASS(mapunit['appmag'])
            c, e = np.histogram(mapunit['redshift'][idx], bins=self.zbins)
            self.zcounts[self.jcount,:,0] = c
            
    def reduce(self, rank=None, comm=None):
        """
        Converts extracted redshift information into a (normalized) dn/dz output and stores it as an attribute of the metric.
        """
        if rank is not None:
            gzcounts = comm.gather(self.zcounts, root=0)


            if rank==0:
                gshape = [self.zcounts.shape[i] for i in range(len(self.zcounts.shape))]
                gshape[0] = self.njacktot

                self.zcounts = np.zeros(gshape)
                jc = 0

                for g in gzcounts:
                    if g is None: continue
                    nj = g.shape[0]
                    self.zcounts[jc:jc+nj,:,:] = g

                    jc += nj

                if self.jtype is not None:
                    area = self.ministry.galaxycatalog.getArea(jackknife=True).reshape(self.njacktot,1,1)
                else:
                    area = self.ministry.galaxycatalog.getArea(jackknife=False)

                if self.normed:
                    dz = (self.zbins[1:]-self.zbins[:-1]).reshape((1,self.zcounts.shape[1],1))
                    jzcounts = self.jackknife(self.zcounts, reduce_jk=False)
                    self.jdndz = jzcounts/area/dz
                else:
                    jzcounts = self.jackknife(self.zcounts, reduce_jk=False)
                    self.jdndz = jzcounts/area

                self.dndz = np.sum(self.jdndz, axis=0) / self.njacktot
                self.vardndz = np.sum( (self.jdndz - self.dndz) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot

        else:
            if self.jtype is not None:
                area = self.ministry.galaxycatalog.getArea(jackknife=True).reshape(self.njacktot,1,1)
            else:
                area = self.ministry.galaxycatalog.getArea(jackknife=False).reshape(self.njacktot,1,1)                

            if self.normed:
                dz = (self.zbins[1:]-self.zbins[:-1]).reshape((1,self.zcounts.shape[1],1))
                jzcounts = self.jackknife(self.zcounts, reduce_jk=False)
                self.jdndz = jzcounts/area/dz
            else:
                jzcounts = self.jackknife(self.zcounts, reduce_jk=False)
                self.jdndz = jzcounts/area

            self.dndz = np.sum(self.jdndz, axis=0) / self.njacktot
            self.vardndz = np.sum( (self.jdndz - self.dndz) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot

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
            sax.set_ylabel(r'$\frac{dN}{dZ}\, [deg^{-2}]$', labelpad=40)
            sax.set_xlabel(r'$z$', labelpad=40)

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

        if (labels is not None) and (len(labels)==len(lines)):
            f.legend(lines, labels, 'best')

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax

class PeakDNDz(DNDz):

    def __init__(self, ministry, **kwargs):

        if 'zbins' not in kwargs.keys():
            kwargs['zbins'] = np.linspace(ministry.minz, ministry.maxz, 60)


        if 'magbins' not in kwargs.keys():
            kwargs['magbins'] = np.linspace(19.5, 22, 30)

        DNDz.__init__(self, ministry, **kwargs)


    def reduce(self, rank=None, comm=None):

        DNDz.reduce(self, rank=rank, comm=comm)

        if (rank is not None):
            if (rank==0):
                self.jzpeak = self.zbins[np.argmax(self.jdndz, axis=1)]
                
                self.zpeak = np.sum(self.jzpeak, axis=0) / self.njacktot
                self.varzpeak = np.sum((self.jzpeak-self.zpeak)**2, axis=0) * (self.njacktot - 1) / self.njacktot
        else:
            self.jzpeak = self.zbins[np.argmax(self.jdndz, axis=1)]

            self.zpeak = np.sum(self.jzpeak, axis=0) / self.njacktot
            self.varzpeak = np.sum((self.jzpeak-self.zpeak)**2, axis=0) * (self.njacktot - 1) / self.njacktot



    def visualize(self, xlabel=None, ylabel=None, compare=False,
                    ax=None, f=None, plotname=None, **kwargs):

        if f is None:
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
            if ylabel is None:
                sax.set_ylabel(r'$Peak of \frac{dN}{dZ}\, [deg^{-2}]$')
            else:
                sax.set_xlabel(xlabel)

            if xlabel is None:
                sax.set_xlabel(r'$mag$')
            else:
                sax.set_ylabel(xlabel)

            l1 = ax[0].errorbar(self.magbins, self.zpeak, yerr=np.sqrt(self.varzpeak), **kwargs)

        #plt.tight_layout()

        if (plotname is not None) and (not compare):
            plt.savefig(plotname)

        return f, ax, l1


    def compare(self, othermetrics, labels=None, plotname=None):
        tocompare = [self]
        tocompare.extend(othermetrics)

        lines = []

        for i, m in enumerate(tocompare):
            if i==0:
                f, ax, l1 = m.visualize(compare=True,
                                         **kwargs)
            else:
                f, ax, l1 = m.visualize(compare=True,
                                          f=f, ax=ax, **kwargs)
            lines.extend(l1)

        if labels[0]!=None:
            f.legend(lines, labels, 'best')

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax


class MedianDNDz(DNDz):

    def __init__(self, ministry, **kwargs):

        if 'zbins' not in kwargs.keys():
            kwargs['zbins'] = np.linspace(ministry.minz, ministry.maxz, 60)


        if 'magbins' not in kwargs.keys():
            kwargs['magbins'] = np.linspace(19.5, 22, 30)

        DNDz.__init__(self, ministry, **kwargs)


    def reduce(self, rank=None, comm=None):

        DNDz.reduce(self, rank=rank, comm=comm)

        if (rank is not None):
            if (rank==0):
                self.jcdf = np.cumsum(self.jdndz, axis=1)
                self.jcdf = self.jcdf/self.jcdf[:,-1,...].reshape(self.njacktot, 1, self.nmagbins)
                self.jzmedian = np.zeros((self.njacktot, self.nmagbins))

                for i in range(self.njacktot):
                    for j in range(self.nmagbins):
                        self.jzmedian[i,j] = self.zbins[self.jcdf[i,:,j].searchsorted(0.5)]
                
                self.zmedian = np.sum(self.jzmedian, axis=0) / self.njacktot
                self.varzmedian = np.sum((self.jzmedian-self.zmedian)**2, axis=0) * (self.njacktot - 1) / self.njacktot
        else:
            self.jcdf = np.cumsum(self.jdndz, axis=1)
            self.jcdf = self.jcdf/self.jcdf[:,-1,...].reshape(self.njacktot, 1, self.nmagbins)
            self.jzmedian = np.zeros((self.njacktot, self.nmagbins))

            for i in range(self.njacktot):
                for j in range(self.nmagbins):
                    self.jzmedian[i,j] = self.zbins[self.jcdf[i,:,j].searchsorted(0.5)]
                
            self.zmedian = np.sum(self.jzmedian, axis=0) / self.njacktot
            self.varzmedian = np.sum((self.jzmedian-self.zmedian)**2, axis=0) * (self.njacktot - 1) / self.njacktot

    def visualize(self, xlabel=None, ylabel=None, compare=False,
                    ax=None, f=None, plotname=None, **kwargs):

        if f is None:
            f, ax = plt.subplots(1, figsize=(15,15))
            ax = np.atleast_1d(ax)
            newaxes = True
        else:
            newaxes = False

        if newaxes:
            sax = f.add_subplot(111)
            plt.setp(sax.get_xticklines(), visible=False)
            plt.setp(sax.get_yticklines(), visible=False)
            plt.setp(sax.get_xticklabels(), visible=False)
            plt.setp(sax.get_yticklabels(), visible=False)
            sax.patch.set_alpha(0.0)
            sax.patch.set_facecolor('none')
            sax.spines['top'].set_color('none')
            sax.spines['bottom'].set_color('none')
            sax.spines['left'].set_color('none')
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            if ylabel is None:
                sax.set_ylabel(r'Median of $\frac{dN}{dZ}\, [deg^{-2}]$', fontsize=16, labelpad=20)
            else:
                sax.set_ylabel(xlabel, fontsize=16, labelpad=20)

            if xlabel is None:
                sax.set_xlabel(r'$mag$', fontsize=16, labelpad=20)
            else:
                sax.set_xlabel(xlabel, fontsize=16, labelpad=20)

        l1 = ax[0].plot(self.magbins, self.zmedian,**kwargs)
        ax[0].fill_between(self.magbins, self.zmedian - np.sqrt(self.varzmedian),
                            self.zmedian + np.sqrt(self.varzmedian),**kwargs)
        ax[0].tick_params(axis='x', labelsize=16)
        ax[0].tick_params(axis='y', labelsize=16)

        #plt.tight_layout()

        if (plotname is not None) and (not compare):
            plt.savefig(plotname)

        return f, ax, l1


    def compare(self, othermetrics, labels=None, plotname=None, **kwargs):
        tocompare = [self]
        tocompare.extend(othermetrics)

        lines = []

        for i, m in enumerate(tocompare):
            if i==0:
                f, ax, l1 = m.visualize(compare=True,color=self._color_list[i],
                                         **kwargs)
            else:
                f, ax, l1 = m.visualize(compare=True,color=self._color_list[i],
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
