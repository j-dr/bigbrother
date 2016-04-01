from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.integrate import quad
from copy import copy
if __name__=='__main__':
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
import numpy as np
import treecorr


class Metric(object):
    """
    An abstract class which all other metric classes should descend from. All metrics
    must define the methods that are declared as abstractmethod here. 
    """
    __metaclass__ = ABCMeta

    def __init__(self, simulation):
        """
        Simple init method. At the very least, init methods for subclasses
        should take a simulation object as an argument.
        """
        self.sim = simulation

    @abstractmethod
    def map(self, mapunit):
        pass

    @abstractmethod
    def reduce(self):
        pass

    @abstractmethod
    def visualize(self, plotname=None):
        pass

    @abstractmethod
    def compare(self, othermetric, plotname=None):
        pass

class GMetric(Metric):
    """
    A generic metric class that deals with measurements made over
    multiple measurements of similar quantities, i.e. mags in
    different bands or mass functions using various mass definitions
    """

    def __init__(self, simulation, zbins=None, xbins=None):
        """
        Initialize a MagnitudeMetric object. Note, all metrics should define
        an attribute called mapkeys which specifies the types of data that they
        expect.

        Arguments
        ---------
        simulation : Simulation
            The simulation object that this metric is associated with.
        zbins : array-like
            An 1-d array containing the edges of the redshift bins to 
            measure the LF in.
        xbins : array-like
            A 1-d array containing the edges of the magnitude bins to 
            measure the metric in.
        """
        Metric.__init__(self, simulation)

        if zbins is None:
            self.zbins = zbins
            self.nzbins = 1
        else:
            self.nzbins = len(zbins)-1
            self.zbins = np.array(zbins)

        if xbins is None:
            self.xbins = np.linspace(-25, -11, 30)
        else:
            self.xbins = xbins

    @abstractmethod
    def map(self, mapunit):
        pass

    @abstractmethod
    def reduce(self):
        pass

    def visualize(self, plotname=None, usecols=None, usez=None,fracdev=True, 
                  ref_y=None, ref_x=[None], xlim=None, ylim=None, fylim=None, 
                  f=None, ax=None, xlabel=None,ylabel=None,**kwargs):
        """
        Plot the calculated metric.
        
        Arguments
        ---------
        plotname : string, optional
            If provided, the plot will be saved to a file by this name.
        usecols : array-like, optional
            The indices of the bands to plot. Default is to use all the 
            bands the metric was measured in.
        fracdev : bool, optional
            Whether or not to plot fractional deviation of this metric
            from the reference metric function provided in
            ref_y. 
        ref_y : 3-d array-like, optional
            If fracdev is set to True, this is the metric that will be compared.
            If must have the same number of bands and z bins, but the 
            magnitude bins may differ.
        ref_x : 1-d array-like, optional
            The mean magnitudes of the bins in which ref_x is measured.
            If ref_x is measured in a different number of magnitude bins
            than this metric, interpolation is performed in order to compare at 
            the same mean magnitudes.
        xlim : array-like, optional
            A list [xmin, xmax], the range of magnitudes to
            plot the metric for.
        ylim: array-like, optional
            A list [ymin, ymax], the range of y values to plot
            the metric for.
        fylim : array-like, optional
            A list [fymin, fymax]. If fracdev is True, plot the 
            fractional deviations over this range.
        f : Figure, optional
           A figure object. If provided the metric will be ploted using this 
           figure.
        ax : array of Axes, optional
           An array of Axes objects. If provided, the metrics
           will be plotted on these axes. 
        """
        if hasattr(self, 'xmean'):
            mxs = self.xmean
        else:
            mxs = np.array([(self.xbins[i]+self.xbins[i+1])/2 
                              for i in range(len(self.xbins)-1)])

        if usecols is None:
            usecols = range(self.nbands)

        if usez is None:
            usez = range(self.nzbins)

        nzbins = len(usez)

        #If want to plot fractional deviations, and ref_y
        #uses different magnitude bins, interpolate ref_y to 
        #magniutdes given at mxs. Don't extrapolate!
        if fracdev & ((len(ref_x)!=len(mxs)) | ((ref_x[0]!=mxs[0]) | (ref_x[-1]!=mxs[-1]))):
            rls = ref_y.shape
            li = mxs.searchsorted(ref_x[0])
            hi = mxs.searchsorted(ref_x[-1])
            iref_y = np.zeros((hi-li, rls[1], rls[2]))

            for i in range(rls[1]):
                for j in range(rls[2]):
                    spl = InterpolatedUnivariateSpline(ref_x, ref_y[:,i,j])
                    iref_y[:,i,j] = spl(mxs[li:hi])

            ref_y = iref_y

        else:
            li = 0
            hi = len(mxs)

        #if no figure provided, set up figure and axes
        if f is None:
            if fracdev==False:
                f, ax = plt.subplots(len(usecols), nzbins,
                                     sharex=True, sharey=True, figsize=(8,8))
            #if want fractional deviations, need to make twice as 
            #many rows of axes. Every other row contains fractional
            #deviations from the row above it.
            else:
                assert(ref_y!=None)
                gs = gridspec.GridSpec(len(usecols)*2, nzbins)
                f = plt.figure()
                ax = []
                for r in range(len(usecols)):
                    ax.append([])
                    ax.append([])
                    for c in range(nzbins):
                        if (r==0) & (c==0):
                            ax[2*r].append(f.add_subplot(gs[2*r,c]))
                            ax[2*r+1].append(f.add_subplot(gs[2*r+1,c], sharex=ax[0][0]))
                        else:
                            ax[2*r].append(f.add_subplot(gs[2*r,c]))
                            ax[2*r+1].append(f.add_subplot(gs[2*r+1,c], sharex=ax[0][0], 
                                                           sharey=ax[1][0]))
            newaxes = True
        else:
            newaxes = False

        if nzbins>1:
            for i, b in enumerate(usecols):
                for j in range(nzbins):
                    if fracdev==False:
                        l1 = ax[i][j].semilogy(mxs, self.y[:,b,j], 
                                          **kwargs)
                    else:
                        l1 = ax[2*i][j].semilogy(mxs, self.y[:,b,j], 
                                          **kwargs)
                        ax[2*i+1][j].plot(mxs[li:hi], 
                                          (self.y[li:hi,b,j]-ref_y[:,b,j])\
                                              /ref_y[:,b,j], **kwargs)
                        if (i==0) & (j==0):
                            if xlim!=None:
                                ax[0][0].set_xlim(xlim)
                            if ylim!=None:
                                ax[0][0].set_ylim(ylim)
                            if fylim!=None:
                                ax[1][0].set_ylim(fylim)

        else:
            for i, b in enumerate(usecols):
                if fracdev==False:
                    try:
                        l1 = ax[i].semilogy(mxs, self.y[:,b,0], 
                                            **kwargs)
                    except:
                        l1 = ax.semilogy(mxs, self.y[:,b,0], 
                                         **kwargs)                        
                else:
                    l1 = ax[2*i][0].semilogy(mxs, self.y[:,b,0], 
                                        **kwargs)
                    ax[2*i+1][0].plot(mxs[li:hi], (self.y[li:hi,b,0]-ref_y[:,b,0])\
                                      /ref_y[:,b,0], **kwargs)

                    if (i==0):
                        if xlim!=None:
                            ax[0][0].set_xlim(xlim)
                        if ylim!=None:
                            ax[0][0].set_ylim(ylim)
                        if fylim!=None:
                            ax[1][0].set_ylim(fylim)

        #if we just created the axes, add labels
        if newaxes:
            sax = f.add_subplot(111)
            sax.spines['top'].set_color('none')
            sax.spines['bottom'].set_color('none')
            sax.spines['left'].set_color('none')
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            sax.set_xlabel(r'%s' % xlabel)
            sax.set_ylabel(r'%s' % ylabel)

        if plotname!=None:
            plt.savefig(plotname)

        return f, ax, l1
        

    def compare(self, othermetrics, plotname=None, usecols=None, fracdev=True, xlim=None,
                ylim=None, fylim=None, labels=None, **kwargs):
        """
        Compare a list of other metrics of the same type

        Arguments
        ---------
        othermetrics -- array-like of MagnitudeMetrics
            An array of luminosity functions to compare to this one.
        labels -- array-like, optional
           An array containing labels for each MagnitudeMetric in 
           othermetrics. Defualt to no labels.
        See visualize for other argument documentation
        """

        tocompare = [self]
        tocompare.extend(othermetrics)

        if usecols!=None:
            if not hasattr(usecols[0], '__iter__'):
                usecols = [usecols]*len(tocompare)
            else:
                assert(len(usecols)==len(tocompare))
        else:
            usecols = [None]*len(tocompare)

        if fracdev:
            if hasattr(self, 'xmean'):
                ref_x = self.xmean
            else:
                ref_x =  np.array([(self.xbins[i]+self.xbins[i+1])/2 
                                    for i in range(len(self.xbins)-1)])

        if labels is None:
            labels = [None]*len(tocompare)
        
        lines = []

        for i, m in enumerate(tocompare):
            if usecols[i]!=None:
                assert(len(usecols[0])==len(usecols[i]))
            if i==0:
                if fracdev:
                    f, ax, l = m.visualize(usecols=usecols[i], fracdev=True, ref_x=ref_x,
                                        ref_y=self.y, xlim=xlim,
                                        ylim=ylim, fylim=fylim, label=labels[i],**kwargs)

                else:
                    f, ax, l = m.visualize(usecols=usecols[i], xlim=xlim, ylim=ylim, 
                                        fracdev=False, fylim=fylim,label=labels[i],**kwargs)
            else:
                if fracdev:
                    f, ax, l = m.visualize(usecols=usecols[i], fracdev=True, ref_x=ref_x,
                                        ref_y=tocompare[0].y, 
                                        xlim=xlim, ylim=ylim, fylim=fylim,
                                        f=f, ax=ax, label=labels[i], **kwargs)
                else:
                    f, ax, l = m.visualize(usecols=usecols[i], xlim=xlim, ylim=ylim,
                                        fylim=fylim, f=f, ax=ax, fracdev=False,
                                        label=labels[i], **kwargs)
            lines.append(l[0])

        if labels[0]!=None:
            f.legend(lines, labels)

        if plotname!=None:
            plt.savefig(plotname)

        return f, ax

class MagnitudeMetric(GMetric):
    """
    Restruct GMetric to magnitudes
    """
    
    def __init__(self, simulation, zbins=None, magbins=None):
        """
        Initialize a MagnitudeMetric object. Note, all metrics should define
        an attribute called mapkeys which specifies the types of data that they
        expect.

        Arguments
        ---------
        simulation : Simulation
            The simulation object that this metric is associated with.
        zbins : array-like
            An 1-d array containing the edges of the redshift bins to 
            measure the LF in.
        xbins : array-like
            A 1-d array containing the edges of the magnitude bins to 
            measure the metric in.
        """



        if magbins is None:
            xbins = np.linspace(-25,-15, 40)

        self.magbins = xbins

        GMetric.__init__(self, simulation, zbins=zbins, xbins=xbins)

class MassMetric(GMetric):
    """
    Restruct GMetric to magnitudes
    """
    
    def __init__(self, simulation, zbins=None, massbins=None):
        """
        Initialize a MassMetric object. Note, all metrics should define
        an attribute called mapkeys which specifies the types of data that they
        expect.

        Arguments
        ---------
        simulation : Simulation
            The simulation object that this metric is associated with.
        zbins : array-like
            An 1-d array containing the edges of the redshift bins to 
            measure the metric in.
        massbins : array-like
            A 1-d array containing the edges of the mass bins to 
            measure the metric in.
        """

        if massbins is None:
            xbins = np.linspace(10, 16, 40)

        self.massbins = xbins

        GMetric.__init__(self, simulation, zbins=zbins, xbins=xbins)

class LuminosityFunction(MagnitudeMetric):
    """
    A generic luminosity function class. More specific types of luminosity
    functions inherit this class.
    """

    def __init__(self, simulation, central_only=False, zbins=None, magbins=None):
        """
        Initialize a LuminosityFunction object. Note, all metrics should define
        an attribute called mapkeys which specifies the types of data that they
        expect.

        Arguments
        ---------
        simulation : Simulation
            The simulation object that this metric is associated with.
        central_only : bool, optional
            Whether the LF should be measured for only central galaxies.
            Defaults to false
        zbins : array-like
            An 1-d array containing the edges of the redshift bins to 
            measure the LF in.
        magbins : array-like
            A 1-d array containing the edges of the luminosity bins to 
            measure the LF in.
        """

        if zbins is None:
            zbins = [0.0, 0.2]

        if magbins is None:
            magbins = np.linspace(-25, -11, 30)

        MagnitudeMetric.__init__(self, simulation, zbins=zbins, magbins=magbins)

        self.central_only = central_only
        if central_only:
            self.mapkeys = ['luminosity', 'redshift', 'central']
        else:
            self.mapkeys = ['luminosity', 'redshift']
        
    def map(self, mapunit):
        """
        A simple example of what a map function should look like. 
        Map functions always take mapunits as input.
        """

        #The number of bands to measure the LF for
        self.nbands = mapunit['luminosity'].shape[1]

        #If only measuring for centrals, get the appropriate
        #rows of the mapunit

        mu = {}
        if self.central_only:
            for k in mapunit.keys():
                mu[k] = mapunit[k][mapunit['central']==1]
        else:
            mu = mapunit

        #Want to count galaxies in bins of luminosity for 
        #self.nbands different bands in self.nzbins
        #redshift bins
        if not hasattr(self, 'lumcounts'):
            self.lumcounts = np.zeros((len(self.magbins)-1, self.nbands, 
                                       self.nzbins))

        #Assume redshifts are provided, and that the 
        #mapunit is sorted in terms of them
        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mu['redshift'].searchsorted(self.zbins[i])
            zhidx = mu['redshift'].searchsorted(self.zbins[i+1])

            #Count galaxies in bins of luminosity
            for j in range(self.nbands):
                c, e = np.histogram(mu['luminosity'][zlidx:zhidx,j], 
                                    bins=self.magbins)
                self.lumcounts[:,j,i] += c

    def reduce(self):
        """
        Given counts in luminosity bins, generate a luminosity function.
        This will be called after all the mapunits are mapped by the map
        method. This turns total counts of galaxies into densities as appropriate
        for a luminosity function. The LF is then saved as an attribute of the
        LuminosityFunction object.
        """
        area = self.sim.galaxycatalog.getArea()
        self.luminosity_function = self.lumcounts

        for i in range(self.nzbins):
            vol = self.sim.calculate_volume(area, self.zbins[i], self.zbins[i+1])
            self.luminosity_function[:,:,i] /= vol

        self.y = self.luminosity_function


    def integrate(self, lmin, lmax, z, band=1):
        """
        Integrate the luminosity function between
        lmin and lmax at a particular redshift.
        """

        if not hasattr(self, 'lummean'):
            self.magmean = np.array([(self.magbins[i]+self.magbins[i+1])/2 
                                     for i in range(len(self.magbins)-1)])
        if not hasattr(self, 'zmean'):
            self.zmean = np.array([(self.zbins[i]+self.zbins[i+1])/2
                                   for i in range(len(self.zbins)-1)])

        if not hasattr(self, 'integ_spline'):
            self.mv_spline = RectBivariateSpline(self.magmean, self.zmean, 
                                                    self.luminosity_function[:,band,:])

        uvspl = lambda l : self.mv_spline(l, z)

        n = quad(uvspl, lmin, lmax)

        return n[0]

class MagCounts(MagnitudeMetric):
    """
    Galaxy counts per magnitude.
    """

    def __init__(self, simulation, zbins=[0.0, 0.2],  magbins=None):

        if magbins is None:
            magbins = np.linspace(10, 30, 60)

        MagnitudeMetric.__init__(self,simulation, zbins=zbins, magbins=magbins)

        self.mapkeys = ['appmag', 'redshift']

    def map(self, mapunit):
        self.nbands = mapunit['appmag'].shape[1]

        if not hasattr(self, 'magcounts'):
            self.magcounts = np.zeros((len(self.magbins)-1, 
                                       self.nbands, self.nzbins))
            
        if self.zbins!=None:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])
                for j in range(self.nbands):
                    c, e = np.histogram(mapunit['appmag'][zlidx:zhidx,j], 
                                        bins=self.magbins)
                    self.magcounts[:,j,i] += c
        else:
            for j in range(self.nbands):
                c, e = np.histogram(mapunit['appmag'][:,j], bins=self.magbins)
                self.magcounts[:,j,0] += c

    def reduce(self):
        area = self.sim.galaxycatalog.getArea()
        self.magcounts = self.magcounts/area
        self.y = self.magcounts

        
class LcenMvir(Metric):
    """
    Central galaxy luminosity - halo virial mass relation.
    """
    def __init__(self, simulation, zbins=None, massbins=None):
        Metric.__init__(self, simulation)

        if zbins is None:
            self.zbins = [0.0, 0.2]
        else:
            self.zbins = zbins
            self.zbins = np.array(self.zbins)

        self.nzbins = len(self.zbins)-1

        if massbins is None:
            self.massbins = np.logspace(12, 15, 20)
        else:
            self.massbins = massbins

        self.mapkeys = ['luminosity', 'redshift', 'central', 'mvir']

        
    def map(self, mapunit):

        self.nbands = mapunit['luminosity'].shape[1]

        mu = {}

        for k in mapunit.keys():
            mu[k] = mapunit[k][mapunit['central']==1]


        if not hasattr(self, 'lumcounts'):
            self.totlum = np.zeros((len(self.massbins)-1, self.nbands, 
                                    len(self.zbins)-1))
            self.bincount = np.zeros((len(self.massbins)-1, self.nbands,
                                      len(self.zbins)-1))
            
        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mu['redshift'].searchsorted(self.zbins[i])
            zhidx = mu['redshift'].searchsorted(self.zbins[i+1])
            mb = np.digitize(mu['mvir'][zlidx:zhidx], bins=self.massbins)

            for j in range(len(self.massbins)-1):
                blum = mu['luminosity'][zlidx:zhidx,:][mb==j]
                self.bincount[j,:,i] += len(blum)
                self.totlum[j,:,i] += np.sum(blum, axis=0)


    def reduce(self):

        self.lcen_mvir = self.totlum/self.bincount


    def visualize(self, plotname=None, f=None, ax=None, usebands=None, **kwargs):

        if hasattr(self, 'massmean'):
            mmass = self.massmean
        else:
            mmass = np.array([(self.massbins[i]+self.massbins[i+1])/2 
                              for i in range(len(self.massbins)-1)])

        if usebands is None:
            usebands = range(self.nbands)

        if f is None:
            f, ax = plt.subplots(len(usebands), self.nzbins,
                                 sharex=True, sharey=True,
                                 figsize=(8,8))
            newaxes = True
        else:
            newaxes = False

        if self.nzbins>1:
            for i, b in enumerate(usebands):
                for j in range(self.nzbins):
                    ax[i][j].semilogx(mmass, self.lcen_mvir[:,b,j], 
                                      **kwargs)
        else:
            for i, b in enumerate(usebands):
                for j in range(self.nzbins):
                    ax[i].semilogx(mmass, self.lcen_mvir[:,b,j], 
                                   **kwargs)

        if newaxes:
            sax = f.add_subplot(111)
            sax.spines['top'].set_color('none')
            sax.spines['bottom'].set_color('none')
            sax.spines['left'].set_color('none')
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            sax.set_xlabel(r'$M_{vir}\, [M_{sun}]$')
            sax.set_ylabel(r'$L_{cen}\, [mag]$')

        if plotname!=None:
            plt.savefig(plotname)

        return f, ax


    def compare(self, othermetrics, plotname=None, usebands=None, **kwargs):
        tocompare = [self]
        tocompare.extend(othermetrics)

        if usebands!=None:
            if not hasattr(usebands[0], '__iter__'):
                usebands = [usebands]*len(tocompare)
            else:
                assert(len(usebands)==len(tocompare))
        else:
            usebands = [None]*len(tocompare)
        
        for i, m in enumerate(tocompare):
            if usebands[i]!=None:
                assert(len(usebands[0])==len(usebands[i]))
            if i==0:
                f, ax = m.visualize(usebands=usebands[i], **kwargs)
            else:
                f, ax = m.visualize(usebands=usebands[i],
                                    f=f, ax=ax, **kwargs)

        if plotname!=None:
            plt.savefig(plotname)

        return f, ax


class ColorColor(Metric):
    """
    Color-color diagram.
    """
    def __init__(self, simulation, zbins=[0.0, 0.2], magbins=None):
        Metric.__init__(self, simulation)
        
        self.zbins = zbins
        if zbins is None:
            self.nzbins = 1
        else:
            self.nzbins = len(zbins)-1
            self.zbins = np.array(self.zbins)

        if magbins is None:
            self.magbins = np.linspace(10, 30, 60)
        else:
            self.magbins = magbins

        self.mapkeys = ['appmag', 'redshift']

    def map(self, mapunit):
        self.nbands = mapunit['appmag'].shape[1]

        if not hasattr(self, 'cc'):
            self.cc = np.zeros((len(self.magbins)-1, len(self.magbins)-1, 
                                self.nbands*(self.nbands-1)/2,
                                self.nzbins))

        if self.zbins!=None:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])
                for j in range(self.nbands):
                    for k in range(self.nbands):
                        if k<=j: continue
                        ind = k*(k-1)/2+j-1
                        c, e0, e1 = np.histogram2d(mapunit['appmag'][zlidx:zhidx,j], 
                                                   mapunit['appmag'][zlidx:zhidx,k], 
                                                   bins=[self.magbins,self.magbins])
                        self.cc[:,:,ind,i] += c
        else:
            for j in range(self.nbands):
                for k in range(self.nbands):
                    if k<=j: continue
                    ind = k*(k-1)/2+j-1
                    c, e0, e1 = np.histogram2d(mapunit['appmag'][:,j], 
                                               mapunit['appmag'][:,k], 
                                               bins=[self.magbins,self.magbins])
                    self.cc[:,:,ind,0] += c


    
    def reduce(self):
        area = self.sim.galaxycatalog.getArea()
        self.cc = self.cc/area

    def visualize(self, plotname=None, f=None, ax=None, usecolors=None, **kwargs):

        if hasattr(self, 'magmean'):
            mmags = self.magmean
        else:
            mmags = np.array([(self.magbins[i]+self.magbins[i+1])/2 
                              for i in range(len(self.magbins)-1)])

        if usecolors is None:
            usecolors = range(self.cc.shape[2])

        if f is None:
            f, ax = plt.subplots(self.nzbins, len(usecolors),
                                 sharex=True, sharey=True, figsize=(8,8))
            newaxes = True
        else:
            newaxes = False

        for i in usecolors:
            for j in range(self.nzbins):
                ax[j][i].pcolormesh(self.magbins, self.cbins, self.cc[::-1,::-1,i,j],
                                    **kwargs)

        return f, ax

    def compare(self, othermetric, plotname=None, usecolors=None, **kwargs):
        if usecolors!=None:
            assert(len(usecolors[0])==len(usecolors[1]))
            f, ax = self.visualize(usecolors=usecolors[0], **kwargs)
            f, ax = othermetric.visualize(usecolors=usecolors[1],
                                          f=f, ax=ax, **kwargs)
        else:
            f, ax = self.visualize(usecolors=usecolors, **kwargs)
            f, ax = othermetric[0].visualize(usecolors=usecolors,
                                          f=f, ax=ax, **kwargs)

        if plotname!=None:
            plt.savefig(plotname)



class ColorMagnitude(Metric):
    """
    Color-magnitude diagram.
    """
    def __init__(self, simulation, zbins=[0.0, 0.2], magbins=None, 
                 cbins=None, central_only=False, logscale=False):

        Metric.__init__(self, simulation)
        
        self.zbins = zbins
        if zbins is None:
            self.nzbins = 1
        else:
            self.nzbins = len(zbins)-1
            self.zbins = np.array(self.zbins)

        if (magbins is None) & (cbins is None):
            self.magbins = np.linspace(-25, -19, 60)
            self.cbins = np.linspace(-1,2,60)
        elif magbins is None:
            self.magbins = np.linspace(-25,-19,60)
            self.cbins = cbins
        elif cbins is None:
            self.magbins = magbins
            self.cbins = np.linspace(-1,2,60)
        else:
            self.magbins = magbins
            self.cbins = cbins

        self.central_only = central_only
        self.logscale = logscale

        if central_only:
            self.mapkeys = ['appmag', 'redshift', 'central']
        else:
            self.mapkeys = ['appmag', 'redshift']

    def map(self, mapunit):
        self.nbands = mapunit['appmag'].shape[1]

        mu = {}
        if self.central_only:
            for k in mapunit.keys():
                mu[k] = mapunit[k][mapunit['central']==1]
        else:
            mu = mapunit


        if not hasattr(self, 'cc'):
            self.cc = np.zeros((len(self.magbins)-1, len(self.cbins)-1, 
                                self.nbands*(self.nbands-1)/2,
                                self.nzbins))

        if self.zbins!=None:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mu['redshift'].searchsorted(self.zbins[i])
                zhidx = mu['redshift'].searchsorted(self.zbins[i+1])
                for j in range(self.nbands):
                    for k in range(self.nbands):
                        if k<=j: continue
                        ind = k*(k-1)/2+j-1
                        c, e0, e1 = np.histogram2d(mu['appmag'][zlidx:zhidx,j], 
                                                   mu['appmag'][zlidx:zhidx,j] -
                                                   mu['appmag'][zlidx:zhidx,k], 
                                                   bins=[self.magbins,self.cbins])
                        self.cc[:,:,ind,i] += c
        else:
            for j in range(self.nbands):
                for k in range(self.nbands):
                    if k<=j: continue
                    ind = k*(k-1)/2+j-1
                    c, e0, e1 = np.histogram2d(mu['appmag'][:,j], 
                                               mu['appmag'][:,j] -
                                               mu['appmag'][:,k], 
                                               bins=[self.magbins,self.cbins])
                    self.cc[:,:,ind,0] += c


    
    def reduce(self):
        area = self.sim.galaxycatalog.getArea()
        self.cc = self.cc/area

    def visualize(self, plotname=None, f=None, ax=None, usecolors=None, **kwargs):

        
        X, Y = np.meshgrid(self.magbins, self.cbins)
        if self.logscale:
            cc = np.log10(self.cc)
            cc[cc==(-np.inf)] = 0.0
        else:
            cc = self.cc

        if usecolors is None:
            usecolors = range(self.cc.shape[2])

        if f is None:
            f, ax = plt.subplots(self.nzbins, len(usecolors),
                                 sharex=True, sharey=True, figsize=(8,8))
            newaxes = True
        else:
            newaxes = False

        for i in usecolors:
            for j in range(self.nzbins):
                ax[j][i].pcolormesh(X, Y, cc[:,:,i,j],
                                    **kwargs)

        return f, ax

    def compare(self, othermetrics, plotname=None, usecolors=None, **kwargs):
        tocompare = [self]
        tocompare.extend(othermetrics)

        if usecolors!=None:
            if not hasattr(usecolors[0], '__iter__'):
                usecolors = [usecolors]*len(tocompare)
            else:
                assert(len(usecolors)==len(tocompare))
        else:
            usecolors = [None]*len(tocompare)
        
        for i, m in enumerate(tocompare):
            if usecolors[i]!=None:
                assert(len(usecolors[0])==len(usecolors[i]))
            if i==0:
                f, ax = m.visualize(usecolors=usecolors[i], **kwargs)
            else:
                f, ax = m.visualize(usecolors=usecolors[i],
                                    f=f, ax=ax, **kwargs)

        if plotname!=None:
            plt.savefig(plotname)

        return f, ax


class AnalyticLuminosityFunction(LuminosityFunction):
    """
    Class for generating luminosity functions from 
    fitting functions.
    """

    def __init__(self, *args, **kwargs):
        
        if 'nbands' in kwargs:
            self.nbands = kwargs.pop('nbands')
        else:
            self.nbands = 5

        LuminosityFunction.__init__(self,*args,**kwargs)


    def genDSGParams(self, z, evol='faber', Q=-0.866):
        """
        Generate the double schechter function plus gaussian
        parameters used for Buzzard_v1.1
        """
        params = np.zeros(8)
        phistar = 10 ** (-1.79574 + (-0.266409 * z))
        mstar = -20.44
        mstar0 = -20.310

        params[0] = 0.0156  #phistar1
        params[1] = -0.166  #alpha1
        params[2] = 0.00671 #phistar2
        params[3] = -1.523  #alpha2
        params[4] = -19.88  #mstar
        params[5] = 3.08e-5 #phistar3
        params[6] = -21.72  #M_hi
        params[7] = 0.484   #sigma_hi

        phistar_rat = phistar/params[0]
        mr_shift = mstar - mstar0
        params[0] *= phistar_rat
        params[2] *= phistar_rat
        params[5] *= phistar_rat
        params[4] += mr_shift
        params[6] += mr_shift

        if evol=='faber':
            params[4] += Q * (np.log10(z) + 1)
            params[6] += Q * (np.log10(z) + 1)
        elif evol=='a':
            params[4] += Q * (1. / (1 + z) - 1. / 1.1)
            params[6] += Q * (1. / (1 + z) - 1. / 1.1)
        elif evol=='a0':
            params[4] += Q  / (1 + z)
            params[6] += Q  / (1 + z) 

        return params

    def genBCCParams(self, evol='faber', Q=-0.866):
        """
        Generate the parameters for the Buzzard_v1.1 
        luminosity function at specified redshifts assuming
        an evolution model.
        
        Arguments
        ---------
        evol : str
            The evolution model to use
        Q : float
            M* evolution parameter
        """

        zmeans = ( self.zbins[1:] + self.zbins[:-1] ) / 2
        par = np.zeros((len(zmeans), 8))

        for i, z in enumerate(zmeans):
            par[i,:] = self.genDSGParams(z, evol=evol, Q=Q)

        return par

    def evolveDSGParams(self, p, Q, Pe=None, evol='faber'):
        """
        Evolve double schechter function plus gaussian parameters
        in redshift according to supplied Q and Pe values.

        Arguments
        ---------
        p : array-like
            An array of parameters measured at z0 (where z0
            depends on which type of evolution chosen)
        Q : float
            M* evolution parameter
        Pe : float, optional
            phi* evolution parameter
        evol : str, optional
            Evolution model
        """
        zmeans = ( self.zbins[1:] + self.zbins[:-1] ) / 2

        par = np.zeros((len(zmeans), 8))

        for i, z in enumerate(zmeans):
            par[i,:] = copy(p)
            
            if evol=='faber':
                par[i,4] += Q * (np.log10(z) + 1)
                par[i,6] += Q * (np.log10(z) + 1)
            elif evol=='z':
                par[i,4] += Q * (z - 0.1)
                par[i,6] += Q * (z - 0.1)
                if Pe!=None:
                    par[i,0] *= 1 + Pe * (z - 0.3)
                    par[i,2] *= 1 + Pe * (z - 0.3)
                    par[i,5] *= 1 + Pe * (z - 0.3)
            elif evol=='a':
                par[i,4] += Q * (1. / (1 + z) - 1. / 1.1)
                par[i,6] += Q * (1. / (1 + z) - 1. / 1.1)

        return par

    def evolveSFParams(self, p, Q, Pe, evol='z', z0=0.0):
        """
        Evolve double schechter function plus gaussian parameters
        in redshift according to supplied Q and Pe values.

        Arguments
        ---------
        p : array-like
            An array of parameters measured at z0 (where z0
            depends on which type of evolution chosen)
        Q : float
            M* evolution parameter
        Pe : float
            phi* evolution parameter
        evol : str, optional
            Evolution model
        z0 : float
            The redshift about which evolution is determined.
        """
        
        zmeans = ( self.zbins[1:] + self.zbins[:-1] ) / 2
        par = np.zeros((len(zmeans), 3))

        for i, z in enumerate(zmeans):
            par[i,:] = copy(p)
            par[i,0] *= 10 ** (0.4 * Pe * (z - z0))
            par[i,1] -= Q * (z - z0)

        return par
        

    def calcNumberDensity(self,par,form='SchechterAmag'):
        """
        Evaluate an analytic form of a luminosity function
        given an array of parameters.
        
        inputs:
        par -- An array of parameters. If one dimensional,
               assumes the same parameters for all z bins
               and bands. If two dimensional, tries to use
               rows as parameters for different z bins, assumes
               same parameters for all bands.
        form -- The form of the luminosity function. If
                string, must be one of the LFs that are implemented
                otherwise should be a function whose first argument
                is an array of luminosities, and whose second argument
                is a one dimensional array of parameters.
        """
        self.magmean = (self.magbins[1:]+self.magbins[:-1])/2
        self.luminosity_function = np.zeros((len(self.magmean), self.nbands,
                                             self.nzbins))

        for i in range(self.nzbins):
            for j in range(self.nbands):
                if len(par.shape)==1:
                    p = par
                elif ((len(par.shape)==2) and (par.shape[0]==self.nzbins)):
                    p = par[i,:]
                elif ((len(par.shape)==3) and (par.shape[0]==self.nzbins)
                      and (par.shape[2]==self.nbands)):
                    p = par[i,j,:]
                else:
                    raise(ValueError("Shape of parameters incompatible with number of z bins"))
                if form=='SchechterAmag':
                    self.luminosity_function[:,j,i] = self.schechterFunctionAmag(self.magmean, p)
                elif form=='doubleSchecterFunctionAmag':
                    self.luminosity_function[:,j,i] = self.doubleSchechterFunctionAmag(self.magmean, p)
                elif form=='doubleSchecterFunctionVarMS':
                    self.luminosity_function[:,j,i] = self.doubleSchechterFunctionVarMS(self.magmean, p)
                elif form=='doubleSchechterGaussian':
                    self.luminosity_function[:,j,i] = self.doubleSchechterGaussian(self.magmean, p)

                elif hasattr(form, '__call__'):
                    try:
                        self.luminosity_function[:,j,i] = form(self.magmean, p)
                    except:
                        raise(TypeError("Functional form is not in the correct format"))

                else:
                    raise(NotImplementedError("Luminosity form {0} is not implemented".format(form)))
        
        self.y = self.luminosity_function

    def schechterFunctionAmag(self,m,p):
        """
        Single Schechter function appropriate for absolute magnitudes.
        
        inputs:
        m -- An array of magnitudes.
        p -- Schechter function parameters. Order 
             should be phi^{star}, M^{star}, \alpha
        """
        phi = 0.4 * np.log(10) * p[0] * np.exp(-10 ** (0.4 * (p[1]-m))) \
                               * 10 **(0.4*(p[1]-m)*(p[2]+1))
        return phi

    def doubleSchechterFunctionAmag(self,m,p):
        """
        Double Schechter function appropriate for absolute magnitudes.
        
        inputs:
        m -- An array of magnitudes.
        p -- Schechter function parameters. Order 
             should be phi^{star}_{1}, M^{star}, \alpha_{1}, 
             phi^{star}_{2}, \alpha_{2}
        """
        phi = 0.4 * np.log(10) * (p[0] * 10 ** (0.4 * (p[2] + 1) * (p[1] - m)) \
                                  + p[3] * 10 ** (0.4 * (p[4] + 1) * (p[1] - m))) \
                                * np.exp(-10 ** (0.4 * (p[1] - m)))
        return phi

    def doubleSchechterFunctionVarMS(self,m,p):
        """
        Double Schechter with two m_star values
        appropriate for absolute magnitudes.
        
        inputs:
        m -- An array of magnitudes.
        p -- Schechter function parameters. Order 
             should be phi^{star}_{1}, M^{star}_1, \alpha_{1}, 
             phi^{star}_{2}, M^{star}_2, \alpha_{2}
        """
        phi = 0.4 * np.log(10) * (p[0] * 10 ** (0.4 * (p[2] + 1) * (p[1] - m)) \
                                  * np.exp(-10 ** (0.4 * (p[1] - m)))
                                  + p[3] * 10 ** (0.4 * (p[5] + 1) * (p[4] - m)) \
                                  * np.exp(-10 ** (0.4 * (p[4] - m))))
        return phi


    def doubleSchechterGaussian(self,m,p):
        """
        Sum of a double schechter function and a gaussian.
        m -- magnitudes at which to calculate the number density
        p -- Function parameters. Order 
             should be phi^{star}_{1}, M^{star}, \alpha_{1}, 
             phi^{star}_{2}, M^{star}, \alpha_{2}, \phi_{gauss},
             \M_{gauss}, \sigma_{gauss}
        """
        phi = 0.4 * np.log(10) * np.exp(-10**(-0.4 * (m - p[4]))) * \
            (p[0] * 10 ** (-0.4 * (m - p[4])*(p[1]+1)) + \
            p[2] * 10 ** (-0.4 * (m - p[4])*(p[3]+1))) + \
            p[5] / np.sqrt(2 * np.pi * p[7] ** 2) * \
            np.exp(-(m - p[6]) ** 2 / (2 * p[7] ** 2))

        return phi

class TabulatedLuminosityFunction(LuminosityFunction):
    """
    Handle tabulated Luminosity Functions.
    """

    def __init__(self, *args, **kwargs):
        
        if 'fname' in kwargs:
            self.fname = kwargs.pop('fname')
        else:
            raise(ValueError("Please supply a path to the tabulated luminosity function using the fname kwarg!"))

        if 'nbands' in kwargs:
            self.nbands = kwargs.pop('fname')
        else:
            self.nbands = 5

        LuminosityFunction.__init__(self,*args,**kwargs)

    def loadLuminosityFunction(self):
        """
        Read in the LF from self.fname. If self.fname is a list
        assumes that LFs in list correspond to zbins specified.
        If self.fname not a list, if more than 2 columns assumes
        first column is luminosities, second column is.
        """

        if len(self.fname)==1:
            tab = np.loadtxt(self.fname[0])
            self.luminosity_function = np.zeros((tab.shape[0], self.nbands, self.nzbins))
            if len(tab.shape)==2:
                self.magmean = tab[:,0]
                if tab.shape[1]==2:
                    for i in range(self.nzbins):
                        for j in range(self.nbands):
                            self.luminosity_function[:,j,i] = tab[:,1]
                else:
                    assert((tab.shape[1]-1)==self.nzbins)
                    for i in range(self.nzbins):
                        for j in range(self.nbands):
                            self.luminosity_function[:,j,i] = tab[:,i+1]

            elif len(tab.shape)==3:
                self.magmean = tab[:,0,0]
                self.luminosity_function[:,:,:] = tab[:,1:,:]
        else:
            if len(self.fname.shape)==1:
                assert(self.fname.shape[0]==self.nzbins)
                for i in range(len(self.fname)):
                    lf = np.loadtxt(self.fname[i])
                    if i==0:
                        self.magmean = lf[:,0]
                        self.luminosity_function = np.zeros((len(self.magmean), self.nbands, self.nzbins))
                    else:
                        assert((lf[:,0]==self.magmean).all())
                    
                    for j in range(self.nbands):
                        self.luminosity_function[:,j,i] = lf[:,1]

            elif len(self.fname.shape)==2:
                for i in range(self.fname.shape[0]):
                    for j in range(self.fname.shape[1]):
                        lf = np.loadtxt(self.fname[i,j])
                        if (i==0) & (j==0):
                            self.magmean = lf[:,0]
                            self.luminosity_function = np.zeros((len(self.magmean), self.nbands, self.nzbins))
                        else:
                            assert(self.magmean==lf[:,0])
                        
                        self.luminosity_function[:,j,i] = lf[:,1]

        self.y = self.luminosity_function


class AngularCorrelationFunction(Metric):

    def __init__(self, simulation, zbins=None, lumbins=None):
        Metric.__init__(self, simulation)

        if zbins is None:
            self.zbins = [0.0, 0.2]
        else:
            self.zbins = zbins
            self.zbins = np.array(self.zbins)

        self.nzbins = len(self.zbins)-1

        if lumbins is None:
            self.lumbins = np.linspace(-25, -11, 30)
        else:
            self.lumbins = lumbins

        self.mapkeys = ['luminosity', 'redshift', 'polar_ang', 'azim_ang']
        
        

