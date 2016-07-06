from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
#if __name__=='__main__':
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
import numpy as np


class Metric(object):
    """
    An abstract class which all other metric classes should descend from. All metrics
    must define the methods that are declared as abstractmethod here.
    """
    __metaclass__ = ABCMeta

    def __init__(self, ministry, catalog_type=None, tag=None, nomap=False):
        """
        Simple init method. At the very least, init methods for subclasses
        should take a ministry object as an argument.
        """
        self.ministry = ministry
        self.catalog_type = catalog_type
        self.tag = tag
        self.nomap = nomap

    @abstractmethod
    def map(self, mapunit):
        pass

    @abstractmethod
    def reduce(self):
        pass

    @abstractmethod
    def visualize(self, plotname=None, compare=False):
        pass

    @abstractmethod
    def compare(self, othermetric, plotname=None):
        pass

class GMetric(Metric):
    """
    A generic metric class that deals with measurements made over
    multiple similar quantities, i.e. mags in
    different bands or mass functions using various mass definitions
    """

    def __init__(self, ministry, zbins=None, xbins=None, catalog_type=None,
                    tag=None):
        """
        Initialize a MagnitudeMetric object. Note, all metrics should define
        an attribute called mapkeys which specifies the types of data that they
        expect.

        Arguments
        ---------
        ministry : Ministry
            The ministry object that this metric is associated with.
        zbins : array-like
            An 1-d array containing the edges of the redshift bins to
            measure the LF in.
        xbins : array-like
            A 1-d array containing the edges of the magnitude bins to
            measure the metric in.
        """
        Metric.__init__(self, ministry, catalog_type=catalog_type, tag=tag)

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

    def visualize(self, plotname=None, usecols=None, usez=None,fracdev=False,
                  ref_y=None, ref_x=[None], xlim=None, ylim=None, fylim=None,
                  f=None, ax=None, xlabel=None,ylabel=None,compare=False,
                  logx=False, **kwargs):
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
        #uses different bins, interpolate ref_y to
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
                            ax[2*r+1].append(f.add_subplot(gs[2*r+1,c],
                              sharex=ax[0][0], sharey=ax[1][0]))

            ax = np.array(ax)
            ax = np.atleast_2d(ax)

            newaxes = True
        else:
            newaxes = False

        if nzbins>1:
            for i, b in enumerate(usecols):
                for j in range(nzbins):
                    if fracdev==False:
                        print(ax.shape)
                        l1 = ax[i][j].semilogy(mxs, self.y[:,b,j],
                                          **kwargs)
                        if logx:
                            ax[i][j].set_xscale('log')
                    else:
                        l1 = ax[2*i][j].semilogy(mxs, self.y[:,b,j],
                                          **kwargs)
                        ax[2*i+1][j].plot(mxs[li:hi],
                                          (self.y[li:hi,b,j]-ref_y[:,b,j])\
                                              /ref_y[:,b,j], **kwargs)
                        if logx:
                            ax[2*i][j].set_xscale('log')
                            ax[2*i+1][j].set_xscale('log')

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
                        l1 = ax[0][i].semilogy(mxs, self.y[:,b,0],
                                            **kwargs)
                        if logx:
                            ax[0][i].set_xscale('log')

                    except Exception as e:
                        print(e)
                        l1 = ax.semilogy(mxs, self.y[:,b,0],
                                         **kwargs)
                        if logx:
                            ax.set_xscale('log')
                else:
                    print(len(mxs))
                    print(len(self.y))
                    l1 = ax[2*i][0].semilogy(mxs, self.y[:,b,0],
                                        **kwargs)
                    ax[2*i+1][0].plot(mxs[li:hi], (self.y[li:hi,b,0]-ref_y[:,b,0])\
                                      /ref_y[:,b,0], **kwargs)
                    if logx:
                        ax[2*i][0].set_xscale('log')
                        ax[2*i+1][0].set_xscale('log')

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
            sax.patch.set_alpha(0.0)
            sax.patch.set_facecolor('none')
            sax.spines['top'].set_color('none')
            sax.spines['top'].set_alpha(0.0)
            sax.spines['bottom'].set_color('none')
            sax.spines['bottom'].set_alpha(0.0)
            sax.spines['left'].set_color('none')
            sax.spines['left'].set_alpha(0.0)
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            sax.set_xlabel(r'%s' % xlabel, fontsize=18)
            sax.set_ylabel(r'%s' % ylabel, fontsize=20)
            plt.tight_layout()

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax, l1


    def compare(self, othermetrics, plotname=None, usecols=None, fracdev=True,
                    xlim=None, ylim=None, fylim=None, labels=None, **kwargs):
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

        print("labels: {0}".format(labels[0]))
        print("usecols: {0}".format(usecols[0]))

        lines = []

        for i, m in enumerate(tocompare):
            if usecols[i]!=None:
                assert(len(usecols[0])==len(usecols[i]))
            if i==0:
                if fracdev:
                    f, ax, l = m.visualize(usecols=usecols[i], fracdev=True, ref_x=ref_x,
                                        ref_y=self.y, xlim=xlim, compare=True,
                                        ylim=ylim, fylim=fylim, label=labels[i],**kwargs)
                else:
                    f, ax, l = m.visualize(usecols=usecols[i], xlim=xlim, ylim=ylim, compare=True,
                                        fracdev=False, fylim=fylim,label=labels[i],**kwargs)
            else:
                if fracdev:
                    f, ax, l = m.visualize(usecols=usecols[i], fracdev=True, ref_x=ref_x,
                                        ref_y=tocompare[0].y, compare=True,
                                        xlim=xlim, ylim=ylim, fylim=fylim,
                                        f=f, ax=ax, label=labels[i], **kwargs)
                else:
                    f, ax, l = m.visualize(usecols=usecols[i], xlim=xlim, ylim=ylim,
                                        fylim=fylim, f=f, ax=ax, fracdev=False,
                                        compare=True, label=labels[i], **kwargs)
            lines.append(l[0])

        if labels[0]!=None:
            f.legend(lines, labels)

        if plotname!=None:
            plt.savefig(plotname)

        #plt.tight_layout()

        return f, ax
