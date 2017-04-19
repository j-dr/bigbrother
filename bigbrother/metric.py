from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
from scipy.interpolate import InterpolatedUnivariateSpline
#if __name__=='__main__':
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
import numpy as np
import warnings

class Metric(object):
    """
    An abstract class which all other metric classes should descend from. All metrics
    must define the methods that are declared as abstractmethod here.
    """
    __metaclass__ = ABCMeta

    _color_list = ['k', 'b', 'r', 'm', 'g', 'c', 'y']

    def __init__(self, ministry, catalog_type=None, tag=None,
                  nomap=False, novis=False, jtype=None):
        """
        At the very least, init methods for subclasses
        should take a ministry object as an argument.
        """
        self.ministry = ministry
        self.catalog_type = catalog_type
        self.tag = tag
        self.nomap = nomap
        self.novis = novis
        self.jtype = jtype
        #njack will be determined at the time that map is called
        self.njack = None
        self.jcount = 0

    @abstractmethod
    def map(self, mapunit):
        pass

    @abstractmethod
    def reduce(self, ntasks=None):
        pass

    @abstractmethod
    def visualize(self, plotname=None, compare=False):
        pass

    @abstractmethod
    def compare(self, othermetric, plotname=None):
        pass
    
    def defaultUnits(self, key):
        if key == 'halomass':
            return 'msunh'
        if key == 'central':
            return 'unitless'

    def splitBimodal(self, x, y, largepoly=30):
        p = np.polyfit(x, y, largepoly) # polynomial coefficients for fit

        extrema = np.roots(np.polyder(p))
        extrema = extrema[np.isreal(extrema)]
        extrema = extrema[(extrema - x[1]) * (x[-2] - extrema) > 0] # exclude the endpoints due false maxima during fitting
        try:
            root_vals = [sum([p[::-1][i]*(root**i) for i in range(len(p))]) for root in extrema]
            peaks = extrema[np.argpartition(root_vals, -2)][-2:] # find two peaks of bimodal distribution

            mid, = np.where((x - peaks[0])* (peaks[1] - x) > 0)
             # want data points between the peaks
        except:
            warnings.warn("Peak finding failed!")
            return None

        try:
            p_mid = np.polyfit(x[mid], y[mid], 2) # fit middle section to a parabola
            midpoint = np.roots(np.polyder(p_mid))[0]
        except:
            warnings.warn("Polynomial fit between peaks of distribution poorly conditioned. Falling back on using the minimum! May result in inaccurate split determination.")
            if len(mid) == 0:
                return None

            midx = np.argmin(y[mid])
            midpoint = x[mid][midx]

        return midpoint

    def splitPercentile(self, x, p):

        pi = p*len(x)
        xidx = x.argsort()
        
        return x[xidx[pi]]

    def selectCMASS(self, mags):
        cpar  = (0.7 * (mags[:,0] - mags[:,1])
                 + 1.2 * (mags[:,1] - mags[:,2] - 0.18))
        cperp = ((mags[:,1] - mags[:,2])
                 - (mags[:,1] - mags[:,1]) / 4  - 0.18)
        dperp = ((mags[:,1] - mags[:,2])
                 - (mags[:,0] - mags[:,1]) / 8)

        cidx = ((dperp > 0.55) & (mags[:,2] < (19.86 + 1.6 * (dperp-0.8)))
                 & (17.5 < mags[:,2]) & (mags[:,2]<19.9))

        return cidx
        

    def jackknife(self, arg, reduce_jk=True):

        jdata = np.zeros(arg.shape)

        for i in range(self.njacktot):
            #generalized so can be used if only one region
            if self.njacktot==1:
                idx = [0]
            else:
                idx = [j for j in range(self.njacktot) if i!=j]

            #jackknife indices should always be first
            jl = len(arg.shape)
            jidx = [slice(0,arg.shape[j]) if j!=0 else idx for j in range(jl)]
            jdidx = [slice(0,arg.shape[j]) if j!=0 else i for j in range(jl)]
            jdata[jdidx] = np.sum(arg[jidx], axis=0)

        if reduce_jk:
            jest = np.sum(jdata, axis=0) / self.njacktot
            jvar = np.sum((jdata - jest)**2, axis=0) * (self.njacktot - 1) / self.njacktot

            return jdata, jest, jvar

        else:
            return jdata

    def setNJack(self):
        if self.jtype is None:
            self.njack = 1
            self.njacktot = 1
        else:
            self.njack = self.ministry.njack
            self.njacktot = self.ministry.njacktot


class GMetric(Metric):
    """
    A generic metric class that deals with measurements made over
    multiple similar quantities, i.e. mags in
    different bands or mass functions using various mass definitions
    """

    def __init__(self, ministry, zbins=None, xbins=None, catalog_type=None,
                    tag=None, **kwargs):
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
        Metric.__init__(self, ministry, catalog_type=catalog_type, tag=tag,
                        **kwargs)

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
    def reduce(self, rank=None, comm=None):
        pass

    def visualize(self, plotname=None, usecols=None,
                    usez=None,fracdev=False, sharex=True,
                    sharey=True,
                    ref_y=None, ref_x=[None], ref_ye=None,
                    xlim=None, ylim=None,
                    fylim=None, f=None, ax=None, label=None,
                    xlabel=None, ylabel=None,compare=False,
                    logx=False, logy=True, rusecols=None,
                    rusez=None, **kwargs):
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
        print('xlabel: {0}'.format(xlabel))
        print('ylabel: {0}'.format(ylabel))

        if usecols is None:
            usecols = range(self.nbands)

        if (rusecols is None) and (ref_y is not None):
            rusecols = range(ref_y.shape[1])

        if usez is None:
            usez = range(self.nzbins)

        if (rusez is None) and (ref_y is not None):
            rusez = range(ref_y.shape[2])

        nzbins = len(usez)

        l1 = None

        #format x-values
        if hasattr(self, 'xmean'):
            
            if len(self.xmean.shape)==1:
                mxs = np.tile(self.xmean, [self.nzbins, self.nbands, 1]).T
            elif len(self.xmean.shape)==2:
                mxs = np.tile(self.xmean.reshape(self.xmean.shape[1],1,self.xmean.shape[0]),
                                [1, self.nbands, 1]).T
            else:
                mxs = self.xmean
        else:
            if len(self.xbins.shape)==1:
                xbins = np.tile(self.xbins, [self.nzbins, self.nbands, 1]).T
            elif len(self.xbins.shape)==2:
                xbins = np.tile(self.xbins.reshape(self.xbins.shape[1],1,self.xbins.shape[0]),
                                [1, self.nbands, 1]).T
            else:
                xbins = self.xbins

            mxs = ( xbins[1:,:,:] + xbins[:-1,:,:] ) / 2

        if fracdev:

            if len(ref_x.shape)==1:
                ref_x = np.tile(ref_x, [self.nzbins, self.nbands, 1]).T
            elif len(ref_x.shape)==2:
                ref_x = np.tile(ref_x.reshape(ref_x.shape[1],1,ref_x.shape[0]),
                                [1, self.nbands, 1]).T



        #If want to plot fractional deviations, and ref_y
        #uses different bins, interpolate ref_y to
        #magniutdes given at mxs. Don't extrapolate!

        if fracdev:
            lidx = np.zeros((len(usecols), len(usez)), dtype=np.int)
            hidx = np.zeros((len(usecols), len(usez)), dtype=np.int)

            rxs = ref_x.shape
            xs  = mxs.shape
            rls = ref_y.shape
            iref_y = np.zeros(rxs)
            if ref_ye is not None:
                iref_ye = np.zeros(rxs)

            for i, c in enumerate(usecols):
                for j in range(len(usez)):
                    xi  = mxs[:,usecols[i],usez[j]]
                    rxi = ref_x[:,rusecols[i],rusez[j]]

                    if fracdev & ((rxs[0]!=xs[0]) | ((rxi[0]!=xi[0])  | (rxi[-1]!=xi[-1]))):
                        lidx[i,j] = xi.searchsorted(rxi[0])
                        hidx[i,j] = xi.searchsorted(rxi[-1])
                        nanidx = np.isnan(ref_y[:,rusecols[i],rusez[j]]) | np.isnan(ref_x[:,rusecols[i],rusez[j]])
                        sply = InterpolatedUnivariateSpline(ref_x[~nanidx,rusecols[i],rusez[j]], ref_y[~nanidx,rusecols[i],rusez[j]])
                        iref_y[lidx[i,j]:hidx[i,j],rusecols[i],rusez[j]] = sply(mxs[lidx[i,j]:hidx[i,j],c,usez[j]])

                        if ref_ye is not None:
                            nanidx = np.isnan(ref_ye[:,rusecols[i],rusez[j]]) | np.isnan(ref_x[:,rusecols[i],rusez[j]])
                            splye = InterpolatedUnivariateSpline(ref_x[~nanidx,rusecols[i],rusez[j]], ref_ye[~nanidx,rusecols[i],rusez[j]])
                            iref_ye[lidx[i,j]:hidx[i,j],rusecols[i],rusez[j]] = splye(mxs[lidx[i,j]:hidx[i,j],c,usez[j]])
                    else:
                        lidx[i,j] = 0
                        hidx[i,j] = len(mxs)
                        iref_y[lidx[i,j]:hidx[i,j],rusecols[i],rusez[j]] = ref_y[:,rusecols[i],rusez[j]]
                        if ref_ye is not None:
                            iref_ye[lidx[i,j]:hidx[i,j],rusecols[i],rusez[j]] = ref_ye[:,rusecols[i],rusez[j]]

            ref_y = iref_y
            if ref_ye is not None:
                ref_ye = iref_ye

        #if no figure provided, set up figure and axes
        if f is None:
            if fracdev==False:
                f, ax = plt.subplots(len(usecols), nzbins,
                                     sharex=True, sharey=True, figsize=(8,8))
                ax = np.array(ax).reshape((len(usecols), nzbins))
            #if want fractional deviations, need to make twice as
            #many rows of axes. Every other row contains fractional
            #deviations from the row above it.
            else:
                assert(ref_y is not None)
                gs = gridspec.GridSpec(len(usecols)*2, nzbins)
                f = plt.figure()
                ax = np.zeros((len(usecols)*2, nzbins), dtype='O')
                for r in range(len(usecols)):
                    for c in range(nzbins):
                        if (r==0) & (c==0):
                            ax[2*r][c] = f.add_subplot(gs[2*r,c])
                            ax[2*r+1][c] = f.add_subplot(gs[2*r+1,c], sharex=ax[0][0])
                        else:
                            if sharex & sharey:
                                ax[2*r][c] = f.add_subplot(gs[2*r,c], sharex=ax[0][0], sharey=ax[0][0])
                            elif sharex:
                                ax[2*r][c] = f.add_subplot(gs[2*r,c], sharex=ax[0][0])
                            elif sharey:
                                ax[2*r][c] = f.add_subplot(gs[2*r,c], sharey=ax[0][0])
                            else:
                                ax[2*r][c]= f.add_subplot(gs[2*r,c])

                            ax[2*r+1][c] = f.add_subplot(gs[2*r+1,c],
                              sharex=ax[0][0], sharey=ax[1][0])

            newaxes = True
        else:
            newaxes = False

        if nzbins>1:
            for i, b in enumerate(usecols):
                for j in range(nzbins):
                    if fracdev==False:
                        if (self.y[:,b,usez[j]]==0).all() | (np.isnan(self.y[:,b,usez[j]]).all()): continue
                        l1 = ax[i][j].plot(mxs[:,b,usez[j]], self.y[:,b,usez[j]], **kwargs)
                        if self.ye is not None:
                            ax[i][j].fill_between(mxs[:,b,usez[j]], self.y[:,b,usez[j]]-self.ye[:,b,usez[j]],
                              self.y[:,b,usez[j]]+self.ye[:,b,usez[j]],
                              alpha=0.5, **kwargs)
                        if logx:
                            ax[i][j].set_xscale('log')
                        if logy:
                            ax[i][j].set_yscale('log')
                    else:
                        li = lidx[i,j]
                        hi = hidx[i,j]

                        rb = rusecols[i]
                        #calculate error on fractional
                        #difference
                        if (ref_ye is not None) & (self.ye is not None):
                            vye = self.ye[li:hi,b,usez[j]]**2
                            vrye = ref_ye[li:hi,rb,rusez[j]]**2
                            fye = (self.y[li:hi,b,usez[j]] - ref_y[li:hi,rb,rusez[j]]) / ref_y[li:hi,rb,rusez[j]]
                            dye = fye * np.sqrt( (vye + vrye) / (self.y[li:hi,b,usez[j]] - ref_y[li:hi,rb,rusez[j]]) ** 2 + ref_ye[li:hi,rb,rusez[j]] ** 2 / ref_y[li:hi,rb,rusez[j]]**2 )
                        else:
                            fye = (self.y[li:hi,b,usez[j]] - ref_y[li:hi,rb,rusez[j]]) / ref_y[li:hi,rb,rusez[j]]
                            dye = None

                        if (self.y[:,b,usez[j]]==0).all() | (np.isnan(self.y[:,b,usez[j]]).all()): continue
                        l1 = ax[2*i][j].plot(mxs[:,b,usez[j]], self.y[:,b,usez[j]], **kwargs)
                        ax[2*i+1][j].plot(mxs[li:hi,b,usez[j]], fye, **kwargs)
                        if self.ye is not None:
                            ax[2*i][j].fill_between(mxs[:,b,usez[j]], self.y[:,b,usez[j]]-self.ye[:,b,usez[j]],
                              self.y[:,b,usez[j]]+self.ye[:,b,usez[j]],
                              alpha=0.5, **kwargs)
                        if dye is not None:
                            ax[2*i+1][j].fill_between(mxs[li:hi,b,usez[j]], fye-dye,
                              fye+dye,
                              alpha=0.5, **kwargs)

                        if logx:
                            ax[2*i][j].set_xscale('log')

                        if logy:
                            ax[2*i][j].set_yscale('log')

                        if (i==0) & (j==0):
                            if xlim is not None:
                                ax[0][0].set_xlim(xlim)
                            if ylim is not None:
                                ax[0][0].set_ylim(ylim)
                            if fylim is not None:
                                ax[1][0].set_ylim(fylim)

        else:
            for i, b in enumerate(usecols):
                if fracdev==False:
                    if (self.y[:,b,0]==0).all() | (np.isnan(self.y[:,b,0]).all()): continue
                    l1 = ax[i][0].plot(mxs[:,b,0], self.y[:,b,0], **kwargs)
                    if self.ye is not None:
                        ax[i][0].fill_between(mxs[:,b,0], self.y[:,b,0] - self.ye[:,b,0],
                                                self.y[:,b,0] + self.ye[:,b,0],
                                                alpha=0.5, **kwargs)
                                            
                    if logx:
                        ax[i][0].set_xscale('log')
                    if logy:
                        ax[i][0].set_yscale('log')

                else:
                    li = lidx[i,0]
                    hi = hidx[i,0]

                    rb = rusecols[i]
                    #calculate error on fractional
                    #difference
                    if (ref_ye is not None) & (self.ye is not None):
                        vye = self.ye[li:hi,b,0]**2
                        vrye = ref_ye[li:hi,rb,0]**2
                        fye = (self.y[li:hi,b,0] - ref_y[li:hi,rb,0]) / ref_y[li:hi,rb,0]
                        dye = fye * np.sqrt( (vye + vrye) / (self.y[li:hi,b,0] - ref_y[li:hi,rb,0]) ** 2 + ref_ye[li:hi,rb,0] ** 2 / ref_y[li:hi,rb,0]**2 )
                    else:
                        fye = (self.y[li:hi,b,0] - ref_y[li:hi,rb,0]) / ref_y[li:hi,rb,0]
                        dye = None

                    if (self.y[:,b,0]==0).all() | (np.isnan(self.y[:,b,0]).all()): continue
                    l1 = ax[2*i][0].plot(mxs[:,b,0], self.y[:,b,0], **kwargs)
                    ax[2*i+1][0].plot(mxs[li:hi,b,0], fye, **kwargs)

                    if self.ye is not None:
                        ax[2*i][0].fill_between(mxs[:,b,0], self.y[:,b,0]-self.ye[:,b,0],
                          self.y[:,b,0]+self.ye[:,b,0],
                          alpha=0.5, **kwargs)
                    if dye is not None:
                        ax[2*i+1][0].fill_between(mxs[li:hi,b,0], fye-dye,
                          fye+dye,
                          alpha=0.5, **kwargs)

                    if logx:
                        ax[2*i][0].set_xscale('log')
                    if logy:
                        ax[2*i][0].set_yscale('log')

                    if (i==0):
                        if xlim is not None:
                            ax[0][0].set_xlim(xlim)
                        if ylim is not None:
                            ax[0][0].set_ylim(ylim)
                        if fylim is not None:
                            ax[1][0].set_ylim(fylim)

        #if we just created the axes, add labels
        if newaxes:
            sax = f.add_subplot(111)
            plt.setp(sax.get_xticklines(), visible=False)
            plt.setp(sax.get_yticklines(), visible=False)
            plt.setp(sax.get_xticklabels(), visible=False)
            plt.setp(sax.get_yticklabels(), visible=False)
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
            sax.set_xlabel(r'%s' % xlabel, labelpad=40)
            sax.set_ylabel(r'%s' % ylabel, labelpad=40)
            #plt.tight_layout()

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax, l1


    def compare(self, othermetrics, plotname=None, usecols=None, usez=None,
                fracdev=True, xlim=None, ylim=None, fylim=None, labels=None,
                **kwargs):
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

        if usecols is not None:
            if not hasattr(usecols[0], '__iter__'):
                usecols = [usecols]*len(tocompare)
            else:
                assert(len(usecols)==len(tocompare))
        else:
            usecols = [None]*len(tocompare)

        if usez is not None:
            if not hasattr(usez[0], '__iter__'):
                usez = [usez]*len(tocompare)
            else:
                assert(len(usez)==len(tocompare))
        else:
            usez = [None]*len(tocompare)

        if fracdev:
            if hasattr(self, 'xmean'):
                ref_x = self.xmean
            else:
                ref_x = (self.xbins[:-1,...] + self.xbins[1:,...]) / 2
            
            if len(ref_x.shape)==1:
                ref_x = np.tile(ref_x, [self.nzbins, self.nbands, 1]).T
            elif len(ref_x.shape)==2:
                ref_x = np.tile(ref_x.reshape(ref_x.shape[1],1,ref_x.shape[0]),
                                [1, self.nbands, 1]).T

        if labels is None:
            labels = [None]*len(tocompare)

        lines = []

        for i, m in enumerate(tocompare):
            if usecols[i] is not None:
                assert(len(usecols[0])==len(usecols[i]))
            if i==0:
                if fracdev:
                    f, ax, l = m.visualize(usecols=usecols[i], fracdev=True, 
                                             ref_x=ref_x, rusecols=usecols[0],
                                             ref_y=self.y, xlim=xlim, compare=True,
                                             ylim=ylim, fylim=fylim, label=labels[i],
                                             usez=usez[i],color=Metric._color_list[i],
                                             rusez=usez[0],**kwargs)
                else:
                    f, ax, l = m.visualize(usecols=usecols[i], xlim=xlim, ylim=ylim, compare=True,
                                             fracdev=False, fylim=fylim,label=labels[i],usez=usez[i],
                                             color=Metric._color_list[i], **kwargs)
            else:
                if fracdev:
                    f, ax, l = m.visualize(usecols=usecols[i], fracdev=True, ref_x=ref_x,
                                             rusecols=usecols[0], ref_y=tocompare[0].y,
                                             ref_ye=tocompare[0].ye, compare=True, xlim=xlim,
                                             ylim=ylim, fylim=fylim, f=f, ax=ax, label=labels[i],
                                             usez=usez[i], rusez=usez[0], color=Metric._color_list[i], 
                                             **kwargs)
                else:
                    f, ax, l = m.visualize(usecols=usecols[i], xlim=xlim, ylim=ylim,
                                             fylim=fylim, f=f, ax=ax, fracdev=False,
                                             compare=True, label=labels[i], usez=usez[i],
                                             color=Metric._color_list[i], **kwargs)
            lines.append(l[0])

        if labels[0] is not None:
            f.legend(lines, labels, 'best')

        if plotname is not None:
            plt.savefig(plotname)

        #plt.tight_layout()

        return f, ax


def jackknifeMap(func):
    def wrapper(self, mapunit):
        if self.njack is None:
            self.setNJack()

        func(self, mapunit)

        if self.jtype is not None:
            self.jcount += 1

    return wrapper
