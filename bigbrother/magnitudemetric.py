from __future__ import print_function, division
from .metric import Metric, GMetric
#if __name__=='__main__':
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
import numpy as np


class MagnitudeMetric(GMetric):
    """
    Restruct GMetric to magnitudes
    """
    
    def __init__(self, ministry, zbins=None, magbins=None, 
                 catalog_type=None):
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

        if magbins is None:
            magbins = np.linspace(-25,-15, 40)

        self.magbins = magbins
        self.aschema = 'galaxyonly'
        self.unitmap = {'appmag':'mag', 'luminosity':'mag'}

        GMetric.__init__(self, ministry, zbins=zbins, xbins=magbins,
                         catalog_type=catalog_type)


class LuminosityFunction(MagnitudeMetric):
    """
    A generic luminosity function class. More specific types of luminosity
    functions inherit this class.
    """

    def __init__(self, ministry, central_only=False, zbins=None, magbins=None,
                 catalog_type=['galaxycatalog']):

        """
        Initialize a LuminosityFunction object. Note, all metrics should define
        an attribute called mapkeys which specifies the types of data that they
        expect.

        Arguments
        ---------
        ministry : Ministry
            The ministry object that this metric is associated with.
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

        MagnitudeMetric.__init__(self, ministry, zbins=zbins, magbins=magbins,
                                 catalog_type=catalog_type)

        self.central_only = central_only
        if central_only:
            self.mapkeys = ['luminosity', 'redshift', 'central']
        else:
            self.mapkeys = ['luminosity', 'redshift']

        self.aschema = 'galaxyonly'
        
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
        area = self.ministry.galaxycatalog.getArea()
        self.luminosity_function = self.lumcounts

        for i in range(self.nzbins):
            vol = self.ministry.calculate_volume(area, self.zbins[i], self.zbins[i+1])
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

    def visualize(self, plotname=None, usecols=None, usez=None,fracdev=False, 
                  ref_y=None, ref_x=[None], xlim=None, ylim=None, fylim=None, 
                  f=None, ax=None, xlabel=None,ylabel=None,compare=False,**kwargs):

        if xlabel is None:
            xlabel = "Mag"
        if ylabel is None:
            ylabel = r'$\phi \, [Mpc^{-3}\, h^{3}]$'

        MagnitudeMetric.visualize(self, plotname=plotname, usecols=usecols, usez=usez,
                                  fracdev=fracdev, ref_y=ref_y, ref_x=ref_x, xlim=xlim,
                                  ylim=ylim, fylim=fylim, f=f, ax=ax, xlabel=xlabel,
                                  ylabel=ylabel, compare=compare, **kwargs)

class MagCounts(MagnitudeMetric):
    """
    Galaxy counts per magnitude.
    """

    def __init__(self, ministry, zbins=[0.0, 0.2],  magbins=None, 
                 catalog_type=['galaxycatalog']):

        if magbins is None:
            magbins = np.linspace(10, 30, 60)

        MagnitudeMetric.__init__(self,ministry, zbins=zbins, magbins=magbins,
                                 catalog_type=catalog_type)
        
        if zbins is not None:
            self.mapkeys = ['appmag', 'redshift']
        else:
            self.mapkeys = ['appmag']

        self.aschema = 'galaxyonly'

    def map(self, mapunit):
        self.nbands = mapunit['appmag'].shape[1]

        if not hasattr(self, 'magcounts'):
            self.magcounts = np.zeros((len(self.magbins)-1, 
                                       self.nbands, self.nzbins))
            
        if self.zbins is not None:
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
        area = self.ministry.galaxycatalog.getArea()
        self.magcounts = self.magcounts/area
        self.y = self.magcounts

    def visualize(self, plotname=None, usecols=None, usez=None,fracdev=False, 
                  ref_y=None, ref_x=[None], xlim=None, ylim=None, fylim=None, 
                  f=None, ax=None, xlabel=None,ylabel=None,compare=False,**kwargs):

        if xlabel is None:
            xlabel = "Mag"
        if ylabel is None:
            ylabel = r'$n \, [mag^{-1}\, deg^{-2}]$'

        MagnitudeMetric.visualize(self, plotname=plotname, usecols=usecols, usez=usez,
                                  fracdev=fracdev, ref_y=ref_y, ref_x=ref_x, xlim=xlim,
                                  ylim=ylim, fylim=fylim, f=f, ax=ax, xlabel=xlabel,
                                  ylabel=ylabel, compare=compare, **kwargs)


        
class LcenMass(Metric):
    """
    Central galaxy luminosity - halo virial mass relation.
    """
    def __init__(self, ministry, zbins=None, massbins=None, 
                 catalog_type=['galaxycatalog']):
        Metric.__init__(self, ministry, catalog_type=catalog_type)

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

        self.mapkeys = ['luminosity', 'redshift', 'central', 'halomass']
        self.aschema = 'galaxyonly'
        self.unitmap = {'luminosity':'mag', 'halomass':'msunh'}

        
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
            mb = np.digitize(mu['halomass'][zlidx:zhidx], bins=self.massbins)

            for j in range(len(self.massbins)-1):
                blum = mu['luminosity'][zlidx:zhidx,:][mb==j]
                self.bincount[j,:,i] += len(blum)
                self.totlum[j,:,i] += np.sum(blum, axis=0)


    def reduce(self):

        self.lcen_mass = self.totlum/self.bincount


    def visualize(self, compare=False, plotname=None, f=None, ax=None, 
                  usebands=None, **kwargs):

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

        if newaxes:
            sax = f.add_subplot(111)
            sax.patch.set_alpha(0.0)
            sax.patch.set_facecolor('none')
            sax.spines['top'].set_color('none')
            sax.spines['bottom'].set_color('none')
            sax.spines['left'].set_color('none')
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            sax.set_xlabel(r'$M_{halo}\, [M_{sun} h^{-1}]$')
            sax.set_ylabel(r'$L_{cen}\, [mag]$')

        if self.nzbins>1:
            for i, b in enumerate(usebands):
                for j in range(self.nzbins):
                    ax[i][j].semilogx(mmass, self.lcen_mass[:,b,j], 
                                      **kwargs)
        else:
            for i, b in enumerate(usebands):
                for j in range(self.nzbins):
                    ax[i].semilogx(mmass, self.lcen_mass[:,b,j], 
                                   **kwargs)

        plt.tight_layout()

        if (plotname is not None) and (not compare):
            plt.savefig(plotname)

        return f, ax


    def compare(self, othermetrics, plotname=None, usebands=None, **kwargs):
        tocompare = [self]
        tocompare.extend(othermetrics)

        if usebands is not None:
            if not hasattr(usebands[0], '__iter__'):
                usebands = [usebands]*len(tocompare)
            else:
                assert(len(usebands)==len(tocompare))
        else:
            usebands = [None]*len(tocompare)
        
        for i, m in enumerate(tocompare):
            if usebands[i] is not None:
                assert(len(usebands[0])==len(usebands[i]))
            if i==0:
                f, ax = m.visualize(usebands=usebands[i], compare=True,
                                    **kwargs)
            else:
                f, ax = m.visualize(usebands=usebands[i], compare=True,
                                    f=f, ax=ax, **kwargs)

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax


class ColorColor(Metric):
    """
    Color-color diagram.
    """
    def __init__(self, ministry, zbins=[0.0, 0.2], cbins=None, 
                 catalog_type=['galaxycatalog'], usebands=None,
                 amagcut=-19.0):
        Metric.__init__(self, ministry, catalog_type=catalog_type)
        
        self.zbins = zbins
        if zbins is None:
            self.nzbins = 1
        else:
            self.nzbins = len(zbins)-1
            self.zbins = np.array(self.zbins)

        if cbins is None:
            self.cbins = np.linspace(-1, 2, 30)
        else:
            self.cbins = cbins

        if zbins is not None:
            self.mapkeys = ['luminosity', 'redshift']
        else:
            self.mapkeys = ['luminosity']

        self.amagcut = amagcut
        self.usebands = usebands
        self.aschema = 'galaxyonly'
        self.unitmap = {'luminosity':'mag'}

    def map(self, mapunit):

        if self.usebands is None:
            self.nbands = mapunit['luminosity'].shape[1]
            self.usebands = range(self.nbands)
        else:
            self.nbands = len(self.usebands)

        self.nclr = self.nbands-1

        clr = np.zeros((len(mapunit['luminosity']),self.nbands-1))
        for i, b in enumerate(self.usebands[:-1]):
            clr[:,i] = mapunit['luminosity'][:,self.usebands[i]] - mapunit['luminosity'][:,self.usebands[i+1]]

        if not hasattr(self, 'cc'):
            self.cc = np.zeros((len(self.cbins)-1, len(self.cbins)-1, 
                                self.nbands-2, self.nzbins))
            
        if self.zbins is not None:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

                if self.amagcut!=None:
                    for e, j in enumerate(self.usebands):
                        if e==0:
                            lidx = mapunit['luminosity'][zlidx:zhidx,j]<self.amagcut
                        else:
                            lix = mapunit['luminosity'][zlidx:zhidx,j]<self.amagcut
                            lidx = lidx & lix

                for j in range(self.nclr-1):
                    c, e0, e1 = np.histogram2d(clr[zlidx:zhidx,j+1][lidx],
                                               clr[zlidx:zhidx,j][lidx],
                                               bins=self.cbins)
                    self.cc[:,:,j,i] += c
        else:
            for i in range(self.nclr-1):
                c, e0, e1 = np.histogram2d(clr[:,i], 
                                           clr[:,i+1], 
                                           bins=self.cbins)
                self.cc[:,:,i,0] += c

    
    def reduce(self):
        area = self.ministry.galaxycatalog.getArea()
        self.cc = self.cc/area

    def visualize(self, compare=False, plotname=None, f=None, ax=None,
                  usecolors=None, **kwargs):

        if hasattr(self, 'magmean'):
            mclr = self.mclr
        else:
            mclr = np.array([(self.cbins[i]+self.cbins[i+1])/2 
                              for i in range(len(self.cbins)-1)])

        if usecolors is None:
            usecolors = range(self.cc.shape[2])

        if f is None:
            f, ax = plt.subplots(self.nzbins, len(usecolors),
                                 sharex=True, sharey=True, figsize=(8,8))
            ax = np.atleast_2d(ax).T
            newaxes = True
        else:
            newaxes = False

        X, Y = np.meshgrid(mclr, mclr)

        for i in usecolors:
            for j in range(self.nzbins):
                ax[j][i].contour(X, Y, self.cc[:,:,i,j].T, 30,
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
            sax.set_xlabel(r'$Color$')
            sax.set_ylabel(r'$Color$')

        plt.tight_layout()

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)


        return f, ax

    def compare(self, othermetric, plotname=None, usecolors=None, **kwargs):
        if usecolors is not None:
            assert(len(usecolors[0])==len(usecolors[1]))
            f, ax = self.visualize(usecolors=usecolors[0], compare=True,
                                   **kwargs)
            f, ax = othermetric.visualize(usecolors=usecolors[1], compare=True,
                                          f=f, ax=ax, **kwargs)
        else:
            f, ax = self.visualize(usecolors=usecolors, compare=True, **kwargs)
            f, ax = othermetric[0].visualize(usecolors=usecolors, compare=True,
                                          f=f, ax=ax, **kwargs)

        if plotname is not None:
            plt.savefig(plotname)



class ColorMagnitude(Metric):
    """
    Color-magnitude diagram.
    """
    def __init__(self, ministry, zbins=[0.0, 0.2], magbins=None, 
                 cbins=None, central_only=False, logscale=False,
                 catalog_type=['galaxycatalog'], usebands=None):

        Metric.__init__(self, ministry, catalog_type=catalog_type)
        
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

        self.usebands = usebands
        if self.usebands is not None:
            self.nbands = len(self.usebands)

        self.central_only = central_only
        self.logscale = logscale

        if central_only & (zbins is not None):
            self.mapkeys = ['luminosity', 'redshift', 'central']
        elif (zbins is not None):
            self.mapkeys = ['luminosity', 'redshift']
        else:
            self.mapkeys = ['luminosity']

        self.aschema = 'galaxyonly'
        self.unitmap = {'luminosity':'mag'}

    def map(self, mapunit):
        
        if self.usebands is None:
            self.nbands = mapunit['luminosity'].shape[1]

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

        if self.zbins is not None:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mu['redshift'].searchsorted(self.zbins[i])
                zhidx = mu['redshift'].searchsorted(self.zbins[i+1])
                for j in range(self.nbands):
                    for k in range(self.nbands):
                        if k<=j: continue
                        ind = k*(k-1)/2+j-1
                        c, e0, e1 = np.histogram2d(mu['luminosity'][zlidx:zhidx,j], 
                                                   mu['luminosity'][zlidx:zhidx,j] -
                                                   mu['luminosity'][zlidx:zhidx,k], 
                                                   bins=[self.magbins,self.cbins])
                        self.cc[:,:,ind,i] += c
        else:
            for j in range(self.nbands):
                for k in range(self.nbands):
                    if k<=j: continue
                    ind = int(k*(k-1)/2+j-1)
                    c, e0, e1 = np.histogram2d(mu['luminosity'][:,j], 
                                               mu['luminosity'][:,j] -
                                               mu['luminosity'][:,k], 
                                               bins=[self.magbins,self.cbins])
                    self.cc[:,:,ind,0] += c


    
    def reduce(self):
        area = self.ministry.galaxycatalog.getArea()
        self.cc = self.cc/area

    def visualize(self, plotname=None, f=None, ax=None, usecolors=None, 
                  compare=False, **kwargs):

        x = (self.magbins[:-1]+self.magbins[1:])/2
        y = (self.cbins[:-1]+self.cbins[1:])/2
        X, Y = np.meshgrid(x, y)

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
            ax = np.atleast_2d(ax).T

            newaxes = True
        else:
            newaxes = False
            
        for i, c in enumerate(usecolors):
            for j in range(self.nzbins):

                ax[i][j].contour(X, Y, cc[:,:,c,j].T,30,
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
            sax.set_xlabel(r'$Color$')
            sax.set_ylabel(r'$Mag$')

        plt.tight_layout()

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax

    def compare(self, othermetrics, plotname=None, usecolors=None, **kwargs):
        tocompare = [self]
        tocompare.extend(othermetrics)

        if usecolors is not None:
            if not hasattr(usecolors[0], '__iter__'):
                usecolors = [usecolors]*len(tocompare)
            else:
                assert(len(usecolors)==len(tocompare))
        else:
            usecolors = [None]*len(tocompare)
        
        for i, m in enumerate(tocompare):
            if usecolors[i] is not None:
                assert(len(usecolors[0])==len(usecolors[i]))
            if i==0:
                f, ax = m.visualize(usecolors=usecolors[i], compare=True,
                                    **kwargs)
            else:
                f, ax = m.visualize(usecolors=usecolors[i], compare=True,
                                    f=f, ax=ax, **kwargs)

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax

class FQuenched(Metric):

    def __init__(self, ministry, zbins=[0.0, 0.2], m=0.0,b=1.2,catalog_type=['galaxycatalog']):
        Metric.__init__(self, ministry, catalog_type=catalog_type)
        self.zbins = zbins

        if zbins is None:
            self.nzbins = 1
        else:
            self.nzbins = len(zbins)-1
            self.zbins = np.array(self.zbins)

        self.m = m
        self.b = b

        self.mapkeys = ['luminosity', 'redshift']
        self.unitmap = {'luminosity':'mag'}
        self.aschema = 'galaxyonly'

    def map(self, mapunit):

        if not hasattr(self, 'qscounts'):
            self.qscounts = np.zeros(self.nzbins)
            self.tcounts = np.zeros(self.nzbins)

        if self.zbins is not None:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

                qidx, = np.where((mapunit['luminosity'][zlidx:zhidx,0] 
                                 - mapunit['luminosity'][zlidx:zhidx,1])
                                > (self.m * mapunit['luminosity'][zlidx:zhidx,0] 
                                   + self.b))

                self.qscounts[i] = len(qidx)
                self.tcounts[i] = zhidx-zlidx

        else:
            qidx = np.where((mapunit['luminosity'][:,0] 
                             - mapunit['luminosity'][:,1])
                            > (self.m * mapunit['luminosity'][:,0] 
                               + self.b))

            self.qscounts[0] = len(qidx)
            self.tcounts[0] = len(mapunit['luminosity'])
        
                
    def reduce(self):
        self.fquenched = self.qscounts/self.tcounts

    def visualize(self, f=None, ax=None, compare=False, plotname=None,
                  **kwargs):
        
        if f is None:
            f, ax = plt.subplots(1, figsize=(8,8))
            newaxes = True
        else:
            newaxes = False

        zm = (self.zbins[:-1] + self.zbins[1:])/2
            
        ax.plot(zm, self.fquenched)

        if newaxes:
            sax = f.add_subplot(111)
            sax.patch.set_alpha(0.0)
            sax.patch.set_facecolor('none')
            sax.spines['top'].set_color('none')
            sax.spines['bottom'].set_color('none')
            sax.spines['left'].set_color('none')
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            sax.set_xlabel(r'$z$')
            sax.set_ylabel(r'$f_{red}$')

        plt.tight_layout()

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax

    def compare(self, othermetrics, plotname=None, **kwargs):
        tocompare = [self]
        tocompare.extend(othermetrics)

        for i, m in enumerate(tocompare):
            if i==0:
                f, ax = m.visualize(compare=True, **kwargs)
            else:
                f, ax = m.visualize(f=f, ax=ax, compare=True, **kwargs)

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax

class FRed(Metric):

    def __init__(self, ministry, zbins=[0.0, 0.2], catalog_type=['galaxycatalog'], zeroind=True):
        Metric.__init__(self, ministry, catalog_type=catalog_type)
        self.zbins = zbins

        if zbins is None:
            self.nzbins = 1
        else:
            self.nzbins = len(zbins)-1
            self.zbins = np.array(self.zbins)

        self.mapkeys = ['ctcatid', 'redshift']
        self.unitmap = {}
        self.aschema = 'galaxyonly'
        self.zeroind = zeroind

        self.ctcat = np.genfromtxt('/nfs/slac/g/ki/ki23/des/jderose/l-addgals/training/cooper/dr6_cooper_id_with_red.dat')

    def map(self, mapunit):

        if not hasattr(self, 'nred'):
            self.qscounts = np.zeros(self.nzbins)
            self.tcounts = np.zeros(self.nzbins)

        if self.zbins is not None:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])
                
                if self.zeroind:
                    qidx, = np.where(self.ctcat[mapunit['ctcatid'][zlidx:zhidx]-1,3]==1)
                else:
                    qidx, = np.where(self.ctcat[mapunit['ctcatid'][zlidx:zhidx],3]==1)

                self.qscounts[i] = len(qidx)
                self.tcounts[i] = zhidx-zlidx

        else:
            if self.zeroind:
                qidx, = np.where(self.ctcat[mapunit['ctcatid']-1,3]==1)
            else:
                qidx, = np.where(self.ctcat[mapunit['ctcatid'],3]==1)

            self.qscounts[0] = len(qidx)
            self.tcounts[0] = len(mapunit['ctcatid'])
        
                
    def reduce(self):
        self.fquenched = self.qscounts/self.tcounts

    def visualize(self, plotname=None, f=None, ax=None, compare=False, **kwargs):
        
        if f is None:
            f, ax = plt.subplots(1, figsize=(8,8))
            newaxes = True
        else:
            newaxes = False

        zm = (self.zbins[:-1] + self.zbins[1:])/2
            
        ax.plot(zm, self.fquenched)

        if newaxes:
            sax = f.add_subplot(111)
            sax.patch.set_alpha(0.0)
            sax.patch.set_facecolor('none')
            sax.spines['top'].set_color('none')
            sax.spines['bottom'].set_color('none')
            sax.spines['left'].set_color('none')
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            sax.set_xlabel(r'$M_{halo}\, [M_{sun} h^{-1}]$')
            sax.set_ylabel(r'$L_{cen}\, [mag]$')

        plt.tight_layout()

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax

    def compare(self, othermetrics, plotname=None, **kwargs):
        tocompare = [self]
        tocompare.extend(othermetrics)

        for i, m in enumerate(tocompare):
            if i==0:
                f, ax = m.visualize(compare=True,**kwargs)
            else:
                f, ax = m.visualize(f=f, ax=ax, compare=True,**kwargs)

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax


class FQuenchedLum(Metric):

    def __init__(self, ministry, zbins=[0.0, 0.2], magbins=None, m=0.0, b=0.8, catalog_type=['galaxycatalog']):
        Metric.__init__(self, ministry, catalog_type=catalog_type)
        self.zbins = zbins

        if zbins is None:
            self.nzbins = 1
        else:
            self.nzbins = len(zbins)-1
            self.zbins = np.array(self.zbins)

        if magbins is None:
            self.magbins = np.linspace(-25, -18, 30)
        else:
            self.magbins = magbins

        self.m = m
        self.b = b

        self.mapkeys = ['luminosity', 'redshift']
        self.aschema = 'galaxyonly'
        self.unitmap = {'luminosity':'mag'}

    def map(self, mapunit):

        if not hasattr(self, 'qscounts'):
            self.qscounts = np.zeros((len(self.magbins)-1,self.nzbins))
            self.tcounts = np.zeros((len(self.magbins)-1,self.nzbins))

        if self.zbins is not None:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

                for j, lum in enumerate(self.magbins[:-1]):
                    lidx, = np.where((self.magbins[j]<mapunit['luminosity'][zlidx:zhidx,0])
                                    & (mapunit['luminosity'][zlidx:zhidx,0]<self.magbins[j+1]))
                    qidx, = np.where((mapunit['luminosity'][zlidx:zhidx,0][lidx]
                                     - mapunit['luminosity'][zlidx:zhidx,1][lidx])
                                    > (self.m * mapunit['luminosity'][zlidx:zhidx,0][lidx] 
                                       + self.b))

                    self.qscounts[j,i] = len(qidx)
                    self.tcounts[j,i] = len(lidx)

        else:
            for i, lum in enumerate(self.magbins[:-1]):
                lidx, = np.where((self.magbins[i]<mapunit['luminosity'][:,0])
                                & (mapunit['luminosity'][:,0]<self.magbins[i+1]))
                qidx, = np.where((mapunit['luminosity'][:,0][lidx]
                                 - mapunit['luminosity'][:,1][lidx])
                                > (self.m * mapunit['luminosity'][:,0][lidx] 
                                   + self.b))

                self.qscounts[i] = len(qidx)
                self.tcounts[i] = len(lidx)
        
                
    def reduce(self):
        self.fquenched = self.qscounts/self.tcounts

    def visualize(self, f=None, ax=None, plotname=None, 
                  compare=False, **kwargs):
        
        if f is None:
            f, ax = plt.subplots(1,self.nzbins, figsize=(8,8))
            ax = np.atleast_2d(ax)
            newaxes = True
        else:
            newaxes = False

        lm = (self.magbins[:-1]+self.magbins[1:])/2
        for i in range(self.nzbins):
            ax[0][i].plot(lm, self.fquenched[:,i])

        if newaxes:
            sax = f.add_subplot(111)
            sax.patch.set_alpha(0.0)
            sax.patch.set_facecolor('none')
            sax.spines['top'].set_color('none')
            sax.spines['bottom'].set_color('none')
            sax.spines['left'].set_color('none')
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            sax.set_xlabel(r'$Mag$')
            sax.set_ylabel(r'$f_{red}$')

        plt.tight_layout()

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)


        return f, ax

    def compare(self, othermetrics, plotname=None, **kwargs):
        tocompare = [self]
        tocompare.extend(othermetrics)

        for i, m in enumerate(tocompare):
            if i==0:
                f, ax = m.visualize(compare=True,**kwargs)
            else:
                f, ax = m.visualize(f=f, ax=ax, compare=True, **kwargs)

        if plotname is not None:
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
