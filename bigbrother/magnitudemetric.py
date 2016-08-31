from __future__ import print_function, division
from .metric import Metric, GMetric, jackknifeMap
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pylab as plt
import numpy as np


class MagnitudeMetric(GMetric):
    """
    Restrict GMetric to magnitudes
    """

    def __init__(self, ministry, zbins=None, magbins=None,
                 catalog_type=None, tag=None, **kwargs):
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
                         catalog_type=catalog_type, tag=tag, **kwargs)


class LuminosityFunction(MagnitudeMetric):
    """
    A generic luminosity function class. More specific types of luminosity
    functions inherit this class.
    """

    def __init__(self, ministry, central_only=False, zbins=None, magbins=None,
                 catalog_type=['galaxycatalog'], tag=None, **kwargs):

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
            zbins = np.linspace(ministry.minz, ministry.maxz, 5)

        if magbins is None:
            magbins = np.linspace(-25, -11, 30)

        MagnitudeMetric.__init__(self, ministry, zbins=zbins, magbins=magbins,
                                 catalog_type=catalog_type, tag=tag, **kwargs)

        self.central_only = central_only
        if central_only:
            self.mapkeys = ['luminosity', 'redshift', 'central']
        else:
            self.mapkeys = ['luminosity', 'redshift']

        self.aschema = 'galaxyonly'

    @jackknifeMap
    def map(self, mapunit):
        """
        A simple example of what a map function should look like.
        Map functions always take mapunits as input.
        """

        #The number of bands to measure the LF for
        if len(mapunit['luminosity'].shape)>1:
            self.nbands = mapunit['luminosity'].shape[1]
        else:
            mapunit['luminosity'] = np.atleast_2d(mapunit['luminosity']).T
            self.nbands = 1

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
            self.lumcounts = np.zeros((self.njack, len(self.magbins)-1,
                                        self.nbands, self.nzbins))

        #Assume redshifts are provided, and that the
        #mapunit is sorted in terms of them
        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mu['redshift'].searchsorted(self.zbins[i])
            zhidx = mu['redshift'].searchsorted(self.zbins[i+1])

            #Count galaxies in bins of luminosity
            for j in range(self.nbands):
                c, e = np.histogram(mu['luminosity'][zlidx:zhidx,j],
                                    bins=self.magbins)
                self.lumcounts[self.jcount,:,j,i] += c


    def reduce(self, rank=None, comm=None):
        """
        Given counts in luminosity bins, generate a luminosity function.
        This will be called after all the mapunits are mapped by the map
        method. This turns total counts of galaxies into densities as appropriate
        for a luminosity function. The LF is then saved as an attribute of the
        LuminosityFunction object.
        """

        if rank is not None:
            gdata = comm.gather(self.lumcounts, root=0)

            gshape = [self.lumcounts.shape[i] for i in range(len(self.lumcounts.shape))]
            gshape[0] = self.njacktot

            if rank==0:
                self.lumcounts = np.zeros(gshape)
                jc = 0
                #iterate over gathered arrays, filling in arrays of rank==0
                #process
                for g in gdata:
                    nj = g.shape[0]
                    self.lumcounts[jc:jc+nj,:,:,:] = g

                    jc += nj

                area = self.ministry.galaxycatalog.getArea(jackknife=True)
                vol = np.zeros((self.njacktot, self.nzbins))
                for i in range(self.nzbins):
                    vol[:,i] = self.ministry.calculate_volume(area, self.zbins[i], self.zbins[i+1])

                self.jlumcounts  = self.jackknife(self.lumcounts, reduce_jk=False)
                self.jluminosity_function = self.jlumcounts / vol.reshape(self.njacktot, 1, 1, -1)

                self.luminosity_function = np.sum(self.jluminosity_function, axis=0) / self.njacktot
                self.varluminosity_function = np.sum((self.jluminosity_function - self.luminosity_function) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot
                self.y = self.luminosity_function
                self.ye = np.sqrt(self.varluminosity_function)
        else:
            area = self.ministry.galaxycatalog.getArea(jackknife=True)
            vol = np.zeros((self.njacktot, self.nzbins))
            for i in range(self.nzbins):
                vol[:,i] = self.ministry.calculate_volume(area, self.zbins[i], self.zbins[i+1])

            self.jlumcounts  = self.jackknife(self.lumcounts, reduce_jk=False)
            self.jluminosity_function = self.jlumcounts / vol.reshape(self.njacktot, 1, 1, -1)

            self.luminosity_function = np.sum(self.jluminosity_function, axis=0) / self.njacktot
            self.varluminosity_function = np.sum((self.jluminosity_function - self.luminosity_function) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot
            self.y = self.luminosity_function
            self.ye = np.sqrt(self.varluminosity_function)


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

        return MagnitudeMetric.visualize(self, plotname=plotname, usecols=usecols, usez=usez,
                                         fracdev=fracdev, ref_y=ref_y, ref_x=ref_x, xlim=xlim,
                                         ylim=ylim, fylim=fylim, f=f, ax=ax, xlabel=xlabel,
                                         ylabel=ylabel, compare=compare, **kwargs)

class MagCounts(MagnitudeMetric):
    """
    Galaxy counts per magnitude.
    """

    def __init__(self, ministry, zbins=[0.0, 0.2],  magbins=None,
                 catalog_type=['galaxycatalog'], tag=None, cumulative=False, **kwargs):

        if magbins is None:
            magbins = np.linspace(10, 30, 60)

        MagnitudeMetric.__init__(self,ministry, zbins=zbins, magbins=magbins,
                                 catalog_type=catalog_type, tag=tag, **kwargs)

        if (zbins is not None):
            self.mapkeys = ['appmag', 'redshift']
        else:
            self.mapkeys = ['appmag']

        self.cumulative = cumulative

        self.aschema = 'galaxyonly'
        self.mc = None

    @jackknifeMap
    def map(self, mapunit):
        self.nbands = mapunit['appmag'].shape[1]
        if self.mc is None:
            self.mc = np.zeros((self.njack,
                                  len(self.magbins)-1,
                                  self.nbands, self.nzbins))

        if self.zbins is not None:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])
                for j in range(self.nbands):
                    c, e = np.histogram(mapunit['appmag'][zlidx:zhidx,j],
                                        bins=self.magbins)
                    self.mc[self.jcount,:,j,i] += c
        else:
            for j in range(self.nbands):
                print(mapunit['appmag'][np.isnan(mapunit['appmag'][:,j]),j])
                c, e = np.histogram(mapunit['appmag'][:,j], bins=self.magbins)
                self.mc[self.jcount,:,j,0] += c

    def reduce(self, rank=None, comm=None):

        if rank is not None:
            gdata = comm.gather(self.mc, root=0)

            gshape = [self.mc.shape[i] for i in range(len(self.mc.shape))]
            gshape[0] = self.njacktot

            if rank==0:
                self.mc = np.zeros(gshape)
                jc = 0
                for g in gdata:
                    nj = g.shape[0]
                    self.mc[jc:jc+nj,:,:,:] = g

                    jc += nj

                area = self.ministry.galaxycatalog.getArea(jackknife=True)

                if not self.cumulative:
                    self.mc = self.mc
                else:
                    self.mc = np.cumsum(self.mc, axis=1)

                self.jmagcounts = self.jackknife(self.mc, reduce_jk=False) / area.reshape(self.njacktot,1,1,1)

                self.magcounts = np.sum(self.jmagcounts, axis=0) / self.njacktot

                self.varmagcounts = np.sum((self.jmagcounts - self.magcounts)**2, axis=0) * (self.njacktot - 1) / self.njacktot

                self.y = self.magcounts
                self.ye = np.sqrt(np.sqrt(self.varmagcounts))
        else:
            area = self.ministry.galaxycatalog.getArea(jackknife=True)

            if not self.cumulative:
                self.mc = self.mc
            else:
                self.mc = np.cumsum(self.mc, axis=1)

            self.jmagcounts = self.jackknife(self.mc, reduce_jk=False) / area.reshape(self.njacktot,1,1,1)

            self.magcounts = np.sum(self.jmagcounts, axis=0) / self.njacktot

            self.varmagcounts = np.sum((self.jmagcounts - self.magcounts)**2, axis=0) * (self.njacktot - 1) / self.njacktot

            self.y = self.magcounts
            self.ye = np.sqrt(np.sqrt(self.varmagcounts))

    def visualize(self, plotname=None, usecols=None, usez=None,fracdev=False,
                  ref_y=None, ref_x=[None], xlim=None, ylim=None, fylim=None,
                  f=None, ax=None, xlabel=None,ylabel=None,compare=False,**kwargs):

        if xlabel is None:
            xlabel = "Mag"
        if self.cumulative:
            if ylabel is None:
                ylabel = r'$N(>m) \, [deg^{-2}]$'
        else:
            if ylabel is None:
                ylabel = r'$n \, [mag^{-1}\, deg^{-2}]$'

        return MagnitudeMetric.visualize(self, plotname=plotname, usecols=usecols, usez=usez,
                                         fracdev=fracdev, ref_y=ref_y, ref_x=ref_x, xlim=xlim,
                                         ylim=ylim, fylim=fylim, f=f, ax=ax, xlabel=xlabel,
                                         ylabel=ylabel, compare=compare, **kwargs)

class LcenMass(Metric):
    """
    Central galaxy luminosity - halo virial mass relation.
    """
    def __init__(self, ministry, zbins=None, massbins=None,
                 catalog_type=['galaxycatalog'], tag=None, **kwargs):
        Metric.__init__(self, ministry, catalog_type=catalog_type, tag=tag, **kwargs)

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

    @jackknifeMap
    def map(self, mapunit):

        self.nbands = mapunit['luminosity'].shape[1]

        mu = {}

        for k in mapunit.keys():
            mu[k] = mapunit[k][mapunit['central']==1]


        if not hasattr(self, 'totlum'):
            self.totlum = np.zeros((self.njack, len(self.massbins)-1,
                                      self.nbands, len(self.zbins)-1))
            self.bincount = np.zeros((self.njack, len(self.massbins)-1,
                                        self.nbands, len(self.zbins)-1))

        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mu['redshift'].searchsorted(self.zbins[i])
            zhidx = mu['redshift'].searchsorted(self.zbins[i+1])
            mb = np.digitize(mu['halomass'][zlidx:zhidx], bins=self.massbins)

            for j in range(len(self.massbins)-1):
                blum = mu['luminosity'][zlidx:zhidx,:][mb==j]
                self.bincount[self.jcount,j,:,i] += len(blum)
                self.totlum[self.jcount,j,:,i] += np.sum(blum, axis=0)


    def reduce(self, rank=None, comm=None):

        if rank is not None:
            gtotlum = comm.gather(self.totlum, root=0)
            gbincount = comm.gather(self.bincount, root=0)

            tshape = [self.totlum.shape[i] for i in range(len(self.totlum.shape))]
            bshape = [self.bincount.shape[i] for i in range(len(self.bincount.shape))]

            tshape[0] = self.njacktot
            bshape[0] = self.njacktot

            if rank==0:
                self.bincount = np.zeros(tshape)
                self.totlum = np.zeros(bshape)

                jc = 0
                for i, g in enumerate(gtotlum):
                    nj = g.shape[0]
                    self.totlum[jc:jc+nj,:,:,:] = g
                    self.bincount[jc:jc+nj,:,:,:] = gbincount[i]

                    jc += nj

                self.jblcen_mass, self.lcen_mass, self.varlcen_mass = self.jackknife(self.totlum/self.bincount)
        else:
            self.jblcen_mass, self.lcen_mass, self.varlcen_mass = self.jackknife(self.totlum/self.bincount)

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

        #plt.tight_layout()

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

class ColorDist(GMetric):

    def __init__(self, ministry, zbins=None, cbins=None,
                    catalog_type=['galaxycatalog'],
                    usebands=None, appmag=False, amagcut=None,
                    pdf=False, cutind=None, **kwargs):

        self.pdf = pdf

        if zbins is None:
            zbins = np.linspace(ministry.minz,
                                  ministry.maxz,
                                  10)

        if cbins is None:
            self.cbins = np.linspace(-2, 4, 101)
        else:
            self.cbins = cbins
        self.ncbins = len(self.cbins) - 1

        GMetric.__init__(self, ministry,
                            catalog_type=catalog_type,
                            xbins=self.cbins,zbins=zbins,
                            **kwargs)

        if appmag:
            self.mkey = 'appmag'
        else:
            self.mkey = 'luminosity'
            if amagcut is None:
                self.amagcut = -19
            else:
                self.amagcut = amagcut

            if cutind is None:
                self.cutind = 0
            else:
                self.cutind = cutind

        if usebands is None:
            self.usebands = [[0, 1]]
        else:
            self.usebands = usebands

        self.ncolors = len(self.usebands)
        self.nbands = self.ncolors

        self.cd = None

        self.mapkeys = [self.mkey, 'redshift']
        self.aschema = 'galaxyonly'
        self.unitmap = {self.mkey:'mag'}

    @jackknifeMap
    def map(self, mapunit):

        if self.cd is None:
            self.cd = np.zeros((self.njack, self.ncbins,
                                self.ncolors, self.nzbins))

        clr = np.zeros((len(mapunit[self.mkey]),
                        self.ncolors))
        for c in range(self.ncolors):
            clr[:,c] = mapunit[self.mkey][:,self.usebands[c][0]] - mapunit[self.mkey][:,self.usebands[c][1]]

        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])
            if self.mkey == 'luminosity':
                lidx = mapunit[self.mkey][zlidx:zhidx,self.cutind] < self.amagcut
            else:
                lidx = slice(0,zhidx-zlidx)

                for ci in range(self.ncolors):
                    c, self.cbins = np.histogram(clr[zlidx:zhidx,ci][lidx], bins=self.cbins)

                    self.cd[self.jcount,:,ci,i] += c

    def reduce(self, rank=None, comm=None):

        if rank is not None:
            gcd = comm.gather(self.cd, root=0)

            gshape = [self.cd.shape[i] for i in range(len(self.cd.shape))]
            gshape[0] = self.njacktot

            if rank==0:
                self.cd = np.zeros(gshape)
                jc = 0
                for i, g in enumerate(gcd):
                    nj = g.shape[0]
                    self.cd[jc:jc+nj,:,:,:] = g

                    jc += nj
                if self.pdf:
                    self.tc = np.sum(self.cd, axis=1)
                    self.jcd = self.jackknife(self.cd, reduce_jk=False)
                    self.jtc = self.jackknife(self.tc, reduce_jk=False)
                    dc = self.cbins[1:] - self.cbins[:-1]
                    self.jcolor_dist = self.jcd / self.jtc / dc.reshape(1,self.ncbins,1,1)
                else:
                    area = self.ministry.galaxycatalog.getArea(jackknife=True)
                    self.jcd = self.jackknife(self.cd, reduce_jk=False)
                    self.jcolor_dist = self.jcd / area.reshape(self.njacktot, 1, 1, 1)

                self.color_dist = np.sum(self.jcolor_dist, axis=0) / self.njacktot
                self.varcolor_dist = np.sum((self.jcolor_dist - self.color_dist) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot

                self.y = self.color_dist
                self.ye = np.sqrt(self.varcolor_dist)
        else:
            if self.pdf:
                self.tc = np.sum(self.cd, axis=1)
                self.jcd = self.jackknife(self.cd, reduce_jk=False)
                self.jtc = self.jackknife(self.tc, reduce_jk=False)
                dc = self.cbins[1:] - self.cbins[:-1]
                self.jcolor_dist = self.jcd / self.jtc / dc.reshape(1, self.ncbins, 1, 1)

            else:
                area = self.ministry.galaxycatalog.getArea(jackknife=True)
                self.jcd = self.jackknife(self.cd, reduce_jk=False)
                self.jcolor_dist = self.jcd / area.reshape(self.njacktot, 1, 1, 1)

            self.color_dist = np.sum(self.jcolor_dist, axis=0) / self.njacktot
            self.varcolor_dist = np.sum((self.jcolor_dist - self.color_dist) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot

            self.y = self.color_dist
            self.ye = np.sqrt(self.varcolor_dist)


class ColorColor(Metric):
    """
    Color-color diagram.
    """
    def __init__(self, ministry, zbins=[0.0, 0.2], cbins=None,
                 catalog_type=['galaxycatalog'],
                 usebands=None,
                 amagcut=-19.0, tag=None, appmag=False,
                 pdf=False, **kwargs):
        Metric.__init__(self, ministry, catalog_type=catalog_type, tag=tag, **kwargs)

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

        if appmag:
            self.mkey = 'appmag'
        else:
            self.mkey = 'luminosity'

        if (zbins is not None):
            self.mapkeys = [self.mkey, 'redshift']
        else:
            self.mapkeys = [self.mkey]

        self.pdf = pdf

        self.amagcut = amagcut
        self.usebands = usebands
        self.aschema = 'galaxyonly'
        self.unitmap = {self.mkey:'mag'}

    @jackknifeMap
    def map(self, mapunit):

        if self.usebands is None:
            self.nbands = mapunit[self.mkey].shape[1]
            self.usebands = range(self.nbands)
        else:
            self.nbands = len(self.usebands)

        self.nclr = self.nbands-1

        clr = np.zeros((len(mapunit[self.mkey]),self.nbands-1))
        for i, b in enumerate(self.usebands[:-1]):
            clr[:,i] = mapunit[self.mkey][:,self.usebands[i]] - mapunit[self.mkey][:,self.usebands[i+1]]

        if not hasattr(self, 'cc'):
            self.cc = np.zeros((self.njack,len(self.cbins)-1, len(self.cbins)-1,
                                self.nbands-2, self.nzbins))

        if self.zbins is not None:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

                if self.amagcut!=None:
                    for e, j in enumerate(self.usebands):
                        if e==0:
                            lidx = mapunit[self.mkey][zlidx:zhidx,j]<self.amagcut
                        else:
                            lix = mapunit[self.mkey][zlidx:zhidx,j]<self.amagcut
                            lidx = lidx & lix

                for j in range(self.nclr-1):
                    c, e0, e1 = np.histogram2d(clr[zlidx:zhidx,j+1][lidx],
                                               clr[zlidx:zhidx,j][lidx],
                                               bins=self.cbins)
                    self.cc[self.jcount,:,:,j,i] += c
        else:
            for i in range(self.nclr-1):
                c, e0, e1 = np.histogram2d(clr[:,i],
                                           clr[:,i+1],
                                           bins=self.cbins)
                self.cc[self.jcount,:,:,i,0] += c


    def reduce(self, rank=None, comm=None):
        if rank is not None:
            gcc = comm.gather(self.cc, root=0)

            gshape = [self.cc.shape[i] for i in range(len(self.cc.shape))]
            gshape[0] = self.njacktot

            if rank==0:
                self.cc = np.zeros(gshape)
                jc = 0
                for i, g in enumerate(gcc):
                    nj = g.shape[0]
                    self.cc[jc:jc+nj,:,:,:] = g

                    jc += nj

                if self.pdf:
                    self.tc = np.sum(np.sum(self.cc, axis=1), axis=1).reshape(-1,1,1,self.nbands-2,self.nzbins)
                    self.jcc = self.jackknife(self.cc, reduce_jk=False)
                    self.jtc = self.jackknife(self.tc, reduce_jk=False)
                    dc = self.cbins[1:] - self.cbins[:-1]
                    dc = np.outer(dc, dc)
                    self.jcolor_color = self.jcc / self.jtc / dc.reshape(-1,self.ncbins,self.ncbins, 1, 1)
                else:
                    area = self.ministry.galaxycatalog.getArea(jackknife=True)
                    self.jcc = self.jackknife(self.cc, reduce_jk=False)
                    self.jcolor_color = self.jcc / area.reshape(self.njacktot,1,1,1,1)

                self.color_color = np.sum(self.jcolor_color, axis=0) / self.njacktot
                self.varcolor_color = np.sum((self.jcolor_color - self.color_color) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot
        else:
            if self.pdf:
                self.tc = np.sum(np.sum(self.cc, axis=1), axis=1).reshape(-1,1,1,self.nbands-2,self.nzbins)
                self.jcc = self.jackknife(self.cc, reduce_jk=False)
                self.jtc = self.jackknife(self.tc, reduce_jk=False)
                dc = self.cbins[1:] - self.cbins[:-1]
                dc = np.outer(dc, dc)
                self.jcolor_color = self.jcc / self.jtc / dc.reshape(1,self.ncbins,self.ncbins,1,1)
            else:
                area = self.ministry.galaxycatalog.getArea(jackknife=True)
                self.jcc = self.jackknife(self.cc, reduce_jk=False)
                self.jcolor_color = self.jcc / area.reshape(self.njacktot,1,1,1,1)

            self.color_color = np.sum(self.jcolor_color, axis=0) / self.njacktot
            self.varcolor_color = np.sum((self.jcolor_color - self.color_color) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot


    def visualize(self, compare=False, plotname=None, f=None, ax=None,
                  usecolors=None, **kwargs):

        if hasattr(self, 'magmean'):
            mclr = self.mclr
        else:
            mclr = np.array([(self.cbins[i]+self.cbins[i+1])/2
                              for i in range(len(self.cbins)-1)])

        if usecolors is None:
            usecolors = range(self.color_color.shape[2])

        if f is None:
            f, ax = plt.subplots(self.nzbins, len(usecolors),
                                 sharex=True, sharey=True, figsize=(8,8))
            ax = np.array(ax)
            ax = ax.reshape(self.nzbins, len(usecolors))
            newaxes = True
        else:
            newaxes = False

        X, Y = np.meshgrid(mclr, mclr)

        for i in usecolors:
            for j in range(self.nzbins):
                l1 = ax[j][i].contour(X, Y, self.color_color[:,:,i,j].T, 10,
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

        #plt.tight_layout()

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)


        return f, ax, l1

    def compare(self, othermetrics, plotname=None, usecolors=None,
                  labels=None, **kwargs):

        tocompare = [self]
        tocompare.extend(othermetrics)

        if usecolors!=None:
            if not hasattr(usecolors[0], '__iter__'):
                usecolors = [usecolors]*len(tocompare)
            else:
                assert(len(usecolors)==len(tocompare))
        else:
            usecolors = [None]*len(tocompare)

        if labels is None:
            labels = [None] * len(tocompare)

        lines = []

        for i, m in enumerate(tocompare):
            if i==0:
                f, ax, l = m.visualize(usecolors=usecolors[i], compare=True,
                                    **kwargs)
            else:
                f, ax, l = m.visualize(usecolors=usecolors[i], compare=True,
                                        ax=ax, f=f, **kwargs)

            lines.append(l)


        if labels[0]!=None:
            f.legend(lines, labels)

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax

class ColorMagnitude(Metric):
    """
    Color-magnitude diagram.
    """
    def __init__(self, ministry, zbins=[0.0, 0.2],
                  magbins=None, cbins=None,
                  central_only=False, logscale=False,
                  catalog_type=['galaxycatalog'],
                  usebands=None, tag=None, appmag=False,
                  pdf=False, **kwargs):

        Metric.__init__(self, ministry, catalog_type=catalog_type, tag=tag, **kwargs)

        self.zbins = zbins
        if zbins is None:
            self.nzbins = 1
        else:
            self.nzbins = len(zbins)-1
            self.zbins = np.array(self.zbins)

        if appmag:
            self.mkey = 'appmag'
            defmbins = np.linspace(15,28,60)
        else:
            self.mkey = 'luminosity'
            defmbins = np.linspace(-25,-19,60)


        if (magbins is None) & (cbins is None):
            self.magbins = defmbins
            self.cbins = np.linspace(-1,2,60)
        elif magbins is None:
            self.magbins = defmbins
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

        self.pdf = pdf

        self.central_only = central_only
        self.logscale = logscale


        if central_only & (zbins is not None):
            self.mapkeys = [self.mkey, 'redshift', 'central']
        elif (zbins is not None):
            self.mapkeys = [self.mkey, 'redshift']
        else:
            self.mapkeys = [self.mkey]


        self.aschema = 'galaxyonly'
        self.unitmap = {self.mkey:'mag'}

    @jackknifeMap
    def map(self, mapunit):

        if self.usebands is None:
            self.nbands = mapunit[self.mkey].shape[1]
            self.usebands = range(self.nbands)

        mu = {}
        if self.central_only:
            for k in mapunit.keys():
                mu[k] = mapunit[k][mapunit['central']==1]
        else:
            mu = mapunit


        if not hasattr(self, 'cc'):
            self.cc = np.zeros((self.njack, len(self.magbins)-1,
                                len(self.cbins)-1,
                                int(self.nbands*(self.nbands-1)/2),
                                self.nzbins))

        if self.zbins is not None:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mu['redshift'].searchsorted(self.zbins[i])
                zhidx = mu['redshift'].searchsorted(self.zbins[i+1])
                for j, b1 in enumerate(self.usebands):
                    for k, b2 in enumerate(self.usebands):
                        if k<=j: continue
                        ind = int(k*(k-1)//2+j-1)
                        c, e0, e1 = np.histogram2d(mu[self.mkey][zlidx:zhidx,b1],
                                                   mu[self.mkey][zlidx:zhidx,b1] -
                                                   mu[self.mkey][zlidx:zhidx,b2],
                                                   bins=[self.magbins,self.cbins])
                        self.cc[self.jcount,:,:,ind,i] += c
        else:
            for j, b1 in enumerate(self.usebands):
                for k, b2 in enumerate(self.usebands):
                    if k<=j: continue
                    ind = int(k*(k-1)//2+j-1)
                    c, e0, e1 = np.histogram2d(mu[self.mkey][:,b1],
                                               mu[self.mkey][:,b1] -
                                               mu[self.mkey][:,b2],
                                               bins=[self.magbins,self.cbins])
                    self.cc[self.jcount,:,:,ind,0] += c

    def reduce(self, rank=None, comm=None):
        if rank is not None:
            gcc = comm.gather(self.cc, root=0)

            if rank==0:
                jc = 0
                dshape = self.cc.shape
                dshape = [dshape[i] for i in range(len(dshape))]
                dshape[0] = self.njacktot

                for i, g in enumerate(gcc):
                    self.cc = np.zeros(dshape)
                    nj = g.shape[0]
                    self.cc[jc:jc+nj,:,:,:,:] = g

                    jc += nj

                if self.pdf:
                    self.tc = np.sum(np.sum(self.cc, axis=1), axis=1).reshape(-1,1,1,int(self.nbands*(self.nbands-1)/2),self.nzbins)
                    self.jcc = self.jackknife(self.cc, reduce_jk=False)
                    self.jtc = self.jackknife(self.tc, reduce_jk=False)
                    dc = self.cbins[1:] - self.cbins[:-1]
                    dm = self.magbins[1:] - self.magbins[:-1]
                    dcdm = np.outer(dc,dm).reshape(-1,self.ncbins, self.nmagbins,1,1)
                    self.jcolor_mag = self.jcc / self.tc / dcdm
                else:
                    area = self.ministry.galaxycatalog.getArea(jackknife=True)
                    self.jcc = self.jackknife(self.cc, reduce_jk=False)
                    self.jcolor_mag = self.jcc / area.reshape(self.njacktot,1,1,1,1)

                self.color_mag = np.sum(self.jcolor_mag, axis=0) / self.njacktot
                self.varcolor_mag = np.sum((self.jcolor_mag - self.color_mag) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot

        else:
            if self.pdf:
                self.tc = np.sum(np.sum(self.cc, axis=1), axis=1).reshape(-1,1,1,int(self.nbands*(self.nbands-1)/2),self.nzbins)
                self.jcc = self.jackknife(self.cc, reduce_jk=False)
                self.jtc = self.jackknife(self.tc, reduce_jk=False)
                dc = self.cbins[1:] - self.cbins[:-1]
                dm = self.magbins[1:] - self.magbins[:-1]
                dcdm = np.outer(dc,dm).reshape(-1,self.ncbins, self.nmagbins,1,1)
                self.jcolor_mag = self.jcc / self.tc / dcdm
            else:
                area = self.ministry.galaxycatalog.getArea(jackknife=True)
                self.jcc = self.jackknife(self.cc, reduce_jk=False)
                self.jcolor_mag = self.jcc / area.reshape(self.njacktot,1,1,1,1)

            self.color_mag = np.sum(self.jcolor_mag, axis=0) / self.njacktot
            self.varcolor_mag = np.sum((self.jcolor_mag - self.color_mag) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot


    def visualize(self, plotname=None, f=None, ax=None, usecolors=None,
                  compare=False, **kwargs):

        x = (self.magbins[:-1]+self.magbins[1:])/2
        y = (self.cbins[:-1]+self.cbins[1:])/2
        X, Y = np.meshgrid(x, y)

        if self.logscale:
            cc = np.log10(self.color_mag)
            cc[cc==(-np.inf)] = 0.0
        else:
            cc = self.color_mag

        if usecolors is None:
            usecolors = range(self.color_mag.shape[2])

        if f is None:
            f, ax = plt.subplots(self.nzbins, len(usecolors),
                                 sharex=True, sharey=True, figsize=(8,8))
            ax = np.array(ax)
            ax = ax.reshape(self.nzbins, len(usecolors))
            newaxes = True
        else:
            newaxes = False

        for i, c in enumerate(usecolors):
            for j in range(self.nzbins):
                l1 = ax[j][i].contour(X, Y, cc[:,:,c,j].T,10,
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

        #plt.tight_layout()

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax, l1

    def compare(self, othermetrics, plotname=None, usecolors=None,
                 labels=None, **kwargs):
        tocompare = [self]
        tocompare.extend(othermetrics)

        if usecolors is not None:
            if not hasattr(usecolors[0], '__iter__'):
                usecolors = [usecolors]*len(tocompare)
            else:
                assert(len(usecolors)==len(tocompare))
        else:
            usecolors = [None]*len(tocompare)

        if labels is None:
            labels = [None]*len(tocompare)

        lines = []

        for i, m in enumerate(tocompare):
            if usecolors[i] is not None:
                assert(len(usecolors[0])==len(usecolors[i]))
            if i==0:
                f, ax, l1 = m.visualize(usecolors=usecolors[i], compare=True,
                                    **kwargs)
            else:
                f, ax, l1 = m.visualize(usecolors=usecolors[i], compare=True,
                                    f=f, ax=ax, **kwargs)
            lines.append(l1)

        if labels[0]!=None:
            f.legend(lines, labels)

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax

class FQuenched(Metric):

    def __init__(self, ministry, zbins=[0.0, 0.2],
                  catalog_type=['galaxycatalog'],
                  tag=None, appmag=True, magind=None,
                  hcbins=None, **kwargs):

        Metric.__init__(self, ministry, catalog_type=catalog_type,tag=tag,**kwargs)
        self.zbins = zbins

        if zbins is None:
            self.nzbins = 1
        else:
            self.nzbins = len(zbins)-1
            self.zbins = np.array(self.zbins)

        self.splitcolor = np.zeros(self.nzbins)

        if magind is None:
            self.magind = [0,1]
        else:
            self.magind = magind

        if hcbins is None:
            self.hcbins = 100
        else:
            self.hcbins = hcbins

        if appmag:
            self.mkey = 'appmag'
        else:
            self.mkey = 'luminosity'

        self.mapkeys = [self.mkey, 'redshift']
        self.unitmap = {self.mkey:'mag'}
        self.aschema = 'galaxyonly'

    @jackknifeMap
    def map(self, mapunit):

        if not hasattr(self, 'qscounts'):
            self.qscounts = np.zeros((self.njack, self.nzbins))
            self.tcounts = np.zeros((self.njack,self.nzbins))

        clr = mapunit[self.mkey][:,self.magind[0]] - mapunit[self.mkey][:,self.magind[1]]

        if self.zbins is not None:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

                ccounts, cbins = np.histogram(clr[zlidx:zhidx], self.hcbins)

                self.splitcolor[i] = self.splitBimodal(cbins[:-1], ccounts)
                if self.splitcolor[i] is None:
                    continue

                qidx, = np.where(clr[zlidx:zhidx]>self.splitcolor[i])

                self.qscounts[self.jcount,i] = len(qidx)
                self.tcounts[self.jcount,i] = zhidx-zlidx

        else:
            ccounts, cbins = np.histogram(clr[zlidx:zhidx], self.hcbins)

            self.splitcolor[i] = self.splitBimodal(cbins[:-1], ccounts)
            if self.splitcolor[i] is None:
                return

            qidx, = np.where(clr[zlidx:zhidx]>self.splitcolor[i])

            self.qscounts[self.jcount,0] = len(qidx)
            self.tcounts[self.jcount,0] = len(mapunit[self.mkey])


    def reduce(self, rank=None, comm=None):
        if rank is not None:
            gqs = comm.gather(self.qscounts, root=0)
            gtc = comm.gather(self.tcounts, root=0)

            qcshape = [self.qscounts.shape[i] for i in range(len(self.qscounts.shape))]
            tcshape = [self.tcounts.shape[i] for i in range(len(self.tcounts.shape))]

            qcshape[0] = self.njacktot
            tcshape[0] = self.njacktot

            if rank==0:
                self.qscounts = np.zeros(qcshape)
                self.tcounts = np.zeros(tcshape)

                jc = 0
                for i, g in enumerate(gqs):
                    nj = g.shape[0]
                    self.qscounts[jc:jc+nj,:] = g
                    self.tcounts[jc:jc+nj,:] = gtc[i]
                    jc += nj

                self.jqscounts = self.jackknife(self.qscounts, reduce_jk=False)
                self.jtcounts = self.jackknife(self.tcounts, reduce_jk=False)

                self.jfquenched = self.jqscounts / self.jtcounts
                self.fquenched = np.sum(self.jfquenched, axis=0) / self.njacktot
                self.varfquenched = np.sum((self.jfquenched - self.fquenched)**2, axis=0) * ( self.njacktot - 1) / self.njacktot

        else:
            self.jfquenched = self.jqscounts / self.jtcounts
            self.fquenched = np.sum(self.jfquenched, axis=0) / self.njacktot
            self.varfquenched = np.sum((self.jfquenched - self.fquenched)**2, axis=0) * ( self.njacktot - 1) / self.njacktot


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

        #plt.tight_layout()

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

    def __init__(self, ministry, zbins=[0.0, 0.2], catalog_type=['galaxycatalog'], zeroind=True,
                  tag=None, **kwargs):
        Metric.__init__(self, ministry, catalog_type=catalog_type, tag=tag, **kwargs)
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

    @jackknifeMap
    def map(self, mapunit):

        if not hasattr(self, 'nred'):
            self.qscounts = np.zeros((self.njack,self.nzbins))
            self.tcounts = np.zeros((self.njack,self.nzbins))

        if self.zbins is not None:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

                if self.zeroind:
                    qidx, = np.where(self.ctcat[mapunit['ctcatid'][zlidx:zhidx]-1,3]==1)
                else:
                    qidx, = np.where(self.ctcat[mapunit['ctcatid'][zlidx:zhidx],3]==1)

                self.qscounts[self.jcount, i] = len(qidx)
                self.tcounts[self.jcount, i] = zhidx-zlidx

        else:
            if self.zeroind:
                qidx, = np.where(self.ctcat[mapunit['ctcatid']-1,3]==1)
            else:
                qidx, = np.where(self.ctcat[mapunit['ctcatid'],3]==1)

            self.qscounts[self.jcount,0] = len(qidx)
            self.tcounts[self.jcount,0] = len(mapunit['ctcatid'])


    def reduce(self, rank=None, comm=None):
        if rank is not None:
            gqs = comm.gather(self.qscounts, root=0)
            gtc = comm.gather(self.tcounts, root=0)

            qcshape = [self.qscounts.shape[i] for i in range(len(self.qscounts.shape))]
            tcshape = [self.tcounts.shape[i] for i in range(len(self.tcounts.shape))]

            qcshape[0] = self.njacktot
            tcshape[0] = self.njacktot

            if rank==0:
                self.qscounts = np.zeros(qcshape)
                self.tcounts = np.zeros(tcshape)

                jc = 0
                for i, g in enumerate(gqs):
                    nj = g.shape[0]
                    self.qscounts[jc:jc+nj,:] = g
                    self.tcounts[jc:jc+nj,:] = gtc[i]
                    jc += nj

                self.jqscounts = self.jackknife(self.qscounts, reduce_jk=False)
                self.jtcounts = self.jackknife(self.tcounts, reduce_jk=False)

                self.jfquenched = self.jqscounts / self.jtcounts
                self.fquenched = np.sum(self.jfquenched, axis=0) / self.njacktot
                self.varfquenched = np.sum((self.jfquenched - self.fquenched)**2, axis=0) * ( self.njacktot - 1) / self.njacktot

        else:
            self.jqscounts = self.jackknife(self.qscounts, reduce_jk=False)
            self.jtcounts = self.jackknife(self.tcounts, reduce_jk=False)

            self.jfquenched = self.jqscounts / self.jtcounts
            self.fquenched = np.sum(self.jfquenched, axis=0) / self.njacktot
            self.varfquenched = np.sum((self.jfquenched - self.fquenched)**2, axis=0) * ( self.njacktot - 1) / self.njacktot


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

        #plt.tight_layout()

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

    def __init__(self, ministry, zbins=[0.0, 0.2], magbins=None,
                 catalog_type=['galaxycatalog'], tag=None,
                 cbins=None, cinds=None, **kwargs):
        Metric.__init__(self, ministry, catalog_type=catalog_type,tag=tag, **kwargs)
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

        self.splitcolor = None

        if cbins is None:
            self.cbins = 50
            self.ncbins = 50
        else:
            self.cbins = cbins
            self.ncbins = len(cbins)

        if cinds is None:
            self.cinds = [0, 1]
        else:
            self.cinds = cinds

        self.mapkeys = ['luminosity', 'redshift']
        self.aschema = 'galaxyonly'
        self.unitmap = {'luminosity':'mag'}

    @jackknifeMap
    def map(self, mapunit):

        if not hasattr(self, 'qscounts'):
            self.qscounts = np.zeros((self.njack,
                                       len(self.magbins)-1,self.nzbins))
            self.tcounts = np.zeros((self.njack,
                                      len(self.magbins)-1,self.nzbins))

        clr = mapunit['luminosity'][:,self.cinds[0]] - mapunit['luminosity'][:,self.cinds[1]]

        if self.zbins is not None:

            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

                if (self.splitcolor is None):
                    ccounts, self.cbins = np.histogram(clr[zlidx:zhidx], self.cbins)
                    self.splitcolor = self.splitBimodal(self.cbins[:-1], ccounts)
                    if self.splitcolor is None:
                        continue


                for j, lum in enumerate(self.magbins[:-1]):
                    lidx, = np.where((self.magbins[j]<mapunit['luminosity'][zlidx:zhidx,0])
                                    & (mapunit['luminosity'][zlidx:zhidx,0]<self.magbins[j+1]))
                    qidx, = np.where(clr[zlidx:zhidx][lidx]>self.splitcolor)

                    self.qscounts[self.jcount,j,i] = len(qidx)
                    self.tcounts[self.jcount,j,i] = len(lidx)

        else:
            if (self.splitcolor is None):
                ccounts, self.cbins = np.histogram(clr, self.cbins)
                self.splitcolor = self.splitBimodal(self.cbins[:-1], ccounts)
                if self.splitcolor is None:
                    return

            for i, lum in enumerate(self.magbins[:-1]):
                lidx, = np.where((self.magbins[i]<mapunit['luminosity'][:,0])
                                & (mapunit['luminosity'][:,0]<self.magbins[i+1]))

                qidx, = np.where(clr[lidx]>self.splitcolor)

                self.qscounts[self.jcount,i,0] = len(qidx)
                self.tcounts[self.jcount,i,0] = len(lidx)


    def reduce(self, rank=None, comm=None):
        if rank is not None:
            gqs = comm.gather(self.qscounts, root=0)
            gtc = comm.gather(self.tcounts, root=0)

            qcshape = [self.qscounts.shape[i] for i in range(len(self.qscounts.shape))]
            tcshape = [self.tcounts.shape[i] for i in range(len(self.tcounts.shape))]

            qcshape[0] = self.njacktot
            tcshape[0] = self.njacktot

            if rank==0:
                self.qscounts = np.zeros(qcshape)
                self.tcounts = np.zeros(tcshape)

                jc = 0
                for i, g in enumerate(gqs):
                    nj = g.shape[0]
                    self.qscounts[jc:jc+nj,:] = g
                    self.tcounts[jc:jc+nj,:] = gtc[i]
                    jc += nj

                self.jqscounts = self.jackknife(self.qscounts, reduce_jk=False)
                self.jtcounts = self.jackknife(self.tcounts, reduce_jk=False)

                self.jfquenched = self.jqscounts / self.jtcounts
                self.fquenched = np.sum(self.jfquenched, axis=0) / self.njacktot
                self.varfquenched = np.sum((self.jfquenched - self.fquenched)**2, axis=0) * ( self.njacktot - 1) / self.njacktot

        else:
            self.jqscounts = self.jackknife(self.qscounts, reduce_jk=False)
            self.jtcounts = self.jackknife(self.tcounts, reduce_jk=False)

            self.jfquenched = self.jqscounts / self.jtcounts
            self.fquenched = np.sum(self.jfquenched, axis=0) / self.njacktot
            self.varfquenched = np.sum((self.jfquenched - self.fquenched)**2, axis=0) * ( self.njacktot - 1) / self.njacktot


    def visualize(self, f=None, ax=None, plotname=None,
                  compare=False, label=None, **kwargs):

        if f is None:
            f, ax = plt.subplots(1,self.nzbins, figsize=(8,8))
            ax = np.atleast_2d(ax)
            newaxes = True
        else:
            newaxes = False

        lm = (self.magbins[:-1]+self.magbins[1:])/2
        for i in range(self.nzbins):
            l1 = ax[0][i].errorbar(lm, self.fquenched[:,i], yerr=np.sqrt(self.varfquenched[:,i]), label=label, **kwargs)

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

        #plt.tight_layout()

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax, l1[0]

    def compare(self, othermetrics, plotname=None, labels=None,
                  **kwargs):
        tocompare = [self]
        tocompare.extend(othermetrics)

        if labels is None:
            labels = [None] * len(tocompare)

        lines = []

        for i, m in enumerate(tocompare):
            if i==0:
                f, ax, l1 = m.visualize(compare=True,label=labels[i], **kwargs)
            else:
                f, ax, l1 = m.visualize(f=f, ax=ax, compare=True, label=labels[i], **kwargs)
            lines.append(l1)

        if plotname is not None:
            plt.savefig(plotname)

        if labels[0] is not None:
            f.legend(lines, labels, 'best')

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

        self.nomap = True

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
            self.nbands = kwargs.pop('nbands')
        else:
            self.nbands = 5

        if 'xcol' in kwargs:
            self.xcol = int(kwargs.pop('xcol'))
        else:
            self.xcol = 0

        if 'ycol' in kwargs:
            self.ycol = int(kwargs.pop('ycol'))
        else:
            self.ycol = 1

        if 'ecol' in kwargs:
            self.ecol = int(kwargs.pop('ecol'))
        else:
            self.ecol = None

        LuminosityFunction.__init__(self,*args,**kwargs)

        #don't need to map this guy
        self.nomap = True
        self.loadLuminosityFunction()

    def loadLuminosityFunction(self):
        """
        Read in the LF from self.fname. If self.fname is a list
        assumes that LFs in list correspond to zbins specified.
        If self.fname not a list, if more than 2 columns assumes
        first column is luminosities, second column is.
        """

        if len(self.fname)==1:
            tab = np.genfromtxt(self.fname[0])
            self.luminosity_function = np.zeros((tab.shape[0], self.nbands, self.nzbins))
            if len(tab.shape)==2:
                self.magmean = tab[:,self.xcol]
                if self.nzbins==1:
                    for i in range(self.nzbins):
                        for j in range(self.nbands):
                            self.luminosity_function[:,j,i] = tab[:,self.ycol]
                else:
                    assert((tab.shape[1]-1)==self.nzbins)
                    for i in range(self.nzbins):
                        for j in range(self.nbands):
                            self.luminosity_function[:,j,i] = tab[:,i+self.ycol]

            elif len(tab.shape)==3:
                self.magmean = tab[:,0,0]
                self.luminosity_function[:,:,:] = tab[:,1:,:]
        else:
            if len(self.fname.shape)==1:
                assert(self.fname.shape[0]==self.nzbins)
                for i in range(len(self.fname)):
                    lf = np.genfromtxt(self.fname[i])
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
                        lf = np.genfromtxt(self.fname[i,j])
                        if (i==0) & (j==0):
                            self.magmean = lf[:,0]
                            self.luminosity_function = np.zeros((len(self.magmean), self.nbands, self.nzbins))
                        else:
                            assert(self.magmean==lf[:,0])

                        self.luminosity_function[:,j,i] = lf[:,1]

        self.xmean = self.magmean
        self.y = self.luminosity_function
