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
                 catalog_type=['galaxycatalog'], tag=None, CMASS=False,
                 lightcone=True, **kwargs):

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
            magbins = np.linspace(-25, -16, 50)

        self.lightcone = lightcone

        MagnitudeMetric.__init__(self, ministry, zbins=zbins, magbins=magbins,
                                 catalog_type=catalog_type, tag=tag, **kwargs)

        self.central_only = central_only
        self.CMASS = CMASS
        
        if central_only:
            self.mapkeys = ['luminosity', 'central']
        else:
            self.mapkeys = ['luminosity']

        if self.lightcone:
            self.mapkeys.append('redshift')

        if self.CMASS:
            self.mapkeys.append('appmag')

        self.aschema = 'galaxyonly'

        self.lumcounts = None

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
            delete_after_map = True
            for k in mapunit.keys():
                mu[k] = mapunit[k][mapunit['central']==1]
        else:
            delete_after_map = False
            mu = mapunit

        #Want to count galaxies in bins of luminosity for
        #self.nbands different bands in self.nzbins
        #redshift bins
        if self.lumcounts is None:
            self.lumcounts = np.zeros((self.njack, len(self.magbins)-1,
                                        self.nbands, self.nzbins))

        #Assume redshifts are provided, and that the
        #mapunit is sorted in terms of them
        
        if self.lightcone:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mu['redshift'].searchsorted(self.zbins[i])
                zhidx = mu['redshift'].searchsorted(self.zbins[i+1])

            #Count galaxies in bins of luminosity
                for j in range(self.nbands):
                    if not self.CMASS:
                        c, e = np.histogram(mu['luminosity'][zlidx:zhidx,j],
                                            bins=self.magbins)
                    else:
                        cidx = self.selectCMASS(mu['appmag'][zlidx:zhidx])
                        c, e = np.histogram(mu['luminosity'][zlidx:zhidx,j][cidx],
                                            bins=self.magbins)
                    
                    self.lumcounts[self.jcount,:,j,i] += c
        else:
            for j in range(self.nbands):
                if not self.CMASS:
                    c, e = np.histogram(mu['luminosity'][:,j],
                                        bins=self.magbins)
                else:
                    cidx = self.selectCMASS(mu['appmag'][:])
                    c, e = np.histogram(mu['luminosity'][:,j][cidx],
                                        bins=self.magbins)
                    
                self.lumcounts[self.jcount,:,j,0] += c

        if delete_after_map:
            True


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


            if rank==0:
                gshape = [self.lumcounts.shape[i] for i in range(len(self.lumcounts.shape))]
                gshape[0] = self.njacktot

                self.lumcounts = np.zeros(gshape)
                jc = 0
                #iterate over gathered arrays, filling in arrays of rank==0
                #process
                for g in gdata:
                    if g is None: continue
                    nj = g.shape[0]
                    self.lumcounts[jc:jc+nj,:,:,:] = g

                    jc += nj

                area = self.ministry.galaxycatalog.getArea(jackknife=True)
                vol = np.zeros((self.njacktot, self.nzbins))
                dl  = (self.xbins[1:] - self.xbins[:-1]).reshape(1,-1,1,1)
                for i in range(self.nzbins):
                    vol[:,i] = self.ministry.calculate_volume(area, self.zbins[i], self.zbins[i+1])

                self.jlumcounts  = self.jackknife(self.lumcounts, reduce_jk=False)
                self.jluminosity_function = self.jlumcounts / vol.reshape(self.njacktot, 1, 1, -1) / dl

                self.luminosity_function = np.sum(self.jluminosity_function, axis=0) / self.njacktot
                self.varluminosity_function = np.sum((self.jluminosity_function - self.luminosity_function) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot
                self.y = self.luminosity_function
                self.ye = np.sqrt(self.varluminosity_function)
        else:
            if self.jtype is not None:
                area = self.ministry.galaxycatalog.getArea(jackknife=True)
            else:
                area = self.ministry.galaxycatalog.getArea(jackknife=False)

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
                 catalog_type=['galaxycatalog'], tag=None, cumulative=False, 
                 mcut=None, mcutind=None, **kwargs):

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
        self.mcut = mcut
        self.mcutind = mcutind

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
                    if self.mcut is not None:
                        if self.mcutind is None:
                            lidx = mapunit['appmag'][zlidx:zhidx] < self.mcut
                        else:
                            lidx = mapunit['appmag'][zlidx:zhidx,self.mcutind] < self.mcut
                    else:
                        lidx = np.ones(zhidx-zlidx, dtype=np.bool)

                    c, e = np.histogram(mapunit['appmag'][zlidx:zhidx,j][lidx],
                                        bins=self.magbins)
                    self.mc[self.jcount,:,j,i] += c
        else:
            for j in range(self.nbands):
                if self.mcut is not None:
                    if self.mcutind is None:
                        lidx = mapunit['appmag']<self.mcut
                    else:
                        lidx = mapunit['appmag'][:,self.mcutind] < self.mcut
                else:
                        lidx = np.ones(len(mapunit['appmag']), dtype=np.bool)

                c, e = np.histogram(mapunit['appmag'][lidx,j], bins=self.magbins)
                self.mc[self.jcount,:,j,0] += c

    def reduce(self, rank=None, comm=None):

        if rank is not None:
            gdata = comm.gather(self.mc, root=0)

            if rank==0:
                gshape = [self.mc.shape[i] for i in range(len(self.mc.shape))]
                gshape[0] = self.njacktot

                self.mc = np.zeros(gshape)
                jc = 0
                for g in gdata:
                    if g is None: continue
                    nj = g.shape[0]
                    self.mc[jc:jc+nj,:,:,:] = g

                    jc += nj

                if self.jtype is not None:
                    area = self.ministry.galaxycatalog.getArea(jackknife=True)
                else:
                    area = self.ministry.galaxycatalog.getArea(jackknife=False)

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
            if self.jtype is not None:
                area = self.ministry.galaxycatalog.getArea(jackknife=True)
            else:
                area = self.ministry.galaxycatalog.getArea(jackknife=False)                

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
                 catalog_type=['galaxycatalog'], tag=None, lightcone=True,
                 **kwargs):
        Metric.__init__(self, ministry, catalog_type=catalog_type, tag=tag, **kwargs)

        self.lightcone = lightcone

        if self.lightcone:
            if zbins is None:
                self.zbins = [self.ministry.minz, self.ministry.maxz]
            else:
                self.zbins = zbins
                self.zbins = np.array(self.zbins)

            self.nzbins = len(self.zbins)-1

        else:
            self.zbins = None
            self.nzbins = 1

        if massbins is None:
            self.massbins = np.logspace(12, 15, 20)
        else:
            self.massbins = massbins

        self.mapkeys = ['luminosity', 'central', 'halomass']
        self.aschema = 'galaxyonly'
        self.unitmap = {'luminosity':'mag', 'halomass':'msunh'}

        if self.lightcone:
            self.mapkeys.append('redshift')
            self.unitmap['redshift'] = 'z'

        self.totlum   = None
        self.bincount = None

    @jackknifeMap
    def map(self, mapunit):

        if len(mapunit['luminosity'].shape) < 2:
            self.nbands = 1
        else:
            self.nbands = mapunit['luminosity'].shape[1]

        mu = {}
        mc = mapunit['central']==1
        mc = mc.reshape(len(mapunit['central']))

        for k in mapunit.keys():
            if (k=='luminosity') & (len(mapunit[k].shape) < 2):
                mu[k] = mapunit[k][mc].reshape(-1,1)
            else:
                mu[k] = mapunit[k][mc]

        del mc

        if self.totlum is None:
            self.totlum = np.zeros((self.njack, len(self.massbins)-1,
                                      self.nbands, self.nzbins))
            self.bincount = np.zeros((self.njack, len(self.massbins)-1,
                                        self.nbands, self.nzbins))

        if self.lightcone:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mu['redshift'].searchsorted(self.zbins[i])
                zhidx = mu['redshift'].searchsorted(self.zbins[i+1])
                mb = np.digitize(mu['halomass'][zlidx:zhidx], bins=self.massbins)

                for j in xrange(1, len(self.massbins)):
                    blum = mu['luminosity'][zlidx:zhidx,:][mb==j]
                    self.bincount[self.jcount,j-1,:,i] += len(blum)
                    self.totlum[self.jcount,j-1,:,i] += np.sum(blum, axis=0)
        else:
            mb = np.digitize(mu['halomass'], bins=self.massbins)

            for j in range(1,len(self.massbins)):
                blum = mu['luminosity'][mb==j]
                self.bincount[self.jcount,j-1,:,0] += len(blum)
                self.totlum[self.jcount,j-1,:,0] += np.sum(blum, axis=0)

        del mu


    def reduce(self, rank=None, comm=None):

        if rank is not None:
            gtotlum = comm.gather(self.totlum, root=0)
            gbincount = comm.gather(self.bincount, root=0)

            if rank==0:
                tshape = [self.totlum.shape[i] for i in range(len(self.totlum.shape))]
                bshape = [self.bincount.shape[i] for i in range(len(self.bincount.shape))]

                tshape[0] = self.njacktot
                bshape[0] = self.njacktot

                self.bincount = np.zeros(tshape)
                self.totlum = np.zeros(bshape)

                jc = 0
                for i, g in enumerate(gtotlum):
                    if g is None: continue
                    nj = g.shape[0]
                    self.totlum[jc:jc+nj,:,:,:] = g
                    self.bincount[jc:jc+nj,:,:,:] = gbincount[i]

                    jc += nj

                self.jtotlum   = self.jackknife(self.totlum, reduce_jk=False)
                self.jbincount = self.jackknife(self.bincount, reduce_jk=False)
                self.jlcen_mass = self.jtotlum / self.jbincount
                self.lcen_mass = (np.sum(self.jlcen_mass, axis=0) /
                                    self.njacktot)
                self.varlcen_mass = (np.sum((self.jlcen_mass - self.lcen_mass)**2,
                                            axis=0) * (self.njacktot - 1) /
                                            self.njacktot)

        else:

            self.jtotlum    = self.jackknife(self.totlum, reduce_jk=False)
            self.jbincount  = self.jackknife(self.bincount, reduce_jk=False)
            self.jlcen_mass = self.jtotlum / self.jbincount
            self.lcen_mass  = np.sum(self.jlcen_mass, axis=0) / self.njacktot
            self.varlcen_mass = (np.sum((self.jlcen_mass - self.lcen_mass)**2,
                                          axis=0) * (self.njacktot - 1) /
                                          self.njacktot)

    def visualize(self, compare=False, plotname=None, f=None, ax=None,
                  usebands=None, usez=None, **kwargs):

        if hasattr(self, 'massmean'):
            mmass = self.massmean
        else:
            mmass = np.array([(self.massbins[i]+self.massbins[i+1])/2
                              for i in range(len(self.massbins)-1)])

        if usebands is None:
            usebands = range(self.nbands)

        if usez is None:
            usez = range(self.nzbins)


        if f is None:
            f, ax = plt.subplots(len(usebands), len(usez),
                                 sharex=True, sharey=True,
                                 figsize=(8,8))
            ax = np.array(ax).reshape(len(usebands), len(usez))
            newaxes = True
        else:
            newaxes = False
            ax = np.array(ax).reshape(len(usebands), len(usez))

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
            sax.set_xlabel(r'$M_{halo}\, [M_{sun} h^{-1}]$', fontsize=16, labelpad=40)
            sax.set_ylabel(r'$L_{cen}\, [mag]$', fontsize=16, labelpad=40)

        for i, b in enumerate(usebands):
            for j, z in enumerate(usez):
                ye = np.sqrt(self.varlcen_mass[:,b,z])

                l = ax[i][j].plot(mmass, self.lcen_mass[:,b,z],
                                **kwargs)
                ax[i][j].fill_between(mmass, self.lcen_mass[:,b,z] - ye,
                                self.lcen_mass[:,b,z] + ye)
                ax[i][j].set_xscale('log')

        #plt.tight_layout()

        if (plotname is not None) and (not compare):
            plt.savefig(plotname)

        return f, ax, l


    def compare(self, othermetrics, plotname=None, usebands=None, 
                 usez=None, labels=None, **kwargs):
        tocompare = [self]
        tocompare.extend(othermetrics)

        if usebands is not None:
            if not hasattr(usebands[0], '__iter__'):
                usebands = [usebands]*len(tocompare)
            else:
                assert(len(usebands)==len(tocompare))
        else:
            usebands = [None]*len(tocompare)

        if usez is not None:
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
            if usebands[i] is not None:
                assert(len(usebands[0])==len(usebands[i]))
            if usez[i] is not None:
                assert(len(usez[0])==len(usez[i]))

            if i==0:
                f, ax, l = m.visualize(usebands=usebands[i], compare=True,
                                    color=Metric._color_list[i], usez=usez[i],
                                    label=labels[i],**kwargs)
            else:
                f, ax, l = m.visualize(usebands=usebands[i], compare=True,
                                    f=f, ax=ax, color=Metric._color_list[i],
                                    usez=usez[i],label=labels[i],
                                    **kwargs)
            lines.append(l[0])

        if labels[0] is not None:
            f.legend(lines, labels, 'best')

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax

class ColorDist(Metric):

    def __init__(self, ministry, zbins=None, cbins=None,
                    catalog_type=['galaxycatalog'],
                    usebands=None, appmag=False, magcuts=None,
                    pdf=False, cutind=None, binnedz=True,
                    appmagcut=False, binnedmag=False, **kwargs):


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

        Metric.__init__(self, ministry,
                          catalog_type=catalog_type,
                          **kwargs)
        
        self.zbins = zbins
        self.binnedz = binnedz
        if self.binnedz:
            self.nzbins = len(zbins)-1
        else:
            self.nzbins = len(zbins)

        if appmag:
            self.mkey = 'appmag'
        else:
            self.mkey = 'luminosity'

        if appmagcut:
            self.mcutkey = 'appmag'
        else:
            self.mcutkey = 'luminosity'

        if magcuts is None:
            if appmag:
                self.magcuts = np.array([23])
            else:
                self.magcuts = np.array([-19])
        else:
            self.magcuts = magcuts

        self.binnedmag = binnedmag

        if not self.binnedmag:
            self.nmagcuts = len(self.magcuts)
        else:
            self.nmagcuts = len(self.magcuts)-1

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

        if self.mcutkey not in self.mapkeys:
            self.mapkeys.append(self.mcutkey)

        self.aschema = 'galaxyonly'
        self.unitmap = {self.mkey:'mag'}
        
        if self.mcutkey not in self.unitmap.keys():
            self.unitmap[self.mcutkey] = 'mag'

    @jackknifeMap
    def map(self, mapunit):

        if self.cd is None:
            self.cd = np.zeros((self.njack, self.ncbins,
                                self.ncolors, self.nmagcuts,
                                self.nzbins))

        clr = np.zeros((len(mapunit[self.mkey]),
                        self.ncolors))

        for c in range(self.ncolors):
            clr[:,c] = mapunit[self.mkey][:,self.usebands[c][0]] - mapunit[self.mkey][:,self.usebands[c][1]]

        for i, z in enumerate(self.zbins):
            if self.binnedz & (i==self.nzbins): continue

            if self.binnedz:
                zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])
            else:
                zlidx = 0
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i])



            for j in range(self.nmagcuts):
                if not self.binnedmag:
                    lidx = mapunit[self.mcutkey][zlidx:zhidx,self.cutind] < self.magcuts[j]
                else:
                    lidx = ((self.magcuts[j]<mapunit[self.mcutkey][zlidx:zhidx,self.cutind])
                           & (mapunit[self.mcutkey][zlidx:zhidx,self.cutind] < self.magcuts[j+1]))
                for ci in range(self.ncolors):
                    c, self.cbins = np.histogram(clr[zlidx:zhidx,ci][lidx], bins=self.cbins)
                    self.cd[self.jcount,:,ci,j,i] += c


    def reduce(self, rank=None, comm=None):

        if rank is not None:
            gcd = comm.gather(self.cd, root=0)

            if rank==0:
                gshape = [self.cd.shape[i] for i in range(len(self.cd.shape))]
                gshape[0] = self.njacktot

                self.cd = np.zeros(gshape)
                jc = 0
                for i, g in enumerate(gcd):
                    if g is None: continue
                    nj = g.shape[0]
                    self.cd[jc:jc+nj,:,:,:] = g

                    jc += nj
                if self.pdf:
                    self.tc = np.sum(self.cd, axis=1).reshape(self.njacktot, 1, self.ncolors, self.nmagcuts, self.nzbins)
                    self.jcd = self.jackknife(self.cd, reduce_jk=False)
                    self.jtc = self.jackknife(self.tc, reduce_jk=False)
                    dc = self.cbins[1:] - self.cbins[:-1]
                    self.jcolor_dist = self.jcd / self.jtc / dc.reshape(1,self.ncbins,1,1,1)
                else:
                    if self.jtype is not None:
                        area = self.ministry.galaxycatalog.getArea(jackknife=True)
                    else:
                        area = self.ministry.galaxycatalog.getArea()

                    self.jcd = self.jackknife(self.cd, reduce_jk=False)
                    dc = self.cbins[1:] - self.cbins[:-1]
                    self.jcolor_dist = self.jcd / area.reshape(self.njacktot, 1, 1, 1, 1) / dc.reshape(1,self.ncbins,1,1,1)

                self.color_dist = np.sum(self.jcolor_dist, axis=0) / self.njacktot
                self.varcolor_dist = np.sum((self.jcolor_dist - self.color_dist) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot

                self.y = self.color_dist
                self.ye = np.sqrt(self.varcolor_dist)
        else:
            if self.pdf:
                self.tc = np.sum(self.cd, axis=1).reshape(self.njacktot, 1, self.ncolors, self.nmagcuts, self.nzbins)
                self.jcd = self.jackknife(self.cd, reduce_jk=False)
                self.jtc = self.jackknife(self.tc, reduce_jk=False)
                dc = self.cbins[1:] - self.cbins[:-1]
                self.jcolor_dist = self.jcd / self.jtc / dc.reshape(1, self.ncbins, 1, 1,1)

            else:
                if self.jtype is not None:
                    area = self.ministry.galaxycatalog.getArea(jackknife=True)
                else:
                    area = self.ministry.galaxycatalog.getArea()

                self.jcd = self.jackknife(self.cd, reduce_jk=False)
                dc = self.cbins[1:] - self.cbins[:-1]
                self.jcolor_dist = self.jcd / area.reshape(self.njacktot,1,1,1,1) / dc.reshape(1,self.ncbins,1,1,1)

            self.color_dist = np.sum(self.jcolor_dist, axis=0) / self.njacktot
            self.varcolor_dist = np.sum((self.jcolor_dist - self.color_dist) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot

            self.y = self.color_dist
            self.ye = np.sqrt(self.varcolor_dist)


    def visualize(self, compare=False, plotname=None, f=None, ax=None,
                  usecolors=None, usezm=None,
                  colors=None, xlabel=None,
                  ylabel=None, **kwargs):

        if hasattr(self, 'magmean'):
            mclr = self.mclr
        else:
            mclr = np.array([(self.cbins[i]+self.cbins[i+1])/2
                              for i in range(len(self.cbins)-1)])

        if usecolors is None:
            usecolors = range(self.color_dist.shape[1])

        if usezm is None:
            usezm = range(self.color_dist.shape[3] * self.color_dist.shape[2])

        if f is None:
            f, ax = plt.subplots(len(usezm), len(usecolors),
                                 sharex=True, sharey=True, figsize=(8,8))
            ax = np.array(ax)
            ax = ax.reshape(len(usezm), len(usecolors))
            newaxes = True
        else:
            newaxes = False

        for i, c in enumerate(usecolors):
            for j, z in enumerate(usezm):
                zi = z / self.nmagcuts
                mi = z % self.nmagcuts
                l1 = ax[j][i].plot(mclr, self.color_dist[:,c,mi,zi],
                                  c=colors, **kwargs)


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
            if xlabel is None:
                sax.set_xlabel(r'$Color$', labelpad=40)
            else:
                sax.set_xlabel(xlabel, labelpad=40)
            if ylabel is None:
                sax.set_ylabel(r'$p(color)$', labelpad=40)
            else:
                sax.set_ylabel(ylabel, labelpad=40)

        #plt.tight_layout()

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)


        return f, ax, l1

    def compare(self, othermetrics, plotname=None, usecolors=None,
                  usezm=None, labels=None, colors=None, 
                  **kwargs):

        tocompare = [self]
        tocompare.extend(othermetrics)

        if usecolors!=None:
            if not hasattr(usecolors[0], '__iter__'):
                usecolors = [usecolors]*len(tocompare)
            else:
                assert(len(usecolors)==len(tocompare))
        else:
            usecolors = [None]*len(tocompare)

        if usezm!=None:
            if not hasattr(usezm[0], '__iter__'):
                usezm = [usezm]*len(tocompare)
            else:
                assert(len(usezm)==len(tocompare))
        else:
            usezm = [None]*len(tocompare)

        if colors is None:
            colors = [None]*len(tocompare)
        else:
            assert(len(colors)==len(tocompare))

        if labels is None:
            labels = [None] * len(tocompare)



        lines = []

        for i, m in enumerate(tocompare):
            if i==0:
                f, ax, l = m.visualize(usecolors=usecolors[i], usezm=usezm[i],
                                       compare=True,
                                       colors=colors[i], **kwargs)
            else:
                f, ax, l = m.visualize(usecolors=usecolors[i], compare=True,
                                        usezm=usezm[i], 
                                        ax=ax, f=f, colors=colors[i],**kwargs)

            lines.append(l)


        if labels[0]!=None:
            f.legend(lines, labels, 'best')

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax



class ColorColor(Metric):
    """
    Color-color diagram.
    """
    def __init__(self, ministry, zbins=[0.0, 0.2], cbins=None,
                 catalog_type=['galaxycatalog'],
                 usebands=None, magcuts=None,
                 tag=None, appmag=False,
                 appmagcut=False,
                 pdf=False, cutind=None, 
                 binnedz=True, binnedmag=False,
                 **kwargs):

        Metric.__init__(self, ministry, catalog_type=catalog_type, tag=tag, **kwargs)

        self.binnedz = binnedz
        self.zbins = zbins
        if zbins is None:
            self.nzbins = 1
        else:
            if self.binnedz:
                self.nzbins = len(zbins)-1
            else:
                self.nzbins = len(zbins)

            self.zbins = np.array(self.zbins)

        if cbins is None:
            self.cbins = np.linspace(-1, 2, 30)
        else:
            self.cbins = cbins

        if appmag:
            self.mkey = 'appmag'
        else:
            self.mkey = 'luminosity'

        if appmagcut:
            self.mcutkey = 'appmag'
        else:
            self.mcutkey = 'luminosity'

        if (zbins is not None):
            self.mapkeys = [self.mkey, 'redshift']
        else:
            self.mapkeys = [self.mkey]

        if self.mcutkey not in self.mapkeys:
            self.mapkeys.append(self.mcutkey)

        if magcuts is None:
            if appmag:
                self.magcuts = np.array([23])
            else:
                self.magcuts = np.array([-19])
        else:
            self.magcuts = magcuts

        self.cutind = cutind
        self.binnedmag = binnedmag
        
        if not self.binnedmag:
            self.nmagcuts = len(self.magcuts)
        else:
            self.nmagcuts = len(self.magcuts)-1

        self.pdf = pdf
        self.usebands = usebands
        self.aschema = 'galaxyonly'
        self.unitmap = {self.mkey:'mag'}

        if self.mcutkey not in self.unitmap.keys():
            self.unitmap[self.mcutkey] = 'mag'

        self.cc = None

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

        if self.cc is None:
            self.cc = np.zeros((self.njack,len(self.cbins)-1, len(self.cbins)-1,
                                self.nbands-2, self.nmagcuts, self.nzbins))

        if self.zbins is not None:
            for i, z in enumerate(self.zbins):
                if self.binnedz & (i==self.nzbins):continue
                
                if self.binnedz:
                    zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                    zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])
                else:
                    zlidx = 0
                    zhidx = mapunit['redshift'].searchsorted(self.zbins[i])

                for j in range(self.nmagcuts):
                    if not self.binnedmag:
                        if self.cutind is None:
                            lidx = mapunit[self.mcutkey][zlidx:zhidx]<=self.magcuts[j]
                        else:
                            lidx = mapunit[self.mcutkey][zlidx:zhidx,self.cutind]<=self.magcuts[j]
                    else:
                        if self.cutind is None:
                            lidx = ((self.magcuts[j]<mapunit[self.mcutkey][zlidx:zhidx])
                                    &(mapunit[self.mcutkey][zlidx:zhidx]<=self.magcuts[j+1]))
                        else:
                            lidx = ((self.magcuts[j]<mapunit[self.mcutkey][zlidx:zhidx,self.cutind])
                                    &(mapunit[self.mcutkey][zlidx:zhidx,self.cutind]<=self.magcuts[j+1]))

                        
                        for k in range(self.nclr-1):
                            c, e0, e1 = np.histogram2d(clr[zlidx:zhidx,k+1][lidx],
                                                       clr[zlidx:zhidx,k][lidx],
                                                       bins=self.cbins)
                            self.cc[self.jcount,:,:,k,j,i] += c
        else:
            for j, m in enumerate(self.magcuts):
                if not self.binnedmag:
                    if self.cutind is None:
                        lidx = mapunit[self.mcutkey][:]<=self.magcuts[j]
                    else:
                        lidx = mapunit[self.mcutkey][:,self.cutind]<=self.magcuts[j]
                else:
                    if self.cutind is None:
                        lidx = ((self.magcuts[j]<mapunit[self.mcutkey][:])
                                &(mapunit[self.mcutkey][:]<=self.magcuts[j+1]))
                    else:
                        lidx = ((self.magcuts[j]<mapunit[self.mcutkey][:,self.cutind])
                                &(mapunit[self.mcutkey][:,self.cutind]<=self.magcuts[j+1]))

                    for k in range(self.nclr-1):
                        c, e0, e1 = np.histogram2d(clr[:,k+1][lidx],
                                                   clr[:,k][lidx],
                                                   bins=self.cbins)
                        self.cc[self.jcount,:,:,k,j,0] += c


    def reduce(self, rank=None, comm=None):
        if rank is not None:
            gcc = comm.gather(self.cc, root=0)

            if rank==0:
                gshape = [self.cc.shape[i] for i in range(len(self.cc.shape))]
                gshape[0] = self.njacktot

                self.cc = np.zeros(gshape)
                jc = 0
                for i, g in enumerate(gcc):
                    if g is None: continue
                    nj = g.shape[0]
                    self.cc[jc:jc+nj,:,:,:] = g

                    jc += nj

                if self.pdf:
                    self.tc = np.sum(np.sum(self.cc, axis=1), axis=1).reshape(-1,1,1,self.nbands-2,self.nmagcuts,self.nzbins)
                    self.jcc = self.jackknife(self.cc, reduce_jk=False)
                    self.jtc = self.jackknife(self.tc, reduce_jk=False)
                    dc = self.cbins[1:] - self.cbins[:-1]
                    dc = np.outer(dc, dc)
                    self.jcolor_color = self.jcc / self.jtc / dc.reshape(-1,self.ncbins,self.ncbins, 1, 1, 1)
                else:
                    if self.jtype is not None:
                        area = self.ministry.galaxycatalog.getArea(jackknife=True)
                    else:
                        area = self.ministry.galaxycatalog.getArea()
                    self.jcc = self.jackknife(self.cc, reduce_jk=False)
                    self.jcolor_color = self.jcc / area.reshape(self.njacktot,1,1,1,1,1)

                self.color_color = np.sum(self.jcolor_color, axis=0) / self.njacktot
                self.varcolor_color = np.sum((self.jcolor_color - self.color_color) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot
        else:
            if self.pdf:
                self.tc = np.sum(np.sum(self.cc, axis=1), axis=1).reshape(-1,1,1,self.nbands-2,self.nmagcuts,self.nzbins)
                self.jcc = self.jackknife(self.cc, reduce_jk=False)
                self.jtc = self.jackknife(self.tc, reduce_jk=False)
                dc = self.cbins[1:] - self.cbins[:-1]
                dc = np.outer(dc, dc)
                self.jcolor_color = self.jcc / self.jtc / dc.reshape(1,self.ncbins,self.ncbins,1,1,1)
            else:
                if self.jtype is not None:
                    area = self.ministry.galaxycatalog.getArea(jackknife=True)
                else:
                    area = self.ministry.galaxycatalog.getArea()
                self.jcc = self.jackknife(self.cc, reduce_jk=False)
                self.jcolor_color = self.jcc / area.reshape(self.njacktot,1,1,1,1,1)

            self.color_color = np.sum(self.jcolor_color, axis=0) / self.njacktot
            self.varcolor_color = np.sum((self.jcolor_color - self.color_color) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot


    def visualize(self, compare=False, plotname=None, f=None, ax=None,
                  usecolors=None, usezm=None, 
                  colors=None, xlabel=None,
                  ylabel=None, nc=5, **kwargs):

        if hasattr(self, 'magmean'):
            mclr = self.mclr
        else:
            mclr = np.array([(self.cbins[i]+self.cbins[i+1])/2
                              for i in range(len(self.cbins)-1)])

        if usecolors is None:
            usecolors = range(self.color_color.shape[2])

        if usezm is None:
            usezm = range(self.color_color.shape[4] * self.color_color.shape[3])

        if f is None:
            f, ax = plt.subplots(len(usezm), len(usecolors),
                                 sharex=True, sharey=True, figsize=(8,8))
            ax = np.array(ax)
            ax = ax.reshape(len(usezm), len(usecolors))
            newaxes = True
        else:
            newaxes = False

        X, Y = np.meshgrid(mclr, mclr)

        for i, c in enumerate(usecolors):
            for j, z in enumerate(usezm):
                zi = z / self.nmagcuts
                mi = z % self.nmagcuts
                try:
                    l1 = ax[j][i].contour(X, Y, self.color_color[:,:,c,mi,zi].T, nc,
                                     colors=colors, **kwargs)
                    l1 = plt.Rectangle((0,0),1,1,fc = l1.collections[0].get_color()[0]) 
                except:
                    l1 = plt.Rectangle((0,0),1,1,fc = 'k')
                
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
            if xlabel is None:
                sax.set_xlabel(r'$Color$', labelpad=40)
            else:
                sax.set_xlabel(xlabel, labelpad=40)
            if ylabel is None:
                sax.set_ylabel(r'$Color$', labelpad=40)
            else:
                sax.set_ylabel(ylabel, labelpad=40)

        #plt.tight_layout()

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)


        return f, ax, l1

    def compare(self, othermetrics, plotname=None, usecolors=None,
                  usezm=None, labels=None, colors=None,
                  **kwargs):

        tocompare = [self]
        tocompare.extend(othermetrics)

        if usecolors!=None:
            if not hasattr(usecolors[0], '__iter__'):
                usecolors = [usecolors]*len(tocompare)
            else:
                assert(len(usecolors)==len(tocompare))
        else:
            usecolors = [None]*len(tocompare)

        if usezm!=None:
            if not hasattr(usezm[0], '__iter__'):
                usezm = [usezm]*len(tocompare)
            else:
                assert(len(usezm)==len(tocompare))
        else:
            usezm = [None]*len(tocompare)

        if colors is None:
            colors = [None]*len(tocompare)
        else:
            assert(len(colors)==len(tocompare))

        if labels is None:
            labels = [None] * len(tocompare)

        lines = []

        for i, m in enumerate(tocompare):
            if i==0:
                f, ax, l = m.visualize(usecolors=usecolors[i], compare=True,
                                    colors=colors[i], usezm=usezm[i],**kwargs)
            else:
                f, ax, l = m.visualize(usecolors=usecolors[i], compare=True,
                                        ax=ax, f=f, colors=colors[i], usezm=usezm[i],
                                        **kwargs)

            lines.append(l)


        if labels[0]!=None:
            f.legend(lines, labels, 'best')

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
                  appcolor=False, pdf=False, 
                  magcut=None, magcutind=None, **kwargs):

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

        if appcolor:
            self.ckey = 'appmag'
        else:
            self.ckey = 'luminosity'

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
        self.magcut = magcut
        self.magcutind = magcutind

        if central_only & (zbins is not None):
            self.mapkeys = [self.mkey, 'redshift', 'central']
        elif (zbins is not None):
            self.mapkeys = [self.mkey, 'redshift']
        else:
            self.mapkeys = [self.mkey]

        if self.ckey not in self.mapkeys:
            self.mapkeys.append(self.ckey)

        self.aschema = 'galaxyonly'
        self.unitmap = {self.mkey:'mag'}
        
        if self.ckey != self.mkey:
            self.unitmap[self.ckey] = 'mag'

        self.cc = None

    @jackknifeMap
    def map(self, mapunit):

        if self.usebands is None:
            self.nbands = mapunit[self.mkey].shape[1]
            self.usebands = range(self.nbands)

        mu = {}
        if self.central_only:
            delete_after_map = True
            for k in mapunit.keys():
                mu[k] = mapunit[k][mapunit['central']==1]
        else:
            delete_after_map = False
            mu = mapunit

        if self.magcut is not None:
            if self.magcutind is not None:
                idx = mu[self.mkey][:,self.magcutind]<self.magcut
            else:
                idx = mu[self.mkey]<self.magcut

            for k in mapunit.keys():
                mu[k] = mapunit[k][idx]
        else:
                mu = mu

        if self.cc is None:
            self.cc = np.zeros((self.njack, len(self.magbins)-1,
                                len(self.cbins)-1,
                                int(self.nbands-1),
                                self.nzbins))

        if self.zbins is not None:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mu['redshift'].searchsorted(self.zbins[i])
                zhidx = mu['redshift'].searchsorted(self.zbins[i+1])
                for j, b1 in enumerate(self.usebands[:-1]):
                    c, e0, e1 = np.histogram2d(mu[self.mkey][zlidx:zhidx,self.usebands[j+1]],
                                               mu[self.ckey][zlidx:zhidx,b1] -
                                               mu[self.ckey][zlidx:zhidx,self.usebands[j+1]],
                                               bins=[self.magbins,self.cbins])
                    self.cc[self.jcount,:,:,j,i] += c
        else:
            for j, b1 in enumerate(self.usebands):
                c, e0, e1 = np.histogram2d(mu[self.mkey][:,self.usebands[j+1]],
                                           mu[self.ckey][:,b1] -
                                           mu[self.ckey][:,self.usebands[j+1]],
                                           bins=[self.magbins,self.cbins])
                self.cc[self.jcount,:,:,j,0] += c

        if delete_after_map:
            del mu

    def reduce(self, rank=None, comm=None):
        if rank is not None:
            gcc = comm.gather(self.cc, root=0)

            if rank==0:
                jc = 0
                dshape = self.cc.shape
                dshape = [dshape[i] for i in range(len(dshape))]
                dshape[0] = self.njacktot

                for i, g in enumerate(gcc):
                    if g is None: continue
                    self.cc = np.zeros(dshape)
                    nj = g.shape[0]
                    self.cc[jc:jc+nj,:,:,:,:] = g

                    jc += nj

                if self.pdf:
                    self.tc = np.sum(np.sum(self.cc, axis=1), axis=1).reshape(-1,1,1,(self.nbands-1),self.nzbins)
                    self.jcc = self.jackknife(self.cc, reduce_jk=False)
                    self.jtc = self.jackknife(self.tc, reduce_jk=False)
                    dc = self.cbins[1:] - self.cbins[:-1]
                    dm = self.magbins[1:] - self.magbins[:-1]
                    dcdm = np.outer(dc,dm).reshape(-1,self.ncbins, self.nmagbins,1,1)
                    self.jcolor_mag = self.jcc / self.tc / dcdm
                else:
                    if self.jtype is not None:
                        area = self.ministry.galaxycatalog.getArea(jackknife=True)
                    else:
                        area = self.ministry.galaxycatalog.getArea(jackknife=False)
                    self.jcc = self.jackknife(self.cc, reduce_jk=False)
                    self.jcolor_mag = self.jcc / area.reshape(self.njacktot,1,1,1,1)

                self.color_mag = np.sum(self.jcolor_mag, axis=0) / self.njacktot
                self.varcolor_mag = np.sum((self.jcolor_mag - self.color_mag) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot

        else:
            if self.pdf:
                self.tc = np.sum(np.sum(self.cc, axis=1), axis=1).reshape(-1,1,1,self.nbands-1,self.nzbins)
                self.jcc = self.jackknife(self.cc, reduce_jk=False)
                self.jtc = self.jackknife(self.tc, reduce_jk=False)
                dc = self.cbins[1:] - self.cbins[:-1]
                dm = self.magbins[1:] - self.magbins[:-1]
                dcdm = np.outer(dc,dm).reshape(-1,self.ncbins, self.nmagbins,1,1)
                self.jcolor_mag = self.jcc / self.tc / dcdm
            else:
                if self.jtype is not None:
                    area = self.ministry.galaxycatalog.getArea(jackknife=True)
                else:
                    area = self.ministry.galaxycatalog.getArea(jackknife=False)
                self.jcc = self.jackknife(self.cc, reduce_jk=False)
                self.jcolor_mag = self.jcc / area.reshape(self.njacktot,1,1,1,1)

            self.color_mag = np.sum(self.jcolor_mag, axis=0) / self.njacktot
            self.varcolor_mag = np.sum((self.jcolor_mag - self.color_mag) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot


    def visualize(self, plotname=None, f=None, ax=None, usecolors=None,
                  compare=False, nc=3, **kwargs):

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
                l1 = ax[j][i].contour(X, Y, cc[:,:,c,j].T,nc,
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

    def __init__(self, ministry, zbins=None,
                  onezbin=False,
                  catalog_type=['galaxycatalog'],
                  tag=None, appmag=True, 
                  cbins=None, splitcolor=None,
                  cinds=None, magcuts=None,**kwargs):

        Metric.__init__(self, ministry, catalog_type=catalog_type,tag=tag,**kwargs)
        self.zbins = zbins

        if (zbins is None) & onezbin:
            self.nzbins = 1
        elif zbins is None:
            self.zbins = np.linspace(ministry.minz,
                                     ministry.maxz,
                                     50)
            self.nzbins = len(zbins)-1
        else:
            self.nzbins = len(zbins)-1
            self.zbins = np.array(self.zbins)

        if magcuts is None:
            self.magcuts = np.zeros(self.nzbins) -18
        else:
            if hasattr(magcuts, '__iter__'):
                self.magcuts = magcuts
            else:
                self.magcuts = np.zeros(self.nzbins)  + self.magcuts
            

        if splitcolor is None:
            self.splitcolor = np.zeros(self.nzbins)
        else:
            self.splitcolor = np.zeros(self.nzbins) + splitcolor

        if cinds is None:
            self.cinds = [0,1]
        else:
            self.cinds = cinds

        self.appmag = appmag

        if appmag:
            self.mkey = 'appmag'
        else:
            self.mkey = 'luminosity'

        if cbins is None:
            self.cbins = 100
        else:
            self.cbins = cbins


        self.mapkeys = [self.mkey, 'redshift']
        self.unitmap = {self.mkey:'mag'}
        self.aschema = 'galaxyonly'

        self.qscounts = None
        self.tcounts  = None


    @jackknifeMap
    def map(self, mapunit):

        if self.qscounts is None:
            self.qscounts = np.zeros((self.njack, self.nzbins))
            self.tcounts = np.zeros((self.njack,self.nzbins))

        clr = mapunit[self.mkey][:,self.cinds[0]] - mapunit[self.mkey][:,self.cinds[1]]

        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

            if self.appmag:
                ccounts, cbins = np.histogram(clr[zlidx:zhidx], self.cbins)
                self.splitcolor[i] = self.splitBimodal(cbins[:-1], ccounts)
            elif self.splitcolor is None:
                czhidx = mapunit['redshift'].searchsorted(0.2)
                ccounts, cbins = np.histogram(clr[:czhidx], self.cbins)
                self.splitcolor[i] = self.splitBimodal(cbins[:-1], ccounts)
 
            if self.splitcolor[i] is None:
                continue
            
            lidx = mapunit[self.mkey][zlidx:zhidx][:,self.cinds[0]] < self.magcuts[i]
            qidx, = np.where(clr[zlidx:zhidx][lidx]>self.splitcolor[i])

            self.qscounts[self.jcount,i] = len(qidx)
            self.tcounts[self.jcount,i] = np.sum(lidx)


    def reduce(self, rank=None, comm=None):
        if rank is not None:
            gqs = comm.gather(self.qscounts, root=0)
            gtc = comm.gather(self.tcounts, root=0)


            if rank==0:
                qcshape = [self.qscounts.shape[i] for i in range(len(self.qscounts.shape))]
                tcshape = [self.tcounts.shape[i] for i in range(len(self.tcounts.shape))]

                qcshape[0] = self.njacktot
                tcshape[0] = self.njacktot

                self.qscounts = np.zeros(qcshape)
                self.tcounts = np.zeros(tcshape)

                jc = 0
                for i, g in enumerate(gqs):
                    if g is None: continue
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


    def visualize(self, f=None, ax=None, compare=False, plotname=None,
                  **kwargs):

        if f is None:
            f, ax = plt.subplots(1, figsize=(8,8))
            newaxes = True
        else:
            newaxes = False

        zm = (self.zbins[:-1] + self.zbins[1:])/2
        ye = np.sqrt(self.varfquenched)
        ax.plot(zm, self.fquenched, **kwargs)
        ax.fill_between(zm, self.fquenched - ye, self.fquenched + ye, **kwargs)

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
            sax.set_xlabel(r'$z$', fontsize=16, labelpad=20)
            sax.set_ylabel(r'$f_{red}$', fontsize=16, labelpad=20)

        #plt.tight_layout()

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax

    def compare(self, othermetrics, plotname=None, **kwargs):
        tocompare = [self]
        tocompare.extend(othermetrics)

        for i, m in enumerate(tocompare):
            if i==0:
                f, ax = m.visualize(compare=True,
                                    color=self._color_list[i],
                                    **kwargs)
            else:
                f, ax = m.visualize(f=f, ax=ax,
                                    color=self._color_list[i],
                                    compare=True,
                                    **kwargs)

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax

class FRed(Metric):

    def __init__(self, ministry, zbins=None,
                  catalog_type=['galaxycatalog'],
                  zeroind=True, tag=None,
                  ctfile=None, **kwargs):
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

        if ctfile is None:
            self.ctfile = '/nfs/slac/g/ki/ki23/des/jderose/l-addgals/training/cooper/dr6_cooper_id_with_red.dat'
        else:
            self.ctfile = ctfile

        self.ctcat = None

        self.qscounts = None
        self.tcounts  = None

    @jackknifeMap
    def map(self, mapunit):

        #if haven't loaded training file, do so
        if self.ctcat is None:
            self.ctcat = np.genfromtxt(self.ctfile)

        mu = {}
        if self.zeroind:
            idx = (mapunit['ctcatid']-1)<len(self.ctcat)
        else:
            idx = (mapunit['ctcatid'])<len(self.ctcat)

        for k in mapunit.keys():
            mu[k] = mapunit[k][idx]


        if self.qscounts is None:
            self.qscounts = np.zeros((self.njack,self.nzbins))
            self.tcounts = np.zeros((self.njack,self.nzbins))

        if self.zbins is not None:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mu['redshift'].searchsorted(self.zbins[i])
                zhidx = mu['redshift'].searchsorted(self.zbins[i+1])

                if self.zeroind:
                    qidx, = np.where(self.ctcat[mu['ctcatid'][zlidx:zhidx]-1,3]==1)
                else:
                    qidx, = np.where(self.ctcat[mu['ctcatid'][zlidx:zhidx],3]==1)

                self.qscounts[self.jcount, i] = len(qidx)
                self.tcounts[self.jcount, i] = zhidx-zlidx

        else:
            if self.zeroind:
                qidx, = np.where(self.ctcat[mu['ctcatid']-1,3]==1)
            else:
                qidx, = np.where(self.ctcat[mu['ctcatid'],3]==1)

            self.qscounts[self.jcount,0] = len(qidx)
            self.tcounts[self.jcount,0] = len(mu['ctcatid'])

        del mu

    def reduce(self, rank=None, comm=None):
        if rank is not None:
            gqs = comm.gather(self.qscounts, root=0)
            gtc = comm.gather(self.tcounts, root=0)

            if rank==0:
                qcshape = [self.qscounts.shape[i] for i in range(len(self.qscounts.shape))]
                tcshape = [self.tcounts.shape[i] for i in range(len(self.tcounts.shape))]

                qcshape[0] = self.njacktot
                tcshape[0] = self.njacktot

                self.qscounts = np.zeros(qcshape)
                self.tcounts = np.zeros(tcshape)

                jc = 0
                for i, g in enumerate(gqs):
                    if g is None: continue
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

        ye = np.sqrt(self.varfquenched)
        ax.plot(zm, self.fquenched, **kwargs)
        ax.fill_between(zm, self.fquenched - ye, self.fquenched + ye, **kwargs)

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
            sax.set_xlabel(r'$z$', fontsize=16, labelpad=20)
            sax.set_ylabel(r'$f_{red}$', fontsize=16, labelpad=20)

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
                 cbins=None, cinds=None, varerr=False, 
                 watershed=False, wszbins=None, splitcolor=None,
                 **kwargs):
        Metric.__init__(self, ministry, catalog_type=catalog_type,tag=tag, **kwargs)
        self.zbins = zbins

        self.watershed = watershed
        self.wszbins   = wszbins
        
        if self.watershed and (self.wszbins is None):
            self.wszbins = [self.zbins[0], self.zbins[-1]]
        
        if zbins is None:
            self.nzbins = 1
        else:
            self.nzbins = len(zbins)-1
            self.zbins = np.array(self.zbins)

        if magbins is None:
            self.magbins = np.linspace(-25, -18, 30)
        else:
            self.magbins = magbins

        self.splitcolor = splitcolor

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

        self.varerr = varerr

        self.mapkeys = ['luminosity', 'redshift']
        self.aschema = 'galaxyonly'
        self.unitmap = {'luminosity':'mag'}

        self.qscounts = None
        self.tcounts  = None

    @jackknifeMap
    def map(self, mapunit):

        if self.qscounts is None:
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
                    if self.watershed:
                        tzlidx = mapunit['redshift'].searchsorted(self.wszbins[0])
                        tzhidx = mapunit['redshift'].searchsorted(self.wszbins[1])

                        ccounts, e0, e1 = np.histogram2d(mapunit['luminosity'][tzlidx:tzhidx,
                                                                               self.cinds[0]],
                                                         clr[tzlidx:tzhidx], 
                                                         bins=[np.linspace(-23,-16,60), self.cbins])
                        self.splitcolor = self.splitWatershed(ccounts.T, e1)
                        print(self.splitcolor)
                    else:
                        ccounts, self.cbins = np.histogram(clr[zlidx:zhidx], self.cbins)
                        self.splitcolor = self.splitBimodal(self.cbins[:-1], ccounts)

                    if self.splitcolor is None:
                        continue


                for j, lum in enumerate(self.magbins[:-1]):
                    lidx, = np.where((self.magbins[j]<mapunit['luminosity'][zlidx:zhidx,self.cinds[0]])
                                    & (mapunit['luminosity'][zlidx:zhidx,self.cinds[0]]<self.magbins[j+1]))
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
                lidx, = np.where((self.magbins[i]<mapunit['luminosity'][:,self.cinds[0]])
                                & (mapunit['luminosity'][:,self.cinds[0]]<self.magbins[i+1]))

                qidx, = np.where(clr[lidx]>self.splitcolor)

                self.qscounts[self.jcount,i,0] = len(qidx)
                self.tcounts[self.jcount,i,0] = len(lidx)


    def reduce(self, rank=None, comm=None):
        if rank is not None:
            gqs = comm.gather(self.qscounts, root=0)
            gtc = comm.gather(self.tcounts, root=0)

            if rank==0:
                qcshape = [self.qscounts.shape[i] for i in range(len(self.qscounts.shape))]
                tcshape = [self.tcounts.shape[i] for i in range(len(self.tcounts.shape))]

                qcshape[0] = self.njacktot
                tcshape[0] = self.njacktot

                self.qscounts = np.zeros(qcshape)
                self.tcounts = np.zeros(tcshape)

                jc = 0
                for i, g in enumerate(gqs):
                    if g is None: continue
                    nj = g.shape[0]
                    self.qscounts[jc:jc+nj,:] = g
                    self.tcounts[jc:jc+nj,:] = gtc[i]
                    jc += nj

                if self.varerr:
                    self.jfquenched = self.qscounts / self.tcounts
                    self.fquenched = np.sum(self.jfquenched, axis=0) / self.njacktot
                    self.varfquenched = np.sum(self.jfquenched - self.fquenched, axis=0) / self.njacktot
                else:
                    self.jqscounts = self.jackknife(self.qscounts, reduce_jk=False)
                    self.jtcounts = self.jackknife(self.tcounts, reduce_jk=False)

                    self.jfquenched = self.jqscounts / self.jtcounts
                    self.fquenched = np.sum(self.jfquenched, axis=0) / self.njacktot
                    self.varfquenched = np.sum((self.jfquenched - self.fquenched)**2, axis=0) * ( self.njacktot - 1) / self.njacktot

        else:
            if self.varerr:
                self.jfquenched = self.qscounts / self.tcounts
                self.fquenched = np.sum(self.jfquenched, axis=0) / self.njacktot
                self.varfquenched = np.sum(self.jfquenched - self.fquenched, axis=0) / self.njacktot
            else:
                self.jqscounts = self.jackknife(self.qscounts, reduce_jk=False)
                self.jtcounts = self.jackknife(self.tcounts, reduce_jk=False)

                self.jfquenched = self.jqscounts / self.jtcounts
                self.fquenched = np.sum(self.jfquenched, axis=0) / self.njacktot
                self.varfquenched = np.sum((self.jfquenched - self.fquenched)**2, axis=0) * ( self.njacktot - 1) / self.njacktot


    def visualize(self, f=None, ax=None, plotname=None,
                  compare=False, label=None, xlabel=None,
                  ylabel=None, onepanel=False, usez=None,
                  **kwargs):
        linestyles = ['-', '--','-.']

        if usez is None:
            usez = np.arange(self.nzbins)
        
        if f is None:
            if onepanel:
                f, ax = plt.subplots(1,figsize=(8,8), sharex=True, sharey=True)
                ax = np.atleast_2d(ax)
                newaxes = True
            else:
                f, ax = plt.subplots(1,self.nzbins, figsize=(8,8), sharex=True, sharey=True)
                ax = np.atleast_2d(ax)
                newaxes = True
        else:
            newaxes = False

        lm = (self.magbins[:-1]+self.magbins[1:])/2
        for iz, i in enumerate(usez):
            ye = np.sqrt(self.varfquenched[:,i])
            if onepanel:
                l1 = ax[0][0].plot(lm, self.fquenched[:,i], label=label,
                                   linestyle=linestyles[i%3],**kwargs)
                ax[0][0].fill_between(lm , self.fquenched[:,i]-ye,
                                      self.fquenched[:,i]+ye,
                                      linestyle=linestyles[i%3],
                                      alpha=0.5, **kwargs)
            else:
                l1 = ax[0][iz].plot(lm, self.fquenched[:,i], label=label, **kwargs)
                ax[0][iz].fill_between(lm , self.fquenched[:,i]-ye,
                                      self.fquenched[:,i]+ye,
                                      alpha=0.5, **kwargs)

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
            if xlabel is None:
                sax.set_xlabel(r'$Mag$', labelpad=30)
            else:
                sax.set_xlabel(xlabel,labelpad=30)
            if ylabel is None:
                sax.set_ylabel(r'$f_{red}$',labelpad=30)
            else:
                sax.set_ylabel(ylabel,labelpad=30)

        #plt.tight_layout()

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax, l1[0]

    def compare(self, othermetrics, plotname=None, labels=None,
                 onepanel=False, usez=None, **kwargs):
        tocompare = [self]
        tocompare.extend(othermetrics)

        if usez is not None:
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
                f, ax, l1 = m.visualize(compare=True,label=labels[i],
                                          color=Metric._color_list[i],
                                          onepanel=onepanel, usez=usez[i],
                                          **kwargs)
            else:
                f, ax, l1 = m.visualize(f=f, ax=ax, compare=True,
                                          label=labels[i],
                                          color=Metric._color_list[i],
                                          onepanel=onepanel, usez=usez[i],
                                          **kwargs)
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

    def __init__(self, ministry, fname, nbands=None,
                  xcol=None, ycol=None, ecol=None,
                  zbins=None,evolve=False,Q=None,
                  P=None,z0=None,skip_header=None,
                  **kwargs):

        LuminosityFunction.__init__(self,ministry,**kwargs)

        self.fname = fname

        if nbands is not None:
            self.nbands = nbands
        else:
            self.nbands = 5

        if zbins is not None:
            self.zbins = zbins
            self.nzbins = len(zbins)
        else:
            self.zbins = None
            self.nzbins = 1

        if skip_header is None:
            self.skip_header = 0
        else:
            self.skip_header = skip_header

        self.evolve = evolve
        self.Q = Q
        self.P = P

        if z0 is not None:
            self.z0 = z0
        else:
            self.z0 = 0.1

        if xcol is not None:
            self.xcol = xcol
        else:
            self.xcol = 0

        if ycol is not None:
            self.ycol = ycol
        else:
            self.ycol = 1

        if ecol is not None:
            self.ecol = ecol
        else:
            self.ecol = None



        #don't need to map this guy
        self.nomap = True
        self.loadLuminosityFunction()
        if self.evolve:
            self.evolveTableQP(self.Q, self.P, self.zbins,
                                z0=self.z0)

    def loadLuminosityFunction(self):
        """
        Read in the LF from self.fname. If self.fname is a list
        assumes that LFs in list correspond to zbins specified.
        If self.fname not a list, if more than 2 columns assumes
        first column is luminosities, second column is.
        """

        tab = np.genfromtxt(self.fname[0], skip_header=self.skip_header)
        if not self.evolve:
            self.luminosity_function = np.zeros((tab.shape[0], self.nbands, self.nzbins))

        else:
            self.luminosity_function = np.zeros((tab.shape[0], self.nbands, 1))

        if self.ecol is not None:
            self.ye = np.zeros(self.luminosity_function.shape)
            imult = 1
        else:
            self.ye = None
            imult = 2

        self.magmean = tab[:,self.xcol]

        if self.nzbins==1:
            for i in range(self.nzbins):
                for j in range(self.nbands):
                    self.luminosity_function[:,j,i] = tab[:,self.ycol]
                    if self.ecol is not None:
                        self.ye[:,j,i] = tab[:,self.ecol]
        else:
            if not self.evolve:
                assert((tab.shape[1]-1)==self.nzbins)
                for i in range(self.nzbins):
                    for j in range(self.nbands):
                        self.luminosity_function[:,j,i] = tab[:,i*imult+self.ycol]
                        if self.ecol is not None:
                            self.ye[:,j,i] = tab[:,i*imult+self.ecol]
            else:
                for j in range(self.nbands):
                    self.luminosity_function[:,j,0] = tab[:,self.ycol]
                    if self.ecol is not None:
                        self.ye[:,j,0] = tab[:,self.ecol]

        self.xmean = self.magmean
        self.y = self.luminosity_function

    def evolveTableQP(self, Q, P, zs, z0=0.1):

        elf = np.zeros((len(self.xmean),self.nbands,len(zs)))
        elfe = np.zeros((len(self.xmean),self.nbands,len(zs)))
        ex  = np.zeros((len(self.xmean),self.nbands,len(zs)))

        for i, z in enumerate(zs):
            elf[:,:,i] = self.luminosity_function[:,:,0] * 10 ** (0.4 * P * (z - z0))
            elfe[:,:,i] = self.ye[:,:,0]
            ex[:,:,i] = self.xmean.reshape(-1,1) - Q * (z - z0)

        self.luminosity_function = elf
        self.y = self.luminosity_function
        self.ye = elfe
        self.xmean = ex
