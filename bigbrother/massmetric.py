from __future__ import print_function, division
from .metric import Metric, GMetric, jackknifeMap
#if __name__=='__main__':
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import cosmocalc
from scipy.integrate import quad


class MassMetric(GMetric):
    """
    Restruct GMetric to magnitudes
    """

    def __init__(self, ministry, zbins=None, massbins=None,
                 catalog_type=None, tag=None, **kwargs):
        """
        Initialize a MassMetric object. Note, all metrics should define
        an attribute called mapkeys which specifies the types of data that they
        expect.

        Arguments
        ---------
        ministry : Ministry
            The ministry object that this metric is associated with.
        zbins : array-like
            An 1-d array containing the edges of the redshift bins to
            measure the metric in.
        massbins : array-like
            A 1-d array containing the edges of the mass bins to
            measure the metric in.
        """

        if massbins is None:
            massbins = np.logspace(10, 16, 40)

        if zbins is None:
            zbins = np.array([0.0, 10.0])

        self.massbins  = massbins
        self.zbins     = zbins
        self.nmassbins = len(self.massbins)-1
        self.nzbins    = len(self.zbins)-1


        GMetric.__init__(self, ministry, zbins=zbins, xbins=massbins,
                         catalog_type=catalog_type, tag=tag, **kwargs)

class MassFunction(MassMetric):

    def __init__(self, ministry, zbins=None, massbins=None, lightcone=True,
                 catalog_type=['halocatalog'], tag=None, **kwargs):

        if massbins is None:
            massbins = np.logspace(10, 16, 40)

        MassMetric.__init__(self, ministry, zbins=zbins, massbins=massbins,
                            catalog_type=catalog_type, tag=tag, **kwargs)

        self.aschema = 'haloonly'

        if lightcone:
            self.mapkeys   = ['halomass', 'redshift']
            self.unitmap = {'halomass':'msunh', 'redshift':'z'}
            self.lightcone = True
        else:
            self.mapkeys   = ['halomass']
            self.unitmap = {'halomass':'msunh'}
            self.lightcone = False

        self.masscounts = None


    @jackknifeMap
    def map(self, mapunit):

        #The number of mass definitions to measure mfcn for
        if len(mapunit['halomass'].shape)>1:
            self.ndefs = mapunit['halomass'].shape[1]
        else:
            self.ndefs = 1
            mapunit['halomass'] = np.atleast_2d(mapunit['halomass']).T

        #temporary fix for plotting w/ GMetric functions
        self.nbands = self.ndefs

        #Want to count galaxies in bins of luminosity for
        #self.nbands different bands in self.nzbins
        #redshift bins
        if self.masscounts is None:
            self.masscounts = np.zeros((self.njack,
                                        len(self.massbins)-1,
                                        self.ndefs,
                                        self.nzbins))

        if self.lightcone:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

                #Count galaxies in bins of luminosity
                for j in range(self.ndefs):
                    c, e = np.histogram(mapunit['halomass'][zlidx:zhidx,j],
                                        bins=self.massbins)
                    self.masscounts[self.jcount,:,j,i] += c
        else:
            for j in range(self.ndefs):
                c, e = np.histogram(mapunit['halomass'][:,j], bins=self.massbins)
                self.masscounts[self.jcount,:,j,0] += c

    def reduce(self, rank=None, comm=None):
        """
        Given counts in luminosity bins, generate a luminosity function.
        This will be called after all the mapunits are mapped by the map
        method. This turns total counts of galaxies into densities as appropriate
        for a luminosity function. The LF is then saved as an attribute of the
        LuminosityFunction object.
        """

        if rank is not None:
            gdata = comm.gather(self.masscounts, root=0)

            if rank==0:
                jc = 0
                dshape = self.masscounts.shape
                dshape = [dshape[i] for i in range(len(dshape))]
                dshape[0] = self.njacktot
                self.masscounts = np.zeros(dshape)

                for g in gdata:
                    if g is None: continue
                    nj = g.shape[0]
                    self.masscounts[jc:jc+nj,:,:,:] = g

                    jc += nj

                area = self.ministry.halocatalog.getArea(jackknife=True)
                self.jmass_function = np.zeros(self.masscounts.shape)
                vol = np.zeros((self.njacktot, self.nzbins))

                for i in range(self.nzbins):
                    vol[:,i] = self.ministry.calculate_volume(area, self.zbins[i], self.zbins[i+1])

                jmasscounts = self.jackknife(self.masscounts, reduce_jk=False)

                self.jmass_function = jmasscounts / vol.reshape(self.njacktot,1,1,self.nzbins)
                self.mass_function  = np.sum(self.jmass_function, axis=0) / self.njacktot
                self.varmass_function = np.sum((self.jmass_function - self.mass_function) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot

                self.y = self.mass_function
                self.ye = np.sqrt(self.varmass_function)
        else:
            area = self.ministry.halocatalog.getArea(jackknife=True)
            self.jmass_function = np.zeros(self.masscounts.shape)
            vol = np.zeros((self.njacktot, self.nzbins))

            for i in range(self.nzbins):
                vol[:,i] = self.ministry.calculate_volume(area, self.zbins[i], self.zbins[i+1])

            jmasscounts = self.jackknife(self.masscounts, reduce_jk=False)

            self.jmass_function = jmasscounts / vol.reshape(self.njacktot,1,1,self.nzbins)
            self.mass_function  = np.sum(self.jmass_function, axis=0) / self.njacktot
            self.varmass_function = np.sum((self.jmass_function - self.mass_function) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot

            self.y = self.mass_function
            self.ye = np.sqrt(self.varmass_function)



    def visualize(self, plotname=None, usecols=None, usez=None,fracdev=False,
                  ref_y=None, ref_x=[None], xlim=None, ylim=None, fylim=None,
                  f=None, ax=None, xlabel=None,ylabel=None,compare=False,**kwargs):

        if xlabel is None:
            xlabel = r"$M_{halo} \, [M_{\odot}\, h^{-1}]$"
        if ylabel is None:
            ylabel = r'$\phi \, [Mpc^{-3}\, h^{3}]$'

        return MassMetric.visualize(self, plotname=plotname, usecols=usecols, usez=usez,
                             fracdev=fracdev, ref_y=ref_y, ref_x=ref_x, xlim=xlim,
                             ylim=ylim, fylim=fylim, f=f, ax=ax, xlabel=xlabel,
                             ylabel=ylabel, compare=compare,logx=True,**kwargs)

class SimpleHOD(MassMetric):

    def __init__(self, ministry, zbins=None, massbins=None, lightcone=True,
                 catalog_type=['halocatalog'], tag=None, **kwargs):

        if massbins is None:
            massbins = np.logspace(10, 16, 40)

        MassMetric.__init__(self, ministry, zbins=zbins, massbins=massbins,
                            catalog_type=catalog_type, tag=tag, **kwargs)

        self.aschema = 'haloonly'

        if lightcone:
            self.mapkeys = ['halomass', 'occ', 'redshift']
            self.lightcone = True
        else:
            self.mapkeys = ['halomass', 'occ']
            self.lightcone = False

        self.unitmap = {'halomass':'msunh'}

        self.occcounts   = None
        self.sqocccounts = None
        self.halocounts  = None

    def map(self, mapunit):

        #The number of mass definitions to measure mfcn for
        if len(mapunit['halomass'].shape)>1:
            self.ndefs = mapunit['halomass'].shape[1]
        else:
            self.ndefs = 1
            mapunit['halomass'] = np.atleast_2d(mapunit['halomass']).T

        #temporary fix for plotting w/ GMetric functions
        self.nbands = self.ndefs

        #Want to count galaxies in bins of luminosity for
        #self.nbands different bands in self.nzbins
        #redshift bins
        if self.occcounts is None:
            self.occcounts = np.zeros((len(self.massbins)-1, self.ndefs,
                                       self.nzbins))
            self.sqocccounts = np.zeros((len(self.massbins)-1, self.ndefs,
                                         self.nzbins))
            self.halocounts = np.zeros((len(self.massbins)-1, self.ndefs,
                                        self.nzbins))

        if self.lightcone:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

                #Count galaxies in bins of mass
                for j in range(self.ndefs):
                    mb = np.digitize(mapunit['halomass'][zlidx:zhidx,j], bins=self.massbins)-1
                    for k in range(len(self.massbins)-1):
                        self.occcounts[k,j,i] += np.sum(mapunit['occ'][zlidx:zhidz][mb==k])
                        self.sqocccounts[k,j,i] += np.sum(mapunit['occ'][zlidx:zhidz][mb==k]**2)
                        self.halocounts[k,j,i] += len(mapunit['occ'][mb==k])
        else:
            for j in range(self.ndefs):
                mb = np.digitize(mapunit['halomass'][:,j], bins=self.massbins)-1
                for k in range(len(self.massbins)-1):
                    self.occcounts[k,j,i] += np.sum(mapunit['occ'][mb==k])
                    self.sqocccounts[k,j,i] += np.sum(mapunit['occ'][mb==k]**2)
                    self.halocounts[k,j,i] += len(mapunit['occ'][mb==k])


    def reduce(self, rank=None, comm=None):
        """
        Given counts in luminosity bins, generate a luminosity function.
        This will be called after all the mapunits are mapped by the map
        method. This turns total counts of galaxies into densities as appropriate
        for a luminosity function. The LF is then saved as an attribute of the
        LuminosityFunction object.
        """
        area = self.ministry.halocatalog.getArea()
        self.hod = self.occcounts/self.halocounts
        self.hoderr = np.sqrt((self.sqocccounts - self.hod**2)/self.halocounts)

        self.y = self.hod
        self.ye = self.hoderr


class GalHOD(MassMetric):

    def __init__(self, ministry, zbins=None, massbins=None, lightcone=True,
                 catalog_type=['galaxycatalog'], tag=None, magcuts=None,
                 cutband=None, **kwargs):

        if massbins is None:
            massbins = np.logspace(10, 16, 40)

        MassMetric.__init__(self, ministry, zbins=zbins, massbins=massbins,
                            catalog_type=catalog_type, tag=tag, **kwargs)

        self.magcuts = magcuts

        if self.magcuts is not None:
            self.usemag = True
            self.nmagcuts= len(self.magcuts)

        else:
            self.usemag = False
            self.nmagcuts = 1

        self.cutband = cutband

        self.aschema = 'galaxyonly'

        if lightcone:
            if self.usemag:
                self.mapkeys = ['halomass', 'central', 'redshift', 'haloid', 'rhalo', 'luminosity']
            else:
                self.mapkeys = ['halomass', 'central', 'redshift', 'haloid', 'rhalo']

            self.lightcone = True

        else:
            if self.usemag:
                self.mapkeys = ['halomass', 'central', 'haloid', 'rhalo', 'luminosity']
            else:
                self.mapkeys = ['halomass', 'central', 'haloid', 'rhalo']

            self.lightcone = False

        if self.usemag:
            self.unitmap = {'halomass':'msunh', 'rhalo':'mpch', 'luminosity':'mag'}
        else:
            self.unitmap = {'halomass':'msunh', 'rhalo':'mpch'}


        self.sqocccounts  = None
        self.cocccounts   = None
        self.sqsocccounts = None
        self.sqcocccounts = None
        self.halocounts   = None

    def map(self, mapunit):
        #The number of mass definitions to measure mfcn for
        if len(mapunit['halomass'].shape)>1:
            self.ndefs = mapunit['halomass'].shape[1]
        else:
            self.ndefs = 1
            mapunit['halomass'] = np.atleast_2d(mapunit['halomass']).T

        #temporary fix for plotting w/ GMetric functions
        self.nbands = self.ndefs

        #Want to count galaxies in bins of luminosity for
        #self.nbands different bands in self.nzbins
        #redshift bins
        if self.socccounts is None:
            self.socccounts = np.zeros((len(self.massbins)-1, self.ndefs,
                                       self.nmagcuts, self.nzbins))
            self.cocccounts = np.zeros((len(self.massbins)-1, self.ndefs,
                                       self.nmagcuts, self.nzbins))

            self.sqsocccounts = np.zeros((len(self.massbins)-1, self.ndefs,
                                         self.nmagcuts, self.nzbins))
            self.sqcocccounts = np.zeros((len(self.massbins)-1, self.ndefs,
                                         self.nmagcuts, self.nzbins))

            self.halocounts = np.zeros((len(self.massbins)-1, self.ndefs,
                                        self.nmagcuts, self.nzbins))

        if self.lightcone:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

                #Count number of halos in mass bins before making mag counts
                u, uidx = np.unique(mapunit['haloid'], return_index=True)

                for j in range(self.ndefs):
                    c, e = np.histogram(mapunit['halomass'][zlidx:zhidx,j][uidx],
                                          bins=self.massbins)
                    self.halocounts[:,j,i] += c.reshape(self.halocounts[:,j,i].shape)

                    for k in range(self.nmagcuts):
                        if self.usemag:
                            if self.cutband is not None:
                                lidx = (mapunit['luminosity'][zlidx:zhidx, self.cutband]<self.magcuts[k])
                            else:
                                lidx = (mapunit['luminosity'][zlidx:zhidx]<self.magcuts[k])
                            cidx = mapunit['central'][zlidx:zhidx][lidx]==1

                            c, e = np.histogram(mapunit['halomass'][zlidx:zhidx,j][lidx][cidx], bins=self.massbins)
                            c = c.reshape(self.cocccounts[:,j,k,i].shape)
                            self.cocccounts[:,j,k,i] += c
                            self.sqcocccounts[:,j,k,i] += c**2

                            c, e = np.histogram(mapunit['halomass'][zlidx:zhidx,j][lidx][~cidx], bins=self.massbins)
                            c = c.reshape(self.cocccounts[:,j,k,i].shape)
                            self.socccounts[:,j,k,i] += c
                            self.sqsocccounts[:,j,k,i] += c**2
                        else:
                            cidx = mapunit['central'][zlidx:zhidx]==1

                            c, e = np.histogram(mapunit['halomass'][zlidx:zhidx,j][cidx], bins=self.massbins)
                            self.cocccounts[:,j,k,i] += c
                            self.sqcocccounts[:,j,k,i] += c**2

                            c, e = np.histogram(mapunit['halomass'][zlidx:zhidx,j][~cidx], bins=self.massbins)
                            self.socccounts[:,j,k,i] += c
                            self.sqsocccounts[:,j,k,i] += c**2



        else:
            raise(NotImplementedError)


    def reduce(self, rank=None, comm=None):
        """
        Given counts in luminosity bins, generate a luminosity function.
        This will be called after all the mapunits are mapped by the map
        method. This turns total counts of galaxies into densities as appropriate
        for a luminosity function. The LF is then saved as an attribute of the
        LuminosityFunction object.
        """

        self.shod = self.socccounts/self.halocounts
        self.chod = self.cocccounts/self.halocounts
        self.shoderr = np.sqrt((self.sqsocccounts - self.shod**2)/self.halocounts)
        self.choderr = np.sqrt((self.sqcocccounts - self.chod**2)/self.halocounts)

        self.y = self.shod + self.chod
        self.ye = np.sqrt(self.shoderr**2 + self.choderr**2)

        if rank is not None:
            gsocc = comm.gather(self.socccounts, root=0)
            gcocc = comm.gather(self.cocccounts, root=0)
            gsqsocc = comm.gather(self.sqsocccounts, root=0)
            gsqcocc = comm.gather(self.sqcocccounts, root=0)
            ghcounts = comm.gather(self.halocounts, root=0)

            if rank==0:
                jc = 0
                hshape = [self.halo_counts.shape[i] for i in range(len(self.halo_counts.shape))]

                hshape[0] = self.njacktot

                self.socccounts = np.zeros(hshape)
                self.sqsocccounts = np.zeros(hshape)
                self.cocccounts = np.zeros(hshape)
                self.sqcocccounts = np.zeros(hshape)
                self.halocounts = np.zeros(hshape)

                for i, g in enumerate(gsocc):
                    if g is None: continue
                    nj = g.shape[0]
                    self.socccounts[jc:jc+nj,:,:,:] = g
                    self.sqsocccounts[jc:jc+nj,:,:,:] = gsqsocc[i]
                    self.cocccounts[jc:jc+nj,:,:,:] = gcocc[i]
                    self.sqcocccounts[jc:jc+nj,:,:,:] = gsqcocc[i]
                    self.halocounts[jc:jc+nj,:,:,:] = ghcounts[i]

                    jc += nj

                self.jsocccounts = self.jackknife(self.socccounts, reduce_jk=False)
                self.jsqsocccounts = self.jackknife(self.sqsocccounts, reduce_jk=False)
                self.jcocccounts = self.jackknife(self.cocccounts, reduce_jk=False)
                self.jsqcocccounts = self.jackknife(self.sqcocccounts, reduce_jk=False)
                self.jhalocounts = self.jackknife(self.sqcocccounts, reduce_jk=False)

                self.jshod = self.jsocccounts/self.jhalocounts
                self.jchod = self.jcocccounts/self.jhalocounts

                self.shod = np.sum(self.jshod, axis=0)
                self.chod = np.sum(self.jchod, axis=0)

                self.shoderr = np.sqrt((self.sqsocccounts - self.shod**2)/self.halocounts)
                self.choderr = np.sqrt((self.sqcocccounts - self.chod**2)/self.halocounts)

                self.y = self.shod + self.chod
                self.ye = np.sqrt(self.shoderr**2 + self.choderr**2)

        else:
            self.joccmass, self.occmass, self.varoccmass = self.jackknife(self.occ/self.count)

            if self.njacktot < 2:

                _, self.varoccmass, _ = self.jackknife((self.count*self.occsq - self.occ**2)/(self.count*(self.count-1)))

            self.y = self.occmass
            self.ye = np.sqrt(self.varoccmass)



class OccMass(MassMetric):

    def __init__(self, ministry, zbins=None, massbins=None, lightcone=True,
                 catalog_type=['halocatalog'], tag=None, **kwargs):

        if massbins is None:
            massbins = np.logspace(10, 16, 40)

        MassMetric.__init__(self, ministry, zbins=zbins, massbins=massbins,
                            catalog_type=catalog_type, tag=tag, **kwargs)

        if lightcone:
            self.mapkeys   = ['halomass', 'occ', 'redshift']
            self.lightcone = True
        else:
            self.mapkeys   = ['halomass', 'occ']
            self.lightcone = False

        self.aschema = 'haloonly'
        self.unitmap = {'halomass':'msunh'}

        self.occ    = None
        self.occsq  = None
        self.count  = None
        self.ndefs = 1

    @jackknifeMap
    def map(self, mapunit):

        if len(mapunit['halomass'].shape)>1:
            self.ndefs = mapunit['halomass'].shape[1]
        else:
            self.ndefs = 1
            mapunit['halomass'] = np.atleast_2d(mapunit['halomass']).T

        self.nbands = self.ndefs

        if self.occ is None:
            self.occ   = np.zeros((self.njack,self.nmassbins,self.ndefs,self.nzbins))
            self.occsq = np.zeros((self.njack,self.nmassbins,self.ndefs,self.nzbins))
            self.count = np.zeros((self.njack,self.nmassbins,self.ndefs,self.nzbins))

        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])
            for j in range(self.ndefs):
                mb = np.digitize(mapunit['halomass'][zlidx:zhidx,j], bins=self.massbins)

                for k, m in enumerate(self.massbins[:-1]):
                    o  = mapunit['occ'][zlidx:zhidx][mb==k]
                    self.occ[self.jcount,k,j,i]   += np.sum(o)
                    self.occsq[self.jcount,k,j,i] += np.sum(o**2)
                    self.count[self.jcount,k,j,i] += np.sum(mb==k)

    def reduce(self, rank=None, comm=None):
        if rank is not None:
            gocc = comm.gather(self.occ, root=0)
            goccsq = comm.gather(self.occsq, root=0)
            gcount = comm.gather(self.count, root=0)

            if rank==0:
                jc = 0
                oshape = [self.njack,self.nmassbins,self.ndefs,self.nzbins]
                osshape = [self.njack,self.nmassbins,self.ndefs,self.nzbins]
                cshape = [self.njack,self.nmassbins,self.ndefs,self.nzbins]

                oshape[0] = self.njacktot
                osshape[0] = self.njacktot
                cshape[0] = self.njacktot

                self.occ = np.zeros(oshape)
                self.occsq = np.zeros(osshape)
                self.cshape = np.zeros(cshape)

                for i, g in enumerate(gocc):
                    if g is None: continue
                    nj = g.shape[0]
                    self.occ[jc:jc+nj,:,:,:] = g
                    self.occsq[jc:jc+nj,:,:,:] = goccsq[i]
                    self.count[jc:jc+nj,:,:,:] = gcount[i]

                    jc += nj

                self.joccmass, self.occmass, self.varoccmass = self.jackknife(self.occ/self.count)

                self.y = self.occmass
                self.ye = np.sqrt(self.varoccmass)

        else:
            self.joccmass, self.occmass, self.varoccmass = self.jackknife(self.occ/self.count)

            if self.njacktot < 2:

                _, self.varoccmass, _ = self.jackknife((self.count*self.occsq - self.occ**2)/(self.count*(self.count-1)))

            self.y = self.occmass
            self.ye = np.sqrt(self.varoccmass)


    def visualize(self, plotname=None, usecols=None, usez=None,fracdev=False,
                  ref_y=None, ref_x=[None], xlim=None, ylim=None, fylim=None,
                  f=None, ax=None, xlabel=None,ylabel=None,compare=False,**kwargs):

        if xlabel is None:
            xlabel = r"$M_{halo} \, [M_{\odot}\, h^{-1}]$"
        if ylabel is None:
            ylabel = r'$<N>$'

        return MassMetric.visualize(self, plotname=plotname, usecols=usecols, usez=usez,
                             fracdev=fracdev, ref_y=ref_y, ref_x=ref_x, xlim=xlim,
                             ylim=ylim, fylim=fylim, f=f, ax=ax, xlabel=xlabel,
                             ylabel=ylabel, compare=compare,logx=True,**kwargs)

class TinkerMassFunction(MassMetric):

    def __init__(self, ministry, zbins=None, massbins=None, lightcone=True,
                 catalog_type=['halocatalog'], tag=None, **kwargs):

        if massbins is None:
            massbins = np.logspace(10, 16, 40)

        MassMetric.__init__(self, ministry, zbins=zbins, massbins=massbins,
                            catalog_type=catalog_type, tag=tag, **kwargs)

        self.aschema = 'haloonly'

        if lightcone:
            self.mapkeys   = ['halomass', 'redshift']
            self.lightcone = True
        else:
            self.mapkeys   = ['halomass']
            self.lightcone = False

        self.unitmap = {'halomass':'msunh'}
        self.nomap = True
        self.calcMassFunction(z=zbins)

    def calcMassFunction(self, z=0, delta=200):

	# mass bins, definitions, z bins

	if hasattr(delta, '__iter__'):
            self.ndefs = len(delta)
        else:
            self.ndefs = 1
	    delta = [delta]

	if hasattr(z , '__iter__'):
            self.nzbins = len(z)
	else:
	    z=[z]
            self.nzbins = 1

	self.nbands = self.ndefs

	cd = {
               	"OmegaM":0.286,
               "OmegaB":0.05,
               "OmegaDE":0.714,
               "OmegaK":0.0,
               "h":0.7,
               "Sigma8":0.82,
               "SpectralIndex":0.96,
               "w0":-1.0,
               "wa":0.0
	}
	cosmocalc.set_cosmology(cd)

	mass_fn_n     = np.zeros((len(self.massbins)-1, self.ndefs, self.nzbins))
	mass_fn_error = np.zeros((len(self.massbins)-1, self.ndefs, self.nzbins))

        for k in range(self.nzbins):
		for j in range(self.ndefs):
	            for i in range(len(self.massbins)-1):
        	        integral = quad(lambda mass: cosmocalc.tinker2008_mass_function(mass, 1/(1+z[k]), delta[j]), self.massbins[i], self.massbins[i+1])
                	mass_fn_n[i][j][k] = integral[0]
                	mass_fn_error[i][j][k] = integral[1]

	self.y  = mass_fn_n
	self.ye = mass_fn_error

    def map(self, mapunit):
        self.map(mapunit)

    def reduce(self, rank=None, comm=None):
        self.reduce()

    def visualize(self, plotname=None, usecols=None, usez=None,fracdev=False,
                  ref_y=None, ref_x=[None], xlim=None, ylim=None, fylim=None,
                  f=None, ax=None, xlabel=None,ylabel=None,compare=False,**kwargs):

        if xlabel is None:
            xlabel = r"$M_{halo} \, [M_{\odot}\, h^{-1}]$"
        if ylabel is None:
            ylabel = r"$N \, [Mpc^{-3}\, h^{3}]$"

        return MassMetric.visualize(self, plotname=plotname,
                                    usecols=usecols, usez=usez,
                                    fracdev=fracdev,
                                    ref_y=ref_y, ref_x=ref_x,
                                    xlim=xlim, ylim=ylim,
                                    fylim=fylim, f=f, ax=ax,
                                    xlabel=xlabel,
                                    ylabel=ylabel,
                                    compare=compare,logx=True,
                                    **kwargs)

class Richness(MassMetric):
    def __init__(self, ministry, zbins=None, massbins=None,
                  lightcone=True,
                  catalog_type=['galaxycatalog'], tag=None,
                  colorbins=None, maxrhalo=None, minlum=None,
                  redsplit=None, splitinfo=False, **kwargs):

        if massbins is None:
            massbins = np.logspace(12, 15, 20)
        else:
            massbins = massbins

        if (lightcone):
            if hasattr(zbins, '__iter__'):
                self.zbins = zbins
            else:
                if (type(zbins)==int and zbins>1):
                    self.zbins = np.logspace(-4, 1, zbins)
                else:
                    self.zbins = np.logspace(-4, 1, 5)
        else:
            self.zbins = [0, 10]
        self.nzbins = len(self.zbins) - 1

        MassMetric.__init__(self, ministry, zbins=self.zbins, massbins=massbins,
                            catalog_type=catalog_type, tag=tag, **kwargs)

        if colorbins is None:
            self.colorbins = 100
        else:
            self.colorbins = colorbins

        self.split_info = splitinfo

        self.aschema = 'galaxyonly'

        if lightcone:
            self.mapkeys   = ['halomass', 'redshift', 'luminosity', 'haloid', 'rhalo']
            self.lightcone = True
        else:
            self.mapkeys   = ['halomass', 'luminosity', 'haloid', 'rhalo']
            self.lightcone = False

        self.unitmap = {'halomass':'msunh'}
        self.nomap = False

        if maxrhalo is None:
            self.max_rhalo  = 1
        if minlum is None:
            self.min_lum    = -19

        self.splitcolor = redsplit
        self.nbands = 1

        self.halo_counts           = None
        self.galaxy_counts         = None
        self.galaxy_counts_squared = None

        self.meanc = []

    @jackknifeMap
    def map(self, mapunit):
        # must convert mapunit dict to a recarray

        self.galaxy_counts         = np.zeros((self.njack, len(self.massbins) - 1, 1, len(self.zbins) - 1))
        self.galaxy_counts_squared = np.zeros((self.njack, len(self.massbins) - 1, 1, len(self.zbins) - 1))
        self.halo_counts           = np.zeros((self.njack, len(self.massbins) - 1, 1, len(self.zbins) - 1))

        dtype = [(key, mapunit[key].dtype, np.shape(mapunit[key])[1:]) for key in mapunit.keys()]

        for ziter in range(len(self.zbins)-1):

            zcut = ((mapunit['redshift'] >= self.zbins[ziter]) & (mapunit['redshift'] < self.zbins[ziter+1]))

            g_r_color = mapunit['luminosity'][zcut][:,0] - mapunit['luminosity'][zcut][:,1] # get g-r color
            color_counts, color_bins = np.histogram(g_r_color, self.colorbins) # place colors into bins

            if self.splitcolor is None:
                self.splitcolor = self.splitBimodal(color_bins[:-1], color_counts)

            previd = -1
            halo_ids = np.unique(mapunit['haloid'][zcut])
            if len(halo_ids)<2:
                continue
            red_galaxy_counts = np.zeros(len(halo_ids)-1) # number of red galaxies in each unique halo

            # cut of galaxies: within max_rhalo of parent halo, above min_lum magnitude, and red
            cut_array =((mapunit['rhalo'] < self.max_rhalo) & (mapunit['luminosity'][:,2] < self.min_lum)
                & ((mapunit['luminosity'][:,0] - mapunit['luminosity'][:,1] >= self.splitcolor)) & ((mapunit['redshift'] >= self.zbins[ziter]) & (mapunit['redshift'] < self.zbins[ziter+1])))
            data_cut = np.recarray((len(cut_array[cut_array]), ), dtype)
            for key in mapunit.keys():
                data_cut[key] = mapunit[key][cut_array]

            data_cut.sort(order='haloid')
            if len(data_cut)==0:
                continue
            elif len(data_cut)==1:
                newhalos = np.zeros(2, dtype=np.int)
                newhalos[1] = 1
            else:
                idx = data_cut['haloid'][1:]-data_cut['haloid'][:-1]
                newhalos = np.where(idx != 0)[0]
                newhalos = np.hstack([[0], newhalos + 1, [len(data_cut) - 1]])

            uniquehalos = data_cut[newhalos[:-1]]
            red_counts = newhalos[1:]-newhalos[:-1]
            mass_bin_indices = np.digitize(uniquehalos['halomass'], self.massbins)

            self.halo_counts[self.jcount,:,0,ziter] += np.histogram(uniquehalos['halomass'], bins=self.massbins)[0]

            for i in range(len(self.massbins)-1):
                self.galaxy_counts[self.jcount, i,0,ziter]         += np.sum(red_counts[(mass_bin_indices == i+1)])
                self.galaxy_counts_squared[self.jcount, i,0,ziter] += np.sum(red_counts[(mass_bin_indices == i+1)]**2)


    def reduce(self,rank=None,comm=None):
        if rank is not None:
            ghc = comm.gather(self.halo_counts, root=0)
            ggc = comm.gather(self.galaxy_counts, root=0)
            ggcs = comm.gather(self.galaxy_counts_squared, root=0)

            if rank==0:
                hshape = [self.halo_counts.shape[i] for i in range(len(self.halo_counts.shape))]
                gshape = [self.galaxy_counts.shape[i] for i in range(len(self.galaxy_counts.shape))]
                gsshape = [self.galaxy_counts_squared.shape[i] for i in range(len(self.galaxy_counts.shape))]

                hshape[0] = self.njacktot
                gshape[0] = self.njacktot
                gsshape[0] = self.njacktot
                
                self.halo_counts = np.zeros(hshape)
                self.galaxy_counts = np.zeros(gshape)
                self.galaxy_counts_squared = np.zeros(gsshape)

                jc = 0
                for i, g in enumerate(ghc):
                    if g is None: continue
                    nj = g.shape[0]
                    self.halo_counts[jc:jc+nj,:,:,:] = g
                    self.galaxy_counts[jc:jc+nj,:,:,:] = ggc[i]
                    self.galaxy_counts_squared[jc:jc+nj,:,:,:] = ggcs[i]

                    jc += nj

                self.jgalaxy_counts = self.jackknife(self.galaxy_counts, reduce_jk=False)
                self.jhalo_counts = self.jackknife(self.halo_counts, reduce_jk=False)
                self.jgalaxy_counts_squared = self.jackknife(self.galaxy_counts_squared,reduce_jk=False)

                jmass_richness = self.jgalaxy_counts/self.jhalo_counts

                self.mass_richness = np.sum(jmass_richness, axis=0) / self.njacktot
                self.varmass_richness = np.sum((jmass_richness - self.mass_richness) ** 2, axis=0) * (self.njacktot - 1 ) / self.njacktot
                self.galaxy_counts_squared = np.sum(self.jgalaxy_counts_squared, axis=0) / self.njacktot
                self.halo_counts = np.sum(self.jhalo_counts, axis=0) / self.njacktot

                self.y           = self.mass_richness

                if self.njacktot==1:
                    self.ye = np.sqrt(self.galaxy_counts_squared / self.halo_counts - self.y**2)
                else:
                    self.ye          = np.sqrt(self.varmass_richness)
        else:

            self.jgalaxy_counts = self.jackknife(self.galaxy_counts, reduce_jk=False)
            self.jhalo_counts = self.jackknife(self.halo_counts, reduce_jk=False)
            self.jgalaxy_counts_squared = self.jackknife(self.galaxy_counts_squared, reduce_jk=False)

            jmass_richness = self.jgalaxy_counts/self.jhalo_counts

            self.mass_richness = np.sum(jmass_richness, axis=0) / self.njacktot
            self.varmass_richness = np.sum((self.jhalo_counts - self.mass_richness) ** 2, axis=0) * (self.njacktot - 1 ) / self.njacktot
            self.galaxy_counts_squared = np.sum( self.jgalaxy_counts_squared, axis=0) / self.njacktot
            self.halo_counts = np.sum(self.jhalo_counts, axis=0) / self.njacktot

            self.y           = self.mass_richness
            if self.njack==1:
                self.ye = np.sqrt(self.galaxy_counts_squared / self.halo_counts - self.y**2)
            else:
                self.ye          = np.sqrt(self.varmass_richness)


    def visualize(self, plotname=None, usecols=None, usez=None,fracdev=False,
                  ref_y=None, ref_x=[None], xlim=None, ylim=None, fylim=None,
                  f=None, ax=None, xlabel=None,ylabel=None,compare=False,**kwargs):

        if xlabel is None:
            xlabel = r"$M_{halo} \, [M_{\odot}\, h^{-1}]$"
        if ylabel is None:
            ylabel = r"$<N_{red}> \, [Mpc^{-3}\, h^{3}]$"

        return MassMetric.visualize(self, plotname=plotname,
                                    usecols=usecols, usez=usez,
                                    fracdev=fracdev,
                                    ref_y=ref_y, ref_x=ref_x,
                                    xlim=xlim, ylim=ylim,
                                    fylim=fylim, f=f, ax=ax,
                                    xlabel=xlabel,
                                    ylabel=ylabel,
                                    compare=compare,logx=True,
                                    **kwargs)
