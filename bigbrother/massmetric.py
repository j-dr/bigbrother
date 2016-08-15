from __future__ import print_function, division
from .metric import Metric, GMetric
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
                 catalog_type=None, tag=None):
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
                         catalog_type=catalog_type, tag=tag)

class MassFunction(MassMetric):

    def __init__(self, ministry, zbins=None, massbins=None, lightcone=True,
                 catalog_type=['halocatalog'], tag=None):

        if massbins is None:
            massbins = np.logspace(10, 16, 40)

        MassMetric.__init__(self, ministry, zbins=zbins, massbins=massbins,
                            catalog_type=catalog_type, tag=tag)

        self.aschema = 'haloonly'

        if lightcone:
            self.mapkeys   = ['mass', 'redshift']
            self.lightcone = True
        else:
            self.mapkeys   = ['mass']
            self.lightcone = False

        self.unitmap = {'mass':'msunh'}

    def map(self, mapunit):

        #The number of mass definitions to measure mfcn for
        if len(mapunit['mass'].shape)>1:
            self.ndefs = mapunit['mass'].shape[1]
        else:
            self.ndefs = 1
            mapunit['mass'] = np.atleast_2d(mapunit['mass']).T

        #temporary fix for plotting w/ GMetric functions
        self.nbands = self.ndefs

        #Want to count galaxies in bins of luminosity for
        #self.nbands different bands in self.nzbins
        #redshift bins
        if not hasattr(self, 'masscounts'):
            self.masscounts = np.zeros((len(self.massbins)-1, self.ndefs,
                                       self.nzbins))

        if self.lightcone:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

                #Count galaxies in bins of luminosity
                for j in range(self.ndefs):
                    c, e = np.histogram(mapunit['mass'][zlidx:zhidx,j],
                                        bins=self.massbins)
                    self.masscounts[:,j,i] += c
        else:
            for j in range(self.ndefs):
                c, e = np.histogram(mapunit['mass'][:,j], bins=self.massbins)
                self.masscounts[:,j,0] += c


    def reduce(self):
        """
        Given counts in luminosity bins, generate a luminosity function.
        This will be called after all the mapunits are mapped by the map
        method. This turns total counts of galaxies into densities as appropriate
        for a luminosity function. The LF is then saved as an attribute of the
        LuminosityFunction object.
        """
        area = self.ministry.halocatalog.getArea()
        self.mass_function = self.masscounts

        for i in range(self.nzbins):
            vol = self.ministry.calculate_volume(area, self.zbins[i], self.zbins[i+1])
            self.mass_function[:,:,i] /= vol

        self.y = self.mass_function

    def visualize(self, plotname=None, usecols=None, usez=None,fracdev=False,
                  ref_y=None, ref_x=[None], xlim=None, ylim=None, fylim=None,
                  f=None, ax=None, xlabel=None,ylabel=None,compare=False,**kwargs):

        if xlabel is None:
            xlabel = r"$M_{halo} \, [M_{\odot}\, h^{-1}]$"
        if ylabel is None:
            ylabel = r'$\phi \, [Mpc^{-3}\, h^{3}]$'

        MassMetric.visualize(self, plotname=plotname, usecols=usecols, usez=usez,
                             fracdev=fracdev, ref_y=ref_y, ref_x=ref_x, xlim=xlim,
                             ylim=ylim, fylim=fylim, f=f, ax=ax, xlabel=xlabel,
                             ylabel=ylabel, compare=compare,logx=True,**kwargs)



class SimpleHOD(MassMetric):

    def __init__(self, ministry, zbins=None, massbins=None, lightcone=True,
                 catalog_type=['halocatalog'], tag=None):

        if massbins is None:
            massbins = np.logspace(10, 16, 40)

        MassMetric.__init__(self, ministry, zbins=zbins, massbins=massbins,
                            catalog_type=catalog_type, tag=tag)

        self.aschema = 'haloonly'

        if lightcone:
            self.mapkeys = ['mass', 'occ', 'redshift']
            self.lightcone = True
        else:
            self.mapkeys = ['mass', 'occ']
            self.lightcone = False

        self.unitmap = {'mass':'msunh'}

    def map(self, mapunit):

        #The number of mass definitions to measure mfcn for
        if len(mapunit['mass'].shape)>1:
            self.ndefs = mapunit['mass'].shape[1]
        else:
            self.ndefs = 1
            mapunit['mass'] = np.atleast_2d(mapunit['mass']).T

        #temporary fix for plotting w/ GMetric functions
        self.nbands = self.ndefs

        #Want to count galaxies in bins of luminosity for
        #self.nbands different bands in self.nzbins
        #redshift bins
        if not hasattr(self, 'occcounts'):
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
                    mb = np.digitize(mapunit['mass'][zlidx:zhidx,j], bins=self.massbins)-1
                    for k in range(len(self.massbins)-1):
                        self.occcounts[k,j,i] += np.sum(mapunit['occ'][zlidx:zhidz][mb==k])
                        self.sqocccounts[k,j,i] += np.sum(mapunit['occ'][zlidx:zhidz][mb==k]**2)
                        self.halocounts[k,j,i] += len(mapunit['occ'][mb==k])
        else:
            for j in range(self.ndefs):
                mb = np.digitize(mapunit['mass'][:,j], bins=self.massbins)-1
                for k in range(len(self.massbins)-1):
                    self.occcounts[k,j,i] += np.sum(mapunit['occ'][mb==k])
                    self.sqocccounts[k,j,i] += np.sum(mapunit['occ'][mb==k]**2)
                    self.halocounts[k,j,i] += len(mapunit['occ'][mb==k])


    def reduce(self):
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
                 cutband=None):

        if massbins is None:
            massbins = np.logspace(10, 16, 40)

        MassMetric.__init__(self, ministry, zbins=zbins, massbins=massbins,
                            catalog_type=catalog_type, tag=tag)

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
                self.mapkeys = ['mass', 'redshift', 'haloid', 'rhalo', 'luminosity']
            else:
                self.mapkeys = ['mass', 'redshift', 'haloid', 'rhalo']

            self.lightcone = True

        else:
            if self.usemag:
                self.mapkeys = ['mass', 'central', 'haloid', 'rhalo', 'luminosity']
            else:
                self.mapkeys = ['mass', 'central', 'haloid', 'rhalo']

            self.lightcone = False

        if self.usemag:
            self.unitmap = {'mass':'msunh', 'rhalo':'mpch', 'luminosity':'mag'}
        else:
            self.unitmap = {'mass':'msunh', 'rhalo':'mpch'}

    def map(self, mapunit):
        #The number of mass definitions to measure mfcn for
        if len(mapunit['mass'].shape)>1:
            self.ndefs = mapunit['mass'].shape[1]
        else:
            self.ndefs = 1
            mapunit['mass'] = np.atleast_2d(mapunit['mass']).T

        #temporary fix for plotting w/ GMetric functions
        self.nbands = self.ndefs

        #Want to count galaxies in bins of luminosity for
        #self.nbands different bands in self.nzbins
        #redshift bins
        if not hasattr(self, 'socccounts'):
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
                    c, e = np.histogram(mapunit['mass'][zlidx:zhidx,j][uidx],
                                          bins=self.massbins)
                    self.halocounts[:,j,i] += c

                    for k in range(self.nmagcuts):
                        if self.usemag:
                            lidx = (mapunit['luminosity'][zlidx:zhidx]<self.magcuts[k+1])
                            cidx = mapunit['central'][zlidx:zhidx][lidx]==1

                            c, e = np.histogram(mapunit['mass'][zlidx:zhidx,j][lidx][cidx], bins=np.massbins)
                            self.cocccounts[:,j,k,i] += c
                            self.sqcocccounts[:,j,k,i] += c**2

                            c, e = np.histogram(mapunit['mass'][zlidx:zhidx,j][lidx][~cidx], bins=np.massbins)
                            self.socccounts[:,j,k,i] += c
                            self.sqsocccounts[:,j,k,i] += c**2
                        else:
                            cidx = mapunit['central'][zlidx:zhidx]==1

                            c, e = np.histogram(mapunit['mass'][zlidx:zhidx,j][cidx], bins=np.massbins)
                            self.cocccounts[:,j,k,i] += c
                            self.sqcocccounts[:,j,k,i] += c**2

                            c, e = np.histogram(mapunit['mass'][zlidx:zhidx,j][~cidx], bins=np.massbins)
                            self.socccounts[:,j,k,i] += c
                            self.sqsocccounts[:,j,k,i] += c**2



        else:
            raise(NotImplementedError)
#            for j in range(self.ndefs):
#                mb = np.digitize(mu['mass'][:,j], bins=self.massbins)-1
#                for k in range(len(self.massbins)-1):
#                    self.occcounts[k,j,i] += np.sum(mu['occ'][mb==k])
#                    self.sqocccounts[k,j,i] += np.sum(mu['occ'][mb==k]**2)
#                    self.halocounts[k,j,i] += len(mu['occ'][mb==k])


    def reduce(self):
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



class OccMass(MassMetric):

    def __init__(self, ministry, zbins=None, massbins=None, lightcone=True,
                 catalog_type=['halocatalog'], tag=None):

        if massbins is None:
            massbins = np.logspace(10, 16, 40)

        MassMetric.__init__(self, ministry, zbins=zbins, massbins=massbins,
                            catalog_type=catalog_type, tag=tag)

        if lightcone:
            self.mapkeys   = ['mass', 'occ', 'redshift']
            self.lightcone = True
        else:
            self.mapkeys   = ['mass', 'occ']
            self.lightcone = False

        self.aschema = 'haloonly'
        self.unitmap = {'mass':'msunh'}

    def map(self, mapunit):

        if len(mapunit['mass'].shape)>1:
            self.ndefs = mapunit['mass'].shape[1]
        else:
            self.ndefs = 1
            mapunit['mass'] = np.atleast_2d(mapunit['mass']).T

        self.nbands = self.ndefs

        if not hasattr(self, 'occmass'):
            self.occ   = np.zeros((self.nmassbins,self.ndefs,self.nzbins))
            self.occsq = np.zeros((self.nmassbins,self.ndefs,self.nzbins))
            self.count = np.zeros((self.nmassbins,self.ndefs,self.nzbins))

        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])
            for j in range(self.ndefs):
                mb = np.digitize(mapunit['mass'][zlidx:zhidx,j], bins=self.massbins)

                for k, m in enumerate(self.massbins[:-1]):
                    o  = mapunit['occ'][zlidx:zhidx][mb==k]
                    self.occ[k,j,i]   += np.sum(o)
                    self.occsq[k,j,i] += np.sum(o**2)
                    self.count[k,j,i] += np.sum(mb==k)

    def reduce(self):

        self.occmass = self.occ/self.count
        self.occvar  = (self.count*self.occsq - self.occ**2)/(self.count*(self.count-1))

        self.y = self.occmass
        self.ye = np.sqrt(self.occvar)


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
                 catalog_type=['halocatalog'], tag=None):

        if massbins is None:
            massbins = np.logspace(10, 16, 40)

        MassMetric.__init__(self, ministry, zbins=zbins, massbins=massbins,
                            catalog_type=catalog_type, tag=tag)

        self.aschema = 'haloonly'

        if lightcone:
            self.mapkeys   = ['mass', 'redshift']
            self.lightcone = True
        else:
            self.mapkeys   = ['mass']
            self.lightcone = False

        self.unitmap = {'mass':'msunh'}
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
               #,"As":2.1e-9,
               #"As_pivot":0.05
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

    def reduce(self):
        self.reduce()

    def visualize(self, plotname=None, usecols=None, usez=None,fracdev=False,
                  ref_y=None, ref_x=[None], xlim=None, ylim=None, fylim=None,
                  f=None, ax=None, xlabel=None,ylabel=None,compare=False,**kwargs):

        if xlabel is None:
            xlabel = r"$M_{halo} \, [M_{\odot}\, h^{-1}]$"
        if ylabel is None:
            ylabel = r"$N \, [Mpc^{-3}\, h^{3}]$"

        return MassMetric.visualize(self, plotname=plotname, usecols=usecols, usez=usez,
                             fracdev=fracdev, ref_y=ref_y, ref_x=ref_x, xlim=xlim,
                             ylim=ylim, fylim=fylim, f=f, ax=ax, xlabel=xlabel,
                             ylabel=ylabel, compare=compare,logx=True,**kwargs)
