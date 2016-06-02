from __future__ import print_function, division
from .metric import Metric, GMetric
if __name__=='__main__':
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pylab as plt
import numpy as np


class MassMetric(GMetric):
    """
    Restruct GMetric to magnitudes
    """
    
    def __init__(self, ministry, zbins=None, massbins=None,
                 catalog_type=None):
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
            massbins = np.linspace(10, 16, 40)

        if zbins is None:
            zbins = np.array([0.0, 10.0])

        self.massbins  = massbins
        self.zbins     = zbins
        self.nmassbins = len(self.massbins)-1
        self.nzbins    = len(self.zbins)-1


        GMetric.__init__(self, ministry, zbins=zbins, xbins=massbins,
                         catalog_type=catalog_type)

class MassFunction(MassMetric):

    def __init__(self, ministry, zbins=None, massbins=None, lightcone=True,
                 catalog_type=['halocatalog']):

        if massbins is None:
            massbins = np.linspace(10, 16, 40)

        MassMetric.__init__(self, ministry, zbins=zbins, massbins=massbins,
                            catalog_type=catalog_type)

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
            mapunit['mass'] = np.atleast_2d(mapunit['mass'])

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
            for j in range(self.nbands):
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


class SimpleHOD(MassMetric):

    def __init__(self, ministry, zbins=None, massbins=None, lightcone=True,
                 catalog_type=['halocatalog']):

        if massbins is None:
            massbins = np.linspace(10, 16, 40)

        MassMetric.__init__(self, ministry, zbins=zbins, massbins=massbins,
                            catalog_type=catalog_type)

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
        self.ndefs = mapunit['mass'].shape[1]

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
                    mb = np.digitize(mu['mass'][zlidx:zhidx,j], bins=self.massbins)-1
                    for k in range(len(self.massbins)-1):
                        self.occcounts[k,j,i] += np.sum(mu['occ'][zlidx:zhidz][mb==k])
                        self.sqocccounts[k,j,i] += np.sum(mu['occ'][zlidx:zhidz][mb==k]**2)
                        self.halocounts[k,j,i] += len(mu['occ'][mb==k])
        else:
            for j in range(self.ndefs):
                mb = np.digitize(mu['mass'][:,j], bins=self.massbins)-1
                for k in range(len(self.massbins)-1):
                    self.occcounts[k,j,i] += np.sum(mu['occ'][mb==k])
                    self.sqocccounts[k,j,i] += np.sum(mu['occ'][mb==k]**2)
                    self.halocounts[k,j,i] += len(mu['occ'][mb==k])


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


class OccMass(MassMetric):

    def __init__(self, ministry, zbins=None, massbins=None, lightcone=True,
                 catalog_type=['halocatalog']):

        if massbins is None:
            massbins = np.linspace(10, 16, 40)

        MassMetric.__init__(self, ministry, zbins=zbins, massbins=massbins,
                            catalog_type=catalog_type)

        if lightcone:
            self.mapkeys   = ['mass', 'occ', 'redshift']
            self.lightcone = True
        else:
            self.mapkeys   = ['mass', 'occ']
            self.lightcone = False

        self.aschema = 'haloonly'
        self.unitmap = {'mass':'msunh'}

    def map(self, mapunit):

        if not hasattr(self, 'occmass'):
            self.occ   = np.zeros((self.nmassbins,self.nzbins))
            self.occsq = np.zeros((self.nmassbins,self.nzbins))
            self.count = np.zeros((self.nmassbins,self.nzbins))

        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])
            
            mb = np.digitize(mapunit['mass'][zlidx:zhidx], bins=self.massbins)
            for j, m in enumerate(self.massbins[:-1]):
                o  = mapunit['occ'][zlidx:zhidx][mb==j]
                self.occ[j,i]   += np.sum(o)
                self.occsq[j,i] += np.sum(o**2)
                self.count[j,i] += np.sum(mb==j)
                
    def reduce(self):

        self.occmass = self.occ/self.count
        self.occstd  = (self.count*self.occsq - self.occ**2)/(self.count*(self.count-1))

    def visualize(self, plotname=None, f=None, ax=None, **kwargs):

        if f is None:
            f, ax = plt.subplots(self.nzbins, sharex=True, sharey=True,
                                 figsize=(8,8))
            newaxes = True
        else:
            newaxes = False

        self.mmean = (self.massbins[:-1] + self.massbins[1:])/2

        for i in range(self.nzbins):
            ax[i].errorbar(self.mmean, self.occmass[:,i], self.occstd[:,i],
                           **kwargs)

        if newaxes:
            sax = f.add_subplot(111)
            sax.spines['top'].set_color('none')
            sax.spines['bottom'].set_color('none')
            sax.spines['left'].set_color('none')
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            sax.set_xlabel(r'$M_{halo}\, [M_{\odot} \, h^{-1}]$')
            sax.set_ylabel(r'$N$')

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax

        

