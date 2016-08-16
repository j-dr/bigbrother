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

        return MassMetric.visualize(self, plotname=plotname, usecols=usecols, usez=usez,
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

class Richness(MassMetric):
    def __init__(self, ministry, zbins=None, massbins=None, lightcone=False,
                 catalog_type=['galaxycatalog'], tag=None, colorbins=None, 
                  maxrhalo=None, minlum=None, redsplit=None, splitinfo=False):

        if massbins is None:
            self.massbins = np.logspace(12, 15, 20)
        else:
            self.massbins = massbins

        if colorbins is None:
            self.colorbins = 100
        else:
            self.colorbins = colorbins

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

        self.split_info = splitinfo

        MassMetric.__init__(self, ministry, zbins=zbins, massbins=massbins,
                            catalog_type=catalog_type, tag=tag)

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
     
        self.galaxy_counts         = np.zeros((len(self.massbins) - 1, 1, len(self.zbins) - 1))
        self.galaxy_counts_squared = np.zeros((len(self.massbins) - 1, 1, len(self.zbins) - 1))
        self.halo_counts           = np.zeros((len(self.massbins) - 1, 1, len(self.zbins) - 1))
       


    def splitBimodal(self, x, y, largepoly=30):
        p = np.polyfit(x, y, largepoly) # polynomial coefficients for fit
        
        extrema = np.roots(np.polyder(p))
        extrema = extrema[np.isreal(extrema)]
        extrema = extrema[(extrema - x[1]) * (x[-2] - extrema) > 0] # exclude the endpoints due false maxima during fitting
        
        root_vals = [sum([p[::-1][i]*(root**i) for i in range(len(p))]) for root in extrema]
        peaks = extrema[np.argpartition(root_vals, -2)][-2:] # find two peaks of bimodal distribution

        mid = np.where((x - peaks[0])* (peaks[1] - x) > 0) # want data points between the peaks
        p_mid = np.polyfit(x[mid], y[mid], 2) # fit middle section to a parabola
        
        midpoint = np.roots(np.polyder(p_mid))[0]
 
        if (self.split_info): # debug info
            mpl.pyplot.plot(x,y)
            mpl.pyplot.plot(x[mid],[sum([p_mid[len(p_mid)-1-i]*(xval**i) for i in range(len(p_mid))]) for xval in x[mid]])
            mpl.pyplot.plot(x,[sum([p[len(p)-1-i]*(xval**i) for i in range(len(p))]) for xval in x])
            mpl.pyplot.plot(peaks, np.partition(root_vals, -2)[-2:], 'ro')
            mpl.pyplot.plot([midpoint], sum([p_mid[::-1][i]*midpoint**i for i in range(len(p_mid))]), 'ro')
            peakvals = np.partition(root_vals, -2)[-2:]
            print('Peaks: ' + str(peakvals) + ' at color ' + str(peaks))

        return midpoint


    def map(self, mapunit):
	
        print('min z: ' + str(min(mapunit['redshift'])))
	print('max z: ' + str(max(mapunit['redshift'])))
        # must convert mapunit dict to a recarray
        dtype = [(key, mapunit[key].dtype, np.shape(mapunit[key])[1:]) for key in mapunit.keys()]
        
        for ziter in range(len(self.zbins)-1):

            zcut = ((mapunit['redshift'] >= self.zbins[ziter]) & (mapunit['redshift'] < self.zbins[ziter+1]))
    
            g_r_color = mapunit['luminosity'][zcut][:,0] - mapunit['luminosity'][zcut][:,1] # get g-r color
            color_counts, color_bins = np.histogram(g_r_color, self.colorbins) # place colors into bins
    
            if self.splitcolor is None:
                self.splitcolor = self.splitBimodal(color_bins[:-1], color_counts)
    
            previd = -1
            halo_ids = np.unique(mapunit['haloid'][zcut])
            red_galaxy_counts = np.zeros(len(halo_ids)-1) # number of red galaxies in each unique halo        
    
            # cut of galaxies: within max_rhalo of parent halo, above min_lum magnitude, and red
            cut_array =((mapunit['rhalo'] < self.max_rhalo) & (mapunit['luminosity'][:,2] < self.min_lum) 
                & ((mapunit['luminosity'][:,0] - mapunit['luminosity'][:,1] >= self.splitcolor)) & ((mapunit['redshift'] >= self.zbins[ziter]) & (mapunit['redshift'] < self.zbins[ziter+1])))
            data_cut = np.recarray((len(cut_array[cut_array]), ), dtype)
            for key in mapunit.keys():
                data_cut[key] = mapunit[key][cut_array]
    
            data_cut.sort(order='haloid')
    
            idx = data_cut['haloid'][1:]-data_cut['haloid'][:-1]
            newhalos = np.where(idx != 0)[0]
            newhalos = np.hstack([[0], newhalos + 1, [len(data_cut) - 1]])
            
            uniquehalos = data_cut[newhalos[:-1]]
            red_counts = newhalos[1:]-newhalos[:-1]
            mass_bin_indices = np.digitize(uniquehalos['halomass'], self.massbins)
            
            self.halo_counts[:,0,ziter] += np.histogram(uniquehalos['halomass'], bins=self.massbins)[0]
    
            for i in range(len(self.massbins)-1):
                self.galaxy_counts[i,0,ziter]         += np.sum(red_counts[(mass_bin_indices == i+1)])
                self.galaxy_counts_squared[i,0,ziter] += np.sum(red_counts[(mass_bin_indices == i+1)]**2)
        
    
    def reduce(self):
        self.y           = self.galaxy_counts/self.halo_counts
        self.ye          = np.sqrt(self.galaxy_counts_squared / self.halo_counts - self.y**2)
        print(np.shape(self.y))

    def visualize(self, plotname=None, usecols=None, usez=None,fracdev=False,
                  ref_y=None, ref_x=[None], xlim=None, ylim=None, fylim=None,
                  f=None, ax=None, xlabel=None,ylabel=None,compare=False,**kwargs):

        if xlabel is None:
            xlabel = r"$M_{halo} \, [M_{\odot}\, h^{-1}]$"
        if ylabel is None:
            ylabel = r"$<N_{red}> \, [Mpc^{-3}\, h^{3}]$"

        return MassMetric.visualize(self, plotname=plotname, usecols=usecols, usez=usez,
                             fracdev=fracdev, ref_y=ref_y, ref_x=ref_x, xlim=xlim,
                             ylim=ylim, fylim=fylim, f=f, ax=ax, xlabel=xlabel,
                             ylabel=ylabel, compare=compare,logx=True,**kwargs)


