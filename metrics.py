from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import numpy as np

class Metric(object):

    __metaclass__ = ABCMeta

    def __init__(self, simulation):
        self.sim = simulation

    @abstractmethod
    def map(self, mapunit):
        pass

    @abstractmethod
    def reduce(self):
        pass

class LuminosityFunction(Metric):
    
    def __init__(self, simulation, zbins=None, lumbins=None):
        Metric.__init__(self, simulation)

        if zbins==None:
            self.zbins = [0.0, 0.2]
        else:
            self.zbins = zbins

        if lumbins==None:
            self.lumbins = np.linspace(10, 30, 60)
        else:
            self.lumbins = lumbins
        
    def map(self, mapunit):
        """
        A simple example of what a map function should look like.
        """

        nbands = mapunit['luminosity'].shape[1]

        if not hasattr(self, 'lumcounts'):
            self.lumcounts = np.zeros((len(self.lumbins)-1, nbands, len(self.zbins)-1))
            
        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

            for j in range(nbands):
                c, e = np.histogram(mapunit['luminosity'][zlidx:zhidx,j], bins=self.lumbins)
                self.lumcounts[:,j,i] += c

    def reduce(self):
        """
        Given counts in luminosity bins, generate a luminosity function
        """
        
        self.luminosity_function = self.lumcounts/self.sim.volume

class MagCounts(Metric):
    """
    Compute count per magnitude in redshift bins
    """

    def __init__(self, simulation, zbins=None, magbins=None):
        Metric.__init__(self,simulation)

        if zbins==None:
            self.zbins = [0.0, 0.2]
        else:
            self.zbins = zbins

        if magbins==None:
            self.magbins = np.linspace(10, 30, 60)
        else:
            self.magbins = magbins

    def map(self, mapunit):
        nbands = mapunit['appmag'].shape[1]

        if not hasattr(self, 'magcounts'):
            self.magcounts = np.zeros((len(self.magbins)-1, nbands, len(self.zbins)-1))
            
        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])
            for j in range(nbands):
                c, e = np.histogram(mapunit['appmag'][zlidx:zhidx,j], bins=self.magbins)
                self.magcounts[:,j,i] += c


    def reduce(self):
        self.magcounts = self.magcounts/self.sim.volume

class ColorColor(Metric):
    
    def __init__(self, simulation, zbins=None, magbins=None):
        Metric.__init__(self, simulation)
        
        if zbins==None:
            self.zbins = [0.0, 0.2]
        else:
            self.zbins = zbins

        if magbins==None:
            self.magbins = np.linspace(10, 30, 60)
        else:
            self.magbins = magbins

    def map(self, mapunit):
        nbands = mapunit['appmag'].shape[1]

        if not hasattr(self, 'cc'):
            self.cc = np.zeros((len(self.magbins)-1, len(self.magbins)-1, 
                                           nbands*(nbands-1)/2, len(self.zbins)-1))
            
        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])
            for j in range(nbands):
                for k in range(nbands):
                    if k<=j: continue
                    ind = k*(k-1)/2+j-1
                    c, e0, e1 = np.histogram2d(mapunit['appmag'][zlidx:zhidx,j], 
                                               mapunit['appmag'][zlidx:zhidx,k], 
                                               bins=[self.magbins,self.magbins])
                    self.cc[:,:,ind,i] += c

    
    def reduce(self):
        self.cc = self.cc/self.sim.volume
