from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
if __name__=='__main__':
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pylab as plt
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

    @abstractmethod
    def visualize(self, plotname=None):
        pass

    @abstractmethod
    def compare(self, othermetric, plotname=None):
        pass

class LuminosityFunction(Metric):
    
    def __init__(self, simulation, zbins=None, lumbins=None):
        Metric.__init__(self, simulation)

        if zbins==None:
            self.zbins = [0.0, 0.2]
        else:
            self.zbins = zbins

        if lumbins==None:
            self.lumbins = np.linspace(-25, -11, 30)
        else:
            self.lumbins = lumbins
        
    def map(self, mapunit):
        """
        A simple example of what a map function should look like.
        """

        self.nbands = mapunit['luminosity'].shape[1]

        if not hasattr(self, 'lumcounts'):
            self.lumcounts = np.zeros((len(self.lumbins)-1, self.nbands, 
                                       len(self.zbins)-1))
            
        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

            for j in range(self.nbands):
                c, e = np.histogram(mapunit['luminosity'][zlidx:zhidx,j], 
                                    bins=self.lumbins)
                self.lumcounts[:,j,i] += c

    def reduce(self):
        """
        Given counts in luminosity bins, generate a luminosity function
        """
        self.luminosity_function = self.lumcounts/self.sim.volume

    def visualize(self, plotname=None, usebands=None, f=None, ax=None, marker='d'):
        """
        Plot the calculated luminosity function.
        """
        mlums = np.array([(self.lumbins[i]+self.lumbins[i+1])/2 
                          for i in range(len(self.lumbins)-1)])

        if usebands==None:
            usebands = range(self.nbands)

        if f==None:
            f, ax = plt.subplots(len(usebands), len(self.zbins)-1,
                                 sharex=True, sharey=True)
            newaxes = True
        else:
            newaxes = False


        for i, b in enumerate(usebands):
            for j in range(len(self.zbins)-1):
                ax[j*self.nbands+i].semilogy(mlums, self.luminosity_function[:,b,j], marker)
        
        if newaxes:
            sax = f.add_subplot(111)
            sax.spines['top'].set_color('none')
            sax.spines['bottom'].set_color('none')
            sax.spines['left'].set_color('none')
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            sax.set_xlabel(r'Luminosity')
            sax.set_ylabel(r'$\phi\, [Mpc^{-3}h^{3}]$')

        if plotname!=None:
            plt.savefig(plotname)

        return f, ax
        

    def compare(self, othermetric, plotname=None, usebands=None):
        
        if usebands!=None:
            assert(len(usebands[0])==len(usebands[1]))
            f, ax = self.visualize(usebands=usebands[0])
            f, ax = othermetric.visualize(usebands=usebands[1],
                                          f=f, ax=ax, marker='s')
        else:
            f, ax = self.visualize(usebands=usebands)
            f, ax = othermetric.visualize(usebands=usebands,
                                          f=f, ax=ax, marker='s')

        if plotname!=None:
            plt.savefig(plotname)
    


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
        self.nbands = mapunit['appmag'].shape[1]

        if not hasattr(self, 'magcounts'):
            self.magcounts = np.zeros((len(self.magbins)-1, 
                                       self.nbands, len(self.zbins)-1))
            
        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])
            for j in range(self.nbands):
                c, e = np.histogram(mapunit['appmag'][zlidx:zhidx,j], 
                                    bins=self.magbins)
                self.magcounts[:,j,i] += c

    def reduce(self):
        self.magcounts = self.magcounts/self.sim.area

    def visualize(self, plotname=None, f=None, ax=None, usebands=None, marker='d'):
        mmags = np.array([(self.magbins[i]+self.magbins[i+1])/2 
                          for i in range(len(self.magbins)-1)])

        if usebands==None:
            usebands = range(self.nbands)

        if f==None:
            f, ax = plt.subplots(len(usebands), len(self.zbins)-1,
                                 sharex=True, sharey=True)
            newaxes = True
        else:
            newaxes = False

        for i, b in enumerate(usebands):
            for j in range(len(self.zbins)-1):
                ax[j*self.nbands+i].semilogy(mmags, self.magcounts[:,b,j], marker)

        if plotname!=None:
            plt.savefig(plotname)

        return f, ax


    def compare(self, othermetric, plotname=None, usebands=None):
        if usebands!=None:
            print('0')
            assert(len(usebands[0])==len(usebands[1]))
            print('1')
            f, ax = self.visualize(usebands=usebands[0])
            print('2')
            f, ax = othermetric.visualize(usebands=usebands[1],
                                          f=f, ax=ax, marker='s')
        else:
            f, ax = self.visualize(usebands=usebands)
            f, ax = othermetric.visualize(usebands=usebands,
                                          f=f, ax=ax, marker='s')

        if plotname!=None:
            plt.savefig(plotname)


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
        self.nbands = mapunit['appmag'].shape[1]

        if not hasattr(self, 'cc'):
            self.cc = np.zeros((len(self.magbins)-1, len(self.magbins)-1, 
                                self.nbands*(self.nbands-1)/2,
                                len(self.zbins)-1))
            
        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])
            for j in range(self.nbands):
                for k in range(self.nbands):
                    if k<=j: continue
                    ind = k*(k-1)/2+j-1
                    c, e0, e1 = np.histogram2d(mapunit['appmag'][zlidx:zhidx,j], 
                                               mapunit['appmag'][zlidx:zhidx,k], 
                                               bins=[self.magbins,self.magbins])
                    self.cc[:,:,ind,i] += c

    
    def reduce(self):
        self.cc = self.cc/self.sim.area

    def visualize(self, plotname=None, f=None, ax=None, usecolors=None):
        mmags = np.array([(self.magbins[i]+self.magbins[i+1])/2 
                          for i in range(len(self.magbins)-1)])

        if usecolors==None:
            usecolors = range(self.cc.shape[2])

        if f==None:
            f, ax = plt.subplots(len(self.zbins)-1, len(usecolors),
                                 sharex=True, sharey=True)
            newaxes = True
        else:
            newaxes = False

        for i in usecolors:
            for j in range(len(self.zbins)-1):
                ax[j*self.nbands+i].pcolormesh(mmags, mmags, self.cc[:,:,i,j])

        return f, ax

    def compare(self, othermetric, plotname=None, usecolors=None):
        if usebands!=None:
            assert(len(usecolors[0])==len(usecolors[1]))
            f, ax = self.visualize(usecolors=usecolors[0])
            f, ax = othermetric.visualize(usecolors=usecolors[1],
                                          f=f, ax=ax)
        else:
            f, ax = self.visualize(usecolors=usecolors)
            f, ax = othermetric.visualize(usecolors=usecolors,
                                          f=f, ax=ax)

        if plotname!=None:
            plt.savefig(plotname)

