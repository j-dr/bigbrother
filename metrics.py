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

        self.mapkeys = ['luminosity', 'redshift']
        
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
        area = self.sim.galaxycatalog.getArea()
        self.luminosity_function = self.lumcounts/self.sim.calculate_volume(area)

    def visualize(self, plotname=None, usebands=None, f=None, ax=None, **kwargs):
        """
        Plot the calculated luminosity function.
        """
        mlums = np.array([(self.lumbins[i]+self.lumbins[i+1])/2 
                          for i in range(len(self.lumbins)-1)])

        if usebands==None:
            usebands = range(self.nbands)

        if f==None:
            f, ax = plt.subplots(len(usebands), len(self.zbins)-1,
                                 sharex=True, sharey=True, figsize=(8,8))
            newaxes = True
        else:
            newaxes = False


        if len(zbins)-1>1:
            for i, b in enumerate(usebands):
                for j in range(len(self.zbins)-1):
                    ax[i][j].semilogy(mlums, self.luminosity_function[:,b,j], 
                                      **kwargs)
        else:
            for i, b in enumerate(usebands):
                ax[i].semilogy(mlums, self.luminosity_function[:,b,j], 
                               **kwargs)
            
        
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
        

    def compare(self, othermetrics, plotname=None, usebands=None, **kwargs):
        tocompare = [self]
        tocompare.extend(othermetrics)

        if usebands!=None:
            if not hasattr(usebands[0], '__iter__'):
                usebands = [usebands]*len(tocompare)
            else:
                assert(len(usebands)==len(tocompare))
        else:
            usebands = [None]*len(tocompare)
        
        for i, m in enumerate(tocompare):
            if usebands[i]!=None:
                assert(len(usebands[0])==len(usebands[i]))
            if i==0:
                f, ax = m.visualize(usebands=usebands[i], **kwargs)
            else:
                f, ax = m.visualize(usebands=usebands[i],
                                    f=f, ax=ax, **kwargs)

        if plotname!=None:
            plt.savefig(plotname)

        return f, ax
    

class MagCounts(Metric):
    """
    Compute count per magnitude in redshift bins
    """

    def __init__(self, simulation, zbins=[0.0, 0.2],  magbins=None):
        Metric.__init__(self,simulation)

        self.zbins = zbins
        if zbins==None:
            self.nzbins = 1
        else:
            self.nzbins = len(zbins)-1

        if magbins==None:
            self.magbins = np.linspace(10, 30, 60)
        else:
            self.magbins = magbins

        self.mapkeys = ['appmag', 'redshift']

    def map(self, mapunit):
        self.nbands = mapunit['appmag'].shape[1]

        if not hasattr(self, 'magcounts'):
            self.magcounts = np.zeros((len(self.magbins)-1, 
                                       self.nbands, self.nzbins))
            
        if self.zbins!=None:
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
        area = self.sim.galaxycatalog.getArea()
        self.magcounts = self.magcounts/area

    def visualize(self, plotname=None, f=None, ax=None, usebands=None, **kwargs):
        mmags = np.array([(self.magbins[i]+self.magbins[i+1])/2 
                          for i in range(len(self.magbins)-1)])

        if usebands==None:
            usebands = range(self.nbands)

        if f==None:
            f, ax = plt.subplots(len(usebands), self.nzbins,
                                 sharex=True, sharey=True,
                                 figsize=(8,8))
            newaxes = True
        else:
            newaxes = False

        if self.nzbins>1:
            for i, b in enumerate(usebands):
                for j in range(self.nzbins):
                    ax[i][j].semilogy(mmags, self.magcounts[:,b,j], 
                                      **kwargs)
        else:
            for i, b in enumerate(usebands):
                for j in range(self.nzbins):
                    ax[i].semilogy(mmags, self.magcounts[:,b,j], 
                                   **kwargs)


        if newaxes:
            sax = f.add_subplot(111)
            sax.spines['top'].set_color('none')
            sax.spines['bottom'].set_color('none')
            sax.spines['left'].set_color('none')
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            sax.set_xlabel(r'$m\, [Mags]$')
            sax.set_ylabel(r'$n\, [deg^{-2}]$')
 


        if plotname!=None:
            plt.savefig(plotname)

        return f, ax


    def compare(self, othermetrics, plotname=None, usebands=None, **kwargs):
        tocompare = [self]
        tocompare.extend(othermetrics)

        if usebands!=None:
            if not hasattr(usebands[0], '__iter__'):
                usebands = [usebands]*len(tocompare)
            else:
                assert(len(usebands)==len(tocompare))
        else:
            usebands = [None]*len(tocompare)
        
        for i, m in enumerate(tocompare):
            if usebands[i]!=None:
                assert(len(usebands[0])==len(usebands[i]))
            if i==0:
                f, ax = m.visualize(usebands=usebands[i], **kwargs)
            else:
                f, ax = m.visualize(usebands=usebands[i],
                                    f=f, ax=ax, **kwargs)

        if plotname!=None:
            plt.savefig(plotname)

        return f, ax


class ColorColor(Metric):
    
    def __init__(self, simulation, zbins=[0.0, 0.2], magbins=None):
        Metric.__init__(self, simulation)
        
        self.zbins = zbins
        if zbins==None:
            self.nzbins = 1
        else:
            self.nzbins = len(zbins)-1

        if magbins==None:
            self.magbins = np.linspace(10, 30, 60)
        else:
            self.magbins = magbins

        self.mapkeys = ['appmag', 'redshift']

    def map(self, mapunit):
        self.nbands = mapunit['appmag'].shape[1]

        if not hasattr(self, 'cc'):
            self.cc = np.zeros((len(self.magbins)-1, len(self.magbins)-1, 
                                self.nbands*(self.nbands-1)/2,
                                self.nzbins))

        if self.zbins!=None:
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
        else:
            for j in range(self.nbands):
                for k in range(self.nbands):
                    if k<=j: continue
                    ind = k*(k-1)/2+j-1
                    c, e0, e1 = np.histogram2d(mapunit['appmag'][:,j], 
                                               mapunit['appmag'][:,k], 
                                               bins=[self.magbins,self.magbins])
                    self.cc[:,:,ind,0] += c


    
    def reduce(self):
        area = self.sim.galaxycatalog.getArea()
        self.cc = self.cc/area

    def visualize(self, plotname=None, f=None, ax=None, usecolors=None, **kwargs):
        mmags = np.array([(self.magbins[i]+self.magbins[i+1])/2 
                          for i in range(len(self.magbins)-1)])

        if usecolors==None:
            usecolors = range(self.cc.shape[2])

        if f==None:
            f, ax = plt.subplots(self.nzbins, len(usecolors),
                                 sharex=True, sharey=True, figsize=(8,8))
            newaxes = True
        else:
            newaxes = False

        for i in usecolors:
            for j in range(self.nzbins):
                ax[j*self.nbands+i].pcolormesh(mmags, mmags, self.cc[:,:,i,j],
                                               **kwargs)

        return f, ax

    def compare(self, othermetric, plotname=None, usecolors=None, **kwargs):
        if usebands!=None:
            assert(len(usecolors[0])==len(usecolors[1]))
            f, ax = self.visualize(usecolors=usecolors[0], **kwargs)
            f, ax = othermetric.visualize(usecolors=usecolors[1],
                                          f=f, ax=ax, **kwargs)
        else:
            f, ax = self.visualize(usecolors=usecolors, **kwargs)
            f, ax = othermetric.visualize(usecolors=usecolors,
                                          f=f, ax=ax, **kwargs)

        if plotname!=None:
            plt.savefig(plotname)

