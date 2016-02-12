from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
from scipy.interpolate import InterpolatedUnivariateSpline
from copy import copy
if __name__=='__main__':
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.gridspec as gridspec

import matplotlib.pylab as plt
import numpy as np
import treecorr


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
    
    def __init__(self, simulation, central_only=False, zbins=None, lumbins=None):
        Metric.__init__(self, simulation)

        if zbins==None:
            self.zbins = [0.0, 0.2]
        else:
            self.zbins = zbins
            self.zbins = np.array(self.zbins)

        self.nzbins = len(self.zbins)-1

        if lumbins==None:
            self.lumbins = np.linspace(-25, -11, 30)
        else:
            self.lumbins = lumbins

        self.central_only = central_only
        if central_only:
            self.mapkeys = ['luminosity', 'redshift', 'central']
        else:
            self.mapkeys = ['luminosity', 'redshift']
        
    def map(self, mapunit):
        """
        A simple example of what a map function should look like.
        """

        self.nbands = mapunit['luminosity'].shape[1]
        mu = {}

        if self.central_only:
            for k in mapunit.keys():
                mu[k] = mapunit[k][mapunit['central']==1]
        else:
            mu = mapunit

        if not hasattr(self, 'lumcounts'):
            self.lumcounts = np.zeros((len(self.lumbins)-1, self.nbands, 
                                       len(self.zbins)-1))
            
        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mu['redshift'].searchsorted(self.zbins[i])
            zhidx = mu['redshift'].searchsorted(self.zbins[i+1])

            for j in range(self.nbands):
                c, e = np.histogram(mu['luminosity'][zlidx:zhidx,j], 
                                    bins=self.lumbins)
                self.lumcounts[:,j,i] += c

    def reduce(self):
        """
        Given counts in luminosity bins, generate a luminosity function
        """
        area = self.sim.galaxycatalog.getArea()
        self.luminosity_function = self.lumcounts

        for i in range(self.nzbins):
            vol = self.sim.calculate_volume(area, self.zbins[i], self.zbins[i+1])
            self.luminosity_function[:,:,i] /= vol

    def visualize(self, plotname=None, usebands=None, fracdev=True, ref_lf=None, 
                  ref_ml=None, xlim=None, ylim=None, fylim=None, f=None, ax=None,
                  **kwargs):
        """
        Plot the calculated luminosity function.
        """
        if hasattr(self, 'lummean'):
            mlums = self.lummean
        else:
            mlums = np.array([(self.lumbins[i]+self.lumbins[i+1])/2 
                              for i in range(len(self.lumbins)-1)])

        if usebands==None:
            usebands = range(self.nbands)
        
        if fracdev & (len(ref_ml)!=len(mlums)):
            rls = ref_lf.shape
            li = mlums.searchsorted(ref_ml[0])
            hi = mlums.searchsorted(ref_ml[-1])
            iref_lf = np.zeros((hi-li, rls[1], rls[2]))
            for i in range(rls[1]):
                for j in range(rls[2]):
                    spl = InterpolatedUnivariateSpline(ref_ml, ref_lf[:,i,j])
                    iref_lf[:,i,j] = spl(mlums[li:hi])

            ref_lf = iref_lf
        else:
            li = 0
            hi = len(mlums)
                    

        if f==None:
            if fracdev==False:
                f, ax = plt.subplots(len(usebands), len(self.zbins)-1,
                                     sharex=True, sharey=True, figsize=(8,8))
            else:
                assert(ref_lf!=None)
                gs = gridspec.GridSpec(len(usebands)*2, self.nzbins)
                f = plt.figure()
                ax = []
                for r in range(len(usebands)):
                    ax.append([])
                    ax.append([])
                    for c in range(self.nzbins):
                        if (r==0) & (c==0):
                            ax[2*r].append(f.add_subplot(gs[2*r,c]))
                            ax[2*r+1].append(f.add_subplot(gs[2*r+1,c], sharex=ax[0][0]))
                        else:
                            ax[2*r].append(f.add_subplot(gs[2*r,c]))
                            ax[2*r+1].append(f.add_subplot(gs[2*r+1,c], sharex=ax[0][0], 
                                                           sharey=ax[1][0]))
            newaxes = True
        else:
            newaxes = False

        if self.nzbins>1:
            for i, b in enumerate(usebands):
                for j in range(len(self.zbins)-1):
                    if fracdev==False:
                        ax[i][j].semilogy(mlums, self.luminosity_function[:,b,j], 
                                          **kwargs)
                    else:
                        ax[2*i][j].semilogy(mlums, self.luminosity_function[:,b,j], 
                                          **kwargs)
                        ax[2*i+1][j].plot(mlums, 
                                          (self.luminosity_function[li:hi,b,j]-ref_lf[:,b,j])\
                                              /ref_lf[:,b,j], **kwargs)
                        if (i==0) & (j==0):
                            if xlim!=None:
                                ax[0][0].set_xlim(xlim)
                            if ylim!=None:
                                ax[0][0].set_ylim(ylim)
                            if fylim!=None:
                                ax[1][0].set_ylim(fylim)

        else:
            for i, b in enumerate(usebands):
                if fracdev==False:
                    ax[i].semilogy(mlums, self.luminosity_function[:,b,0], 
                                   **kwargs)
                else:
                    ax[0][i].semilogy(mlums, self.luminosity_function[:,b,0], 
                                        **kwargs)
                    ax[1][i].plot(mlums, (self.luminosity_function[li:hi,b,0]-ref_lf[:,b,0])\
                                      /ref_lf[:,b,0], **kwargs)
        
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
        

    def compare(self, othermetrics, plotname=None, usebands=None, fracdev=True, xlim=None,
                ylim=None, fylim=None, labels=None, **kwargs):
        tocompare = [self]
        tocompare.extend(othermetrics)

        if usebands!=None:
            if not hasattr(usebands[0], '__iter__'):
                usebands = [usebands]*len(tocompare)
            else:
                assert(len(usebands)==len(tocompare))
        else:
            usebands = [None]*len(tocompare)

        if fracdev:
            if hasattr(self, 'lummean'):
                ref_ml = self.lummean
            else:
                ref_ml =  np.array([(self.lumbins[i]+self.lumbins[i+1])/2 
                                    for i in range(len(self.lumbins)-1)])

        if labels==None:
            labels = [None]*len(tocompare)

        for i, m in enumerate(tocompare):
            if usebands[i]!=None:
                assert(len(usebands[0])==len(usebands[i]))
            if i==0:
                if fracdev:
                    f, ax = m.visualize(usebands=usebands[i], fracdev=True, ref_ml=ref_ml,
                                        ref_lf=self.luminosity_function, xlim=xlim,
                                        ylim=ylim, fylim=fylim, label=labels[i],**kwargs)
                else:
                    f, ax = m.visualize(usebands=usebands[i], xlim=xlim, ylim=ylim, 
                                        fracdev=False, fylim=fylim,label=labels[i],**kwargs)
            else:
                if fracdev:
                    f, ax = m.visualize(usebands=usebands[i], fracdev=True, ref_ml=ref_ml,
                                        ref_lf=tocompare[0].luminosity_function, 
                                        xlim=xlim, ylim=ylim, fylim=fylim,
                                        f=f, ax=ax, label=labels[i], **kwargs)
                else:
                    f, ax = m.visualize(usebands=usebands[i], xlim=xlim, ylim=ylim,
                                        fylim=fylim, f=f, ax=ax, fracdev=False,
                                        label=labels[i], **kwargs)
        if plotname!=None:
            plt.savefig(plotname)

        return f, ax


class LcenMvir(Metric):

    def __init__(self, simulation, zbins=None, massbins=None):
        Metric.__init__(self, simulation)

        if zbins==None:
            self.zbins = [0.0, 0.2]
        else:
            self.zbins = zbins
            self.zbins = np.array(self.zbins)

        self.nzbins = len(self.zbins)-1

        if massbins==None:
            self.massbins = np.logspace(12, 15, 20)
        else:
            self.massbins = massbins

        self.mapkeys = ['luminosity', 'redshift', 'central', 'mvir']

        
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
            mb = np.digitize(mu['mvir'][zlidx:zhidx], bins=self.massbins)

            for j in range(len(self.massbins)-1):
                blum = mu['luminosity'][zlidx:zhidx,:][mb==j]
                self.bincount[j,:,i] += len(blum)
                self.totlum[j,:,i] += np.sum(blum, axis=0)


    def reduce(self):

        self.lcen_mvir = self.totlum/self.bincount


    def visualize(self, plotname=None, f=None, ax=None, usebands=None, **kwargs):

        if hasattr(self, 'massmean'):
            mmass = self.massmean
        else:
            mmass = np.array([(self.massbins[i]+self.massbins[i+1])/2 
                              for i in range(len(self.massbins)-1)])

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
                    ax[i][j].semilogx(mmass, self.lcen_mvir[:,b,j], 
                                      **kwargs)
        else:
            for i, b in enumerate(usebands):
                for j in range(self.nzbins):
                    ax[i].semilogx(mmass, self.lcen_mvir[:,b,j], 
                                   **kwargs)

        if newaxes:
            sax = f.add_subplot(111)
            sax.spines['top'].set_color('none')
            sax.spines['bottom'].set_color('none')
            sax.spines['left'].set_color('none')
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            sax.set_xlabel(r'$M_{vir}\, [M_{sun}]$')
            sax.set_ylabel(r'$L_{cen}\, [mag]$')

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
            self.zbins = np.array(self.zbins)

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

        if hasattr(self, 'magmean'):
            mmags = self.magmean
        else:
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
            self.zbins = np.array(self.zbins)

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

        if hasattr(self, 'magmean'):
            mmags = self.magmean
        else:
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
                ax[j][i].pcolormesh(mmags, mmags, self.cc[:,:,i,j],
                                    **kwargs)

        return f, ax

    def compare(self, othermetric, plotname=None, usecolors=None, **kwargs):
        if usecolors!=None:
            assert(len(usecolors[0])==len(usecolors[1]))
            f, ax = self.visualize(usecolors=usecolors[0], **kwargs)
            f, ax = othermetric.visualize(usecolors=usecolors[1],
                                          f=f, ax=ax, **kwargs)
        else:
            f, ax = self.visualize(usecolors=usecolors, **kwargs)
            f, ax = othermetric[0].visualize(usecolors=usecolors,
                                          f=f, ax=ax, **kwargs)

        if plotname!=None:
            plt.savefig(plotname)

class AnalyticLuminosityFunction(LuminosityFunction):

    def __init__(self, *args, **kwargs):
        
        if 'nbands' in kwargs:
            self.nbands = kwargs.pop('nbands')
        else:
            self.nbands = 5

        LuminosityFunction.__init__(self,*args,**kwargs)


    def genDSGParams(self, z, evol='faber', Q=-0.866):
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

        return params

    def genBCCParams(self, evol='faber', Q=-0.866):
        
        zmeans = ( self.zbins[1:] + self.zbins[:-1] ) / 2
        par = np.zeros((len(zmeans), 8))

        for i, z in enumerate(zmeans):
            par[i,:] = self.genDSGParams(z, evol=evol)

        return par

    def evolveDSGParams(self, p, Q, evol='faber'):

        zmeans = ( self.zbins[1:] + self.zbins[:-1] ) / 2

        par = np.zeros((len(zmeans), 8))

        for i, z in enumerate(zmeans):
            par[i,:] = copy(p)
            
            if evol=='faber':
                par[i,4] += Q * (np.log10(z) + 1)
                par[i,6] += Q * (np.log10(z) + 1)
            else:
                par[i,4] += Q * (1. / (1 + z) - 1. / 1.1)
                par[i,6] += Q * (1. / (1 + z) - 1. / 1.1)

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
        self.lummean = (self.lumbins[1:]+self.lumbins[:-1])/2
        self.luminosity_function = np.zeros((len(self.lummean), self.nbands,
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
                    self.luminosity_function[:,j,i] = self.schechterFunctionAmag(self.lummean, p)
                elif form=='doubleSchecterFunctionAmag':
                    self.luminosity_function[:,j,i] = self.doubleSchechterFunctionAmag(self.lummean, p)
                elif form=='doubleSchechterGaussian':
                    self.luminosity_function[:,j,i] = self.doubleSchechterGaussian(self.lummean, p)

                elif hasattr(form, '__call__'):
                    try:
                        self.luminosity_function[:,j,i] = form(self.lummean, p)
                    except:
                        raise(TypeError("Functional form is not in the correct format"))

                else:
                    raise(NotImplementedError("Luminosity form {0} is not implemented".format(form)))

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
        Single Schechter function appropriate for absolute magnitudes.
        
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

    def doubleSchechterGaussian(self,m,p):

        phi = 0.4 * np.log(10) * np.exp(-10**(-0.4 * (m - p[4]))) * \
            (p[0] * 10 ** (-0.4 * (m - p[4])*(p[1]+1)) + \
            p[2] * 10 ** (-0.4 * (m - p[4])*(p[3]+1))) + \
            p[5] / np.sqrt(2 * np.pi * p[7] ** 2) * \
            np.exp(-(m - p[6]) ** 2 / (2 * p[7] ** 2))

        return phi

class TabulatedLuminosityFunction(LuminosityFunction):

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
        

        if len(self.fname)==1:
            tab = np.loadtxt(self.fname[0])
            self.luminosity_function = np.zeros((tab.shape[0], self.nbands, self.nzbins))
            if len(tab.shape)==2:
                self.lummean = tab[:,0]
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
                self.lummean = tab[:,0,0]
                self.luminosity_function[:,:,:] = tab[:,1:,:]
        else:
            if len(self.fname.shape)==1:
                assert(self.fname.shape[0]==self.nzbins)
                for i in range(len(self.fname)):
                    lf = np.loadtxt(self.fname[i])
                    if i==0:
                        self.lummean = lf[:,0]
                        self.luminosity_function = np.zeros((len(self.lummean), self.nbands, self.nzbins))
                    else:
                        assert((lf[:,0]==self.lummean).all())
                    
                    for j in range(self.nbands):
                        self.luminosity_function[:,j,i] = lf[:,1]

            elif len(self.fname.shape)==2:
                for i in range(self.fname.shape[0]):
                    for j in range(self.fname.shape[1]):
                        lf = np.loadtxt(self.fname[i,j])
                        if (i==0) & (j==0):
                            self.lummean = lf[:,0]
                            self.luminosity_function = np.zeros((len(self.lummean), self.nbands, self.nzbins))
                        else:
                            assert(self.lummean==lf[:,0])
                        
                        self.luminosity_function[:,j,i] = lf[:,1]


class AngularCorrelationFunction(Metric):

    def __init__(self, simulation, zbins=None, lumbins=None):
        Metric.__init__(self, simulation)

        if zbins==None:
            self.zbins = [0.0, 0.2]
        else:
            self.zbins = zbins
            self.zbins = np.array(self.zbins)

        self.nzbins = len(self.zbins)-1

        if lumbins==None:
            self.lumbins = np.linspace(-25, -11, 30)
        else:
            self.lumbins = lumbins

        self.mapkeys = ['luminosity', 'redshift', 'polar_ang', 'azim_ang']
        
        

