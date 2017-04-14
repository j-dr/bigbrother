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
                 cutband=None, upper_limit=False, **kwargs):

        if massbins is None:
            massbins = np.logspace(10, 16, 40)

        MassMetric.__init__(self, ministry, zbins=zbins, massbins=massbins,
                            catalog_type=catalog_type, tag=tag, **kwargs)
        self.nomap = False
        self.magcuts = magcuts
        self.upper_limit = upper_limit
        
        if self.magcuts is not None:
            self.usemag = True
            if self.upper_limit:
                self.nmagcuts = len(self.magcuts)
            else:
                self.nmagcuts = len(self.magcuts) - 1

        else:
            self.usemag = False
            self.nmagcuts = 1

        self.cutband = cutband

        self.aschema = 'galaxyonly'

        if lightcone:
            if self.usemag:
                self.mapkeys = ['halomass', 'central', 'redshift', 'haloid', 'rhalo', 'r200', 'luminosity']
            else:
                self.mapkeys = ['halomass', 'central', 'redshift', 'haloid', 'rhalo', 'r200']

            self.lightcone = True

        else:
            if self.usemag:
                self.mapkeys = ['halomass', 'central', 'haloid', 'rhalo', 'r200', 'luminosity']
            else:
                self.mapkeys = ['halomass', 'central', 'haloid', 'rhalo', 'r200']

            self.lightcone = False

        if self.usemag:
            self.unitmap = {'halomass':'msunh', 'rhalo':'mpch', 'luminosity':'mag', 'r200':'mpch'}
        else:
            self.unitmap = {'halomass':'msunh', 'rhalo':'mpch', 'r200':'mpch'}


        self.socccounts  = None
        self.cocccounts   = None
        self.sqsocccounts = None
        self.sqcocccounts = None
        self.halocounts   = None

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
        if self.socccounts is None:
            self.socccounts = np.zeros((self.njack, len(self.massbins)-1, self.ndefs,
                                       self.nmagcuts, self.nzbins))
            self.cocccounts = np.zeros((self.njack, len(self.massbins)-1, self.ndefs,
                                       self.nmagcuts, self.nzbins))

            self.sqsocccounts = np.zeros((self.njack,len(self.massbins)-1, self.ndefs,
                                         self.nmagcuts, self.nzbins))
            self.sqcocccounts = np.zeros((self.njack,len(self.massbins)-1, self.ndefs,
                                         self.nmagcuts, self.nzbins))

            self.halocounts = np.zeros((self.njack,len(self.massbins)-1, self.ndefs,
                                        self.nmagcuts, self.nzbins))

        if self.lightcone:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

                #Count number of halos in mass bins before making mag counts
                u, uidx = np.unique(mapunit['haloid'][zlidx:zhidx], return_index=True)

                for j in range(self.ndefs):
                    c, e = np.histogram(mapunit['halomass'][zlidx:zhidx,j][uidx],
                                          bins=self.massbins)
                    self.halocounts[self.jcount,:,j,:,i] += c.reshape((self.massbins.shape[0]-1, 1))

                    for k in range(self.nmagcuts):
                        if self.usemag:
                            if self.cutband is not None:
                                if self.upper_limit:
                                    lidx = (mapunit['luminosity'][zlidx:zhidx, self.cutband]<self.magcuts[k])
                                else:
                                    lidx = ((self.magcuts[k] < mapunit['luminosity'][zlidx:zhidx, self.cutband]) &
                                            (mapunit['luminosity'][zlidx:zhidx, self.cutband]<self.magcuts[k+1]))
                            else:
                                if self.upper_limit:
                                    lidx = (mapunit['luminosity'][zlidx:zhidx]<self.magcuts[k])
                                else:
                                    lidx = ((self.magcuts[k] < mapunit['luminosity'][zlidx:zhidx]) &
                                            (mapunit['luminosity'][zlidx:zhidx]<self.magcuts[k+1]))

                            cidx = mapunit['central'][zlidx:zhidx][lidx]==1
                            sidx = mapunit['rhalo'][zlidx:zhidx][lidx] < mapunit['r200'][zlidx:zhidx][lidx]

                            c, e = np.histogram(mapunit['halomass'][zlidx:zhidx,j][lidx][cidx], bins=self.massbins)
                            c = c.reshape(self.cocccounts[self.jcount,:,j,k,i].shape)
                            self.cocccounts[self.jcount,:,j,k,i] += c
                            self.sqcocccounts[self.jcount,:,j,k,i] += c**2

                            c, e = np.histogram(mapunit['halomass'][zlidx:zhidx,j][lidx][(~cidx) & sidx], bins=self.massbins)
                            c = c.reshape(self.cocccounts[self.jcount,:,j,k,i].shape)
                            self.socccounts[self.jcount,:,j,k,i] += c
                            self.sqsocccounts[self.jcount,:,j,k,i] += c**2
                        else:
                            cidx = mapunit['central'][zlidx:zhidx]==1

                            c, e = np.histogram(mapunit['halomass'][zlidx:zhidx,j][cidx], bins=self.massbins)
                            self.cocccounts[self.jcount,:,j,k,i] += c
                            self.sqcocccounts[self.jcount,:,j,k,i] += c**2

                            c, e = np.histogram(mapunit['halomass'][zlidx:zhidx,j][(~cidx) & sidx], bins=self.massbins)
                            self.socccounts[self.jcount,:,j,k,i] += c
                            self.sqsocccounts[self.jcount,:,j,k,i] += c**2

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
        if rank is not None:
            gsocc = comm.gather(self.socccounts, root=0)
            gcocc = comm.gather(self.cocccounts, root=0)
            gsqsocc = comm.gather(self.sqsocccounts, root=0)
            gsqcocc = comm.gather(self.sqcocccounts, root=0)
            ghcounts = comm.gather(self.halocounts, root=0)

            if rank==0:
                jc = 0
                hshape = [self.halocounts.shape[i] for i in range(len(self.halocounts.shape))]

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

                self.shod = np.sum(self.jshod, axis=0) / self.njacktot
                self.chod = np.sum(self.jchod, axis=0) / self.njacktot

                self.shoderr = np.sqrt(np.sum((self.jshod - self.shod**2), axis=0) * (self.njacktot - 1) / self.njacktot)
                self.choderr = np.sqrt(np.sum((self.jchod - self.chod**2), axis=0) * (self.njacktot - 1) / self.njacktot)

                self.y = self.shod + self.chod
                self.ye = np.sqrt(self.shoderr**2 + self.choderr**2)

        else:
            self.jsocccounts = self.jackknife(self.socccounts, reduce_jk=False)
            self.jsqsocccounts = self.jackknife(self.sqsocccounts, reduce_jk=False)
            self.jcocccounts = self.jackknife(self.cocccounts, reduce_jk=False)
            self.jsqcocccounts = self.jackknife(self.sqcocccounts, reduce_jk=False)
            self.jhalocounts = self.jackknife(self.sqcocccounts, reduce_jk=False)

            self.jshod = self.jsocccounts/self.jhalocounts
            self.jchod = self.jcocccounts/self.jhalocounts

            self.shod = np.sum(self.jshod, axis=0) / self.njacktot
            self.chod = np.sum(self.jchod, axis=0) / self.njacktot

            self.shoderr = np.sqrt(np.sum((self.jshod - self.shod**2), axis=0) * (self.njacktot - 1) / self.njacktot)
            self.choderr = np.sqrt(np.sum((self.jchod - self.chod**2), axis=0) * (self.njacktot - 1) / self.njacktot)

            self.y = self.shod + self.chod
            self.ye = np.sqrt(self.shoderr**2 + self.choderr**2)

    def visualize(self, plotname=None, usecols=None,
                    usez=None,sharex=True, sharey=True, 
                    xlim=None, ylim=None, f=None, ax=None, 
                    label=None, xlabel=None, ylabel=None,
                    compare=False, logx=False, logy=True, 
                    **kwargs):


        mmean = (self.massbins[:-1] + self.massbins[1:]) / 2
        
        if usecols is None:
            usecols = range(self.nmagcuts)
        
        if usez is None:
            usez = range(self.nzbins)

        if f is None:
            f, ax = plt.subplots(len(usecols), self.nzbins,
                                 sharex=True, sharey=False,
                                 figsize=(10,10))
            ax = np.array(ax).reshape((len(usecols), self.nzbins))

            newaxes = True
        else:
            newaxes = False

        
        for i in range(self.nzbins):
            for j, b in enumerate(usecols):
                sy = self.shod[:,0,j,i]
                cy  = self.chod[:,0,j,i]
                sye = self.shoderr[:,0,j,i]
                cye = self.choderr[:,0,j,i]

                ls = ax[j,i].errorbar(mmean, sy, yerr=sye, fmt='^', barsabove=True,
                                      **kwargs)

                lc = ax[j,i].errorbar(mmean, cy, yerr=cye, fmt='s', barsabove=True,
                                      **kwargs)

                lt = ax[j,i].errorbar(mmean, self.y[:,0,j,i],
                                      yerr=self.ye[:,0,j,i],
                                      fmt='.', barsabove=True,
                                      **kwargs)
                


        if not compare:
            for i in range(self.nzbins):
                for j, b in enumerate(usecols):
                    if not (((self.shod[:,0,j,i]==0).all()
                            | ~np.isfinite(self.shod[:,0,j,i]).any())
                            & ((self.chod[:,0,j,i]==0).all()
                            | ~np.isfinite(self.chod[:,0,j,i]).any())):
                        if logx:
                            ax[j,i].set_xscale('log')
                        if logy:
                            ax[j,i].set_yscale('log')

        if xlim is not None:
            ax[0][0].set_xlim(xlim)
        if ylim is not None:
            ax[0][0].set_ylim(ylim)


        return f, ax, ls, lc, lt
                        
    def compare(self, othermetrics, plotname=None, usecols=None, usez=None,
                xlim=None, ylim=None, labels=None, logx=False,
                logy=True,**kwargs):

        tocompare = [self]
        tocompare.extend(othermetrics)

        if usecols is not None:
            if not hasattr(usecols[0], '__iter__'):
                usecols = [usecols]*len(tocompare)
            else:
                assert(len(usecols)==len(tocompare))
        else:
            usecols = [None]*len(tocompare)

        if usez is not None:
            if not hasattr(usez[0], '__iter__'):
                usez = [usez]*len(tocompare)
            else:
                assert(len(usez)==len(tocompare))
        else:
            usez = [None]*len(tocompare)

        if labels is None:
            labels = [None]*len(tocompare)

        lines = []
        labels = []

        for i, m in enumerate(tocompare):
            if usecols[i] is not None:
                assert(len(usecols[0])==len(usecols[i]))
            if i==0:
                f, ax, ls, lc, lt = m.visualize(usecols=usecols[i], xlim=xlim, ylim=ylim, 
                                         compare=True,label=labels[i],usez=usez[i],
                                         color=Metric._color_list[i], **kwargs)
            else:
                f, ax, ls, lc, lt = m.visualize(usecols=usecols[i], xlim=xlim, ylim=ylim,
                                        f=f, ax=ax, compare=True, label=labels[i], 
                                        usez=usez[i], color=Metric._color_list[i], 
                                        **kwargs)
            lines.append(ls[0])
            labels.append(labels[i] + '-sat')
            lines.append(lc[0])
            labels.append(labels[i] + '-cen')
            lines.append(lt[0])
            labels.append(labels[i] + '-tot')


        if labels[0] is not None:
            f.legend(lines, labels, 'best')

        if logx:
            ax[0,0].set_xscale('log')
        if logy:
            ax[0,0].set_yscale('log')



        if plotname is not None:
            plt.savefig(plotname)

        #plt.tight_layout()

        return f, ax
            


class GalCLF(MassMetric):

    def __init__(self, ministry, zbins=None, massbins=None, lightcone=True,
                 catalog_type=['galaxycatalog'], tag=None, magbins=None,
                 magband=None, **kwargs):

        if massbins is None:
            massbins = np.logspace(13, 16, 5)

        MassMetric.__init__(self, ministry, zbins=zbins, massbins=massbins,
                            catalog_type=catalog_type, tag=tag, **kwargs)

        if magbins is None:
            self.magbins = np.linspace(-24,-18,30)
            self.nmagbins = len(self.magbins) - 1
        if magband is None:
            self.magband = 1

        self.nomap = False
        self.aschema = 'galaxyonly'

        if lightcone:
            self.mapkeys = ['halomass', 'central', 'haloid', 'redshift', 'rhalo', 'r200', 'luminosity']
            self.lightcone = True
        else:
            self.mapkeys = ['halomass', 'central', 'haloid', 'rhalo', 'r200', 'luminosity']
            self.lightcone = False

        if self.lightcone:
            self.unitmap = {'halomass':'msunh', 'rhalo':'mpch', 'luminosity':'mag', 'r200':'mpch', 'redshift':'z'}
        else:
            self.unitmap = {'halomass':'msunh', 'rhalo':'mpch', 'r200':'mpch', 'luminosity':'mag'}

        self.slumcounts   = None
        self.clumcounts   = None
        self.halocounts   = None

    @jackknifeMap
    def map(self, mapunit):
        
        if self.slumcounts is None:
            self.slumcounts = np.zeros((self.njack, self.nmagbins, self.nmassbins, self.nzbins))
            self.clumcounts = np.zeros((self.njack, self.nmagbins, self.nmassbins, self.nzbins))
            self.halocounts = np.zeros((self.njack, self.nmassbins, self.nzbins))

        cen = mapunit['central']==1
        sat = ~cen

        for i, z in enumerate(self.zbins[:-1]):
            zidx = (self.zbins[i] < mapunit['redshift']) & (mapunit['redshift'] < self.zbins[i+1])
            
            for j, m in enumerate(self.massbins[:-1]):
                midx = ((self.massbins[j] < mapunit['halomass']) & (mapunit['halomass'] < self.massbins[j+1])).reshape(len(zidx))
                midx = midx & zidx
                
                uids = np.unique(mapunit['haloid'][midx])
                self.halocounts[self.jcount,j,i] = len(uids)

                sc,e = np.histogram(mapunit['luminosity'][midx&sat,self.magband], bins=self.magbins)
                cc,e = np.histogram(mapunit['luminosity'][midx&cen,self.magband], bins=self.magbins)                
                self.slumcounts[self.jcount,:,j,i] = sc
                self.clumcounts[self.jcount,:,j,i] = cc

    def reduce(self,rank=None,comm=None):

        dl = (self.magbins[1:] - self.magbins[:-1]).reshape(1,self.nmagbins,1,1)

        if rank is not None:
            gslumcounts = comm.gather(self.slumcounts, root=0)
            gclumcounts = comm.gather(self.clumcounts, root=0)
            ghalocounts = comm.gather(self.halocounts, root=0)

            if rank==0:
                jc = 0
                sshape = [self.njack,self.nmagbins,self.nmassbins,self.nzbins]
                cshape = [self.njack,self.nmagbins,self.nmassbins,self.nzbins]
                hshape = [self.njack,self.nmassbins,self.nzbins]


                sshape[0] = self.njacktot
                cshape[0] = self.njacktot
                hshape[0] = self.njacktot

                self.slumcounts = np.zeros(sshape)
                self.clumcounts = np.zeros(cshape)
                self.halocounts = np.zeros(hshape)

                for i, g in enumerate(gslumcounts):
                    if g is None: continue
                    nj = g.shape[0]
                    self.slumcounts[jc:jc+nj,:,:,:] = g
                    self.clumcounts[jc:jc+nj,:,:,:] = gclumcounts[i]
                    self.halocounts[jc:jc+nj,:,:] = ghalocounts[i]

                    jc += nj

                self.jslumcounts = self.jackknife(self.slumcounts, reduce_jk=False)
                self.jclumcounts = self.jackknife(self.clumcounts, reduce_jk=False)
                self.jhalocounts = self.jackknife(self.halocounts, reduce_jk=False)

                self.jslumfunction = (self.jslumcounts / self.jhalocounts.reshape(self.njacktot, 
                                                                                 1,self.nmassbins,
                                                                                 self.nzbins) / dl)

                self.jclumfunction = (self.jclumcounts / self.jhalocounts.reshape(self.njacktot, 
                                                                                 1,self.nmassbins,
                                                                                 self.nzbins) / dl)

                self.jsatellite_frac = self.jslumfunction / (self.jslumfunction + self.jclumfunction)
                
                self.slumfunction = (np.sum( self.jslumfunction, axis=0 ) * 
                                     (self.njacktot - 1) / self.njacktot)
                self.clumfunction = (np.sum( self.jclumfunction, axis=0 ) * 
                                     (self.njacktot - 1) / self.njacktot)
                self.satellite_frac = (np.sum( self.jsatellite_frac, axis=0) *
                                       (self.njacktot - 1) / self.njacktot)

                self.varslumfunction = (np.sum((self.jslumfunction - self.slumfunction)**2, 
                                               axis=0) * (self.njacktot - 1) / self.njacktot)

                self.varclumfunction = (np.sum((self.jclumfunction - self.clumfunction)**2, 
                                               axis=0) * (self.njacktot - 1) / self.njacktot)
                self.varsatellite_frac = (np.sum((self.jsatellite_frac - self.satellite_frac)**2, 
                                               axis=0) * (self.njacktot - 1) / self.njacktot)

        else:
            self.jslumcounts = self.jackknife(self.slumcounts, reduce_jk=False)
            self.jclumcounts = self.jackknife(self.clumcounts, reduce_jk=False)
            self.jhalocounts = self.jackknife(self.halocounts, reduce_jk=False)

            self.jslumfunction = (self.jslumcounts / self.jhalocounts.reshape(self.njacktot, 
                                                                             1,self.nmassbins,
                                                                             self.nzbins) / dl)

            self.jclumfunction = (self.jclumcounts / self.jhalocounts.reshape(self.njacktot, 
                                                                             1,self.nmassbins,
                                                                             self.nzbins) / dl)
            self.slumfunction = (np.sum( self.jslumfunction, axis=0 ) 
                                  / self.njacktot)
            self.clumfunction = (np.sum( self.jclumfunction, axis=0 )  
                                  / self.njacktot)
            self.satellite_frac = (np.sum( self.jsatellite_frac, axis=0) *
                                   (self.njacktot - 1) / self.njacktot)
                                           

            self.varslumfunction = (np.sum((self.jslumfunction - self.slumfunction)**2, 
                                           axis=0) * (self.njacktot - 1) / self.njacktot)

            self.varclumfunction = (np.sum((self.jclumfunction - self.clumfunction)**2, 
                                           axis=0) * (self.njacktot - 1) / self.njacktot)
            self.varsatellite_frac = (np.sum((self.jsatellite_frac - self.satellite_frac)**2, 
                                           axis=0) * (self.njacktot - 1) / self.njacktot)
                                           

    def visualize(self, plotname=None, usecols=None,
                    usez=None,sharex=True, sharey=True, 
                    xlim=None, ylim=None, f=None, ax=None, 
                    label=None, xlabel=None, ylabel=None,
                    compare=False, logx=False, logy=True,
                    satellite_frac=False, **kwargs):

        lmean = (self.magbins[:-1] + self.magbins[1:]) / 2
        
        if usecols is None:
            usecols = range(self.nmassbins)
        
        if usez is None:
            usez = range(self.nzbins)

        if f is None:
            f, ax = plt.subplots(len(usecols), self.nzbins,
                                 sharex=True, sharey=False,
                                 figsize=(10,10))
            ax = np.array(ax).reshape((len(usecols), self.nzbins))

            newaxes = True
        else:
            newaxes = False
                                 
        
        for i in range(self.nzbins):
            for j, b in enumerate(usecols):
                sy = self.slumfunction[:,j,i]
                cy  = self.clumfunction[:,j,i]
                sye = np.sqrt(self.varslumfunction[:,j,i])
                cye = np.sqrt(self.varclumfunction[:,j,i])

                ls = ax[j,i].errorbar(lmean, sy, yerr=sye, barsabove=True,
                                      **kwargs)
                #ax[j,i].fill_between(lmean, sy-sye, sy+sye, alpha=0.5,
                #                     **kwargs)
                lc = ax[j,i].errorbar(lmean, cy, yerr=cye, fmt='--', barsabove=True,
                                      **kwargs)
                #ax[j,i].fill_between(lmean, cy-cye, cy+cye, alpha=0.5,
                #                     **kwargs)
#                if logx:
#                    ax[j,i].set_xscale('log')
#                if logy:
#                    ax[j,i].set_yscale('log')


        if not compare:
            for i in range(self.nzbins):
                for j, b in enumerate(usecols):
                    if not (((self.slumfunction[:,j,i]==0).all()
                            | ~np.isfinite(self.slumfunction[:,j,i]).any())
                            & ((self.clumfunction[:,j,i]==0).all()
                            | ~np.isfinite(self.clumfunction[:,j,i]).any())):
                        if logx:
                            ax[j,i].set_xscale('log')
                        if logy:
                            ax[j,i].set_yscale('log')

#                if (i==0) & (j==0):
#                    if xlim is not None:
#                        ax[0][0].set_xlim(xlim)
#                    if ylim is not None:
#                        ax[0][0].set_ylim(ylim)
        if newaxes:
            sax = f.add_subplot(111)
            plt.setp(sax.get_xticklines(), visible=False)
            plt.setp(sax.get_yticklines(), visible=False)
            plt.setp(sax.get_xticklabels(), visible=False)
            plt.setp(sax.get_yticklabels(), visible=False)
            sax.patch.set_alpha(0.0)
            sax.patch.set_facecolor('none')
            sax.spines['top'].set_color('none')
            sax.spines['top'].set_alpha(0.0)
            sax.spines['bottom'].set_color('none')
            sax.spines['bottom'].set_alpha(0.0)
            sax.spines['left'].set_color('none')
            sax.spines['left'].set_alpha(0.0)
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            sax.set_xlabel(r'%s' % xlabel, labelpad=40)
            sax.set_ylabel(r'%s' % ylabel, labelpad=40)
            #plt.tight_layout()


        return f, ax, ls, lc
                        
    def compare(self, othermetrics, plotname=None, usecols=None, usez=None,
                xlim=None, ylim=None, labels=None, logx=False,
                logy=True,**kwargs):

        tocompare = [self]
        tocompare.extend(othermetrics)

        if usecols is not None:
            if not hasattr(usecols[0], '__iter__'):
                usecols = [usecols]*len(tocompare)
            else:
                assert(len(usecols)==len(tocompare))
        else:
            usecols = [None]*len(tocompare)

        if usez is not None:
            if not hasattr(usez[0], '__iter__'):
                usez = [usez]*len(tocompare)
            else:
                assert(len(usez)==len(tocompare))
        else:
            usez = [None]*len(tocompare)

        if labels is None:
            labels = [None]*len(tocompare)

        lines = []
        clflabels = []

        for i, m in enumerate(tocompare):
            if usecols[i] is not None:
                assert(len(usecols[0])==len(usecols[i]))
            if i==0:
                f, ax, ls, lc = m.visualize(usecols=usecols[i], xlim=xlim, ylim=ylim, 
                                         compare=True,label=labels[i],usez=usez[i],
                                         color=Metric._color_list[i], **kwargs)
            else:
                f, ax, ls, lc = m.visualize(usecols=usecols[i], xlim=xlim, ylim=ylim,
                                        f=f, ax=ax, compare=True, label=labels[i], 
                                        usez=usez[i], color=Metric._color_list[i], 
                                        **kwargs)
            lines.append(ls[0])
            clflabels.append(labels[i] + 'sat')
            lines.append(lc[0])
            clflabels.append(labels[i] + 'cen')


        if labels[0] is not None:
            f.legend(lines, clflabels, 'best')

        if logx:
            ax[0,0].set_xscale('log')
        if logy:
            ax[0,0].set_yscale('log')



        if plotname is not None:
            plt.savefig(plotname)

        #plt.tight_layout()

        return f, ax
        
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
            mass_bin_indices = np.digitize(uniquehalos['halomass'], self.massbins).reshape(len(uniquehalos))

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
