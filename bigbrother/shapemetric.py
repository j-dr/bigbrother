from __future__ import print_function, division
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import fitsio
import warnings
import seaborn as sns
import sys

try:
    import treecorr
    hastreecorr = True
except:
    hastreecorr = False

import numpy as np
import healpy as hp
import healpix_util as hu

from .metric import Metric, GMetric, jackknifeMap
from .corrmetric import CrossCorrelationFunction

class ShearShear(CrossCorrelationFunction):

    def __init__(self, ministry, min_sep=1, max_sep=400, 
                 nabins=100, sep_units='arcmin', bin_slop=1.0,
                 **kwargs):
        """
        Shear-shear correlation function
        """
        CrossCorrelationFunction.__init__(self, ministry, **kwargs)
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.nabins  = nabins
        self.sep_units = sep_units
        self.bin_slop = bin_slop
        self.amean = None

        self.mapkeys = ['gamma1', 'gamma2', 'azim_ang', 'polar_ang', 'redshift']
        self.unitmap = {'polar_ang':'dec', 'azim_ang':'ra', 'redshift':'z',
                        'gamma1':'ra', 'gamma2':'dec'}

        if self.mkey:
            self.mapkeys.append(self.mkey)
        if self.mkey1:
            self.mapkeys.append(self.mkey1)

        if (self.mkey == 'appmag') | (self.mkey=='luminosity'):
            self.unitmap[self.mkey] = 'mag'
        if (self.mkey == 'halomass'):
            self.unitmap[self.mkey] = 'msunh'

        if (self.mkey1 == 'appmag') | (self.mkey1=='luminosity'):
            self.unitmap[self.mkey1] = 'mag'
        if (self.mkey1 == 'halomass'):
            self.unitmap[self.mkey1] = 'msunh'

        self.catalog_type = [self.catalog_type[0]]

        self.jsamples = 0
        
        
        self.unxi_m  = None
        self.unxi_p  = None
        self.varg1  = None
        self.varg2 = None
        self.nd1   = None
        self.nd2   = None
        self.weights = None


    @jackknifeMap
    def map(self, mapunit):

        if not self.unxi_m:
            self.unxi_m   = np.zeros((self.njack, self.nabins, self.nmbins,
                                     self.nmbins1, self.nzbins))
            self.unxi_p   = np.zeros((self.njack, self.nabins, self.nmbins,
                                     self.nmbins1, self.nzbins))
            self.weights = np.zeros((self.njack, self.nabins, self.nmbins,
                                     self.nmbins1, self.nzbins))
            self.varg1    = np.zeros((self.njack, self.nmbins,
                                      self.nmbins1, self.nzbins))
            self.varg2   = np.zeros((self.njack, self.nmbins,
                                      self.nmbins1, self.nzbins))
            self.nd1      = np.zeros((self.njack, self.nmbins,
                                      self.nmbins1, self.nzbins))
            self.nd2      = np.zeros((self.njack, self.nmbins,
                                      self.nmbins1, self.nzbins))

        for i in range(self.nzbins):
            print('Finding redshift indices')

            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

            if (zlidx==zhidx):
                print("No galaxies in redshift bin {0} to {1}".format(self.zbins[i], self.zbins[i+1]))
                print('z: {}'.format(z))

                print("Min and max z: {0}, {1}".format(np.min(z), np.max(z)))
                continue

            for li, j in enumerate(self.minds):
                if self.mkey is not None:
                    if self.mcutind is not None:
                        if self.upper_limit:
                            lidx = mapunit[self.mkey][zlidx:zhidx,self.mcutind] < self.mbins[j]
                        else:
                            lidx = ((self.mbins[j] <= mapunit[self.mkey][zlidx:zhidx,self.mcutind]) 
                                    & (mapunit[self.mkey][zlidx:zhidx,self.mcutind] < self.mbins[j+1]))
                    else:
                        if self.upper_limit:
                            lidx = mapunit[self.mkey][zlidx:zhidx] < self.mbins[j]
                        else:
                            lidx = (self.mbins[j] <= mapunit[self.mkey][zlidx:zhidx]) & (mapunit[self.mkey][zlidx:zhidx] < self.mbins[j+1])
                else:
                    lidx = np.ones(zhidx-zlidx, dtype=np.bool)

                for li1, j1 in enumerate(self.minds1):
                    if self.mkey1 is not None:
                        if self.mcutind1 is not None:
                            if self.upper_limit1:
                                lidx1 = mapunit[self.mkey1][zlidx:zhidx,self.mcutind1] < self.mbins1[j1]
                            else:
                                lidx1 = (self.mbins1[j1] <= mapunit[self.mkey1][zlidx:zhidx,self.mcutind1]) & (mapunit[self.mkey1][zlidx:zhidx,self.mcutind1] < self.mbins1[j1+1])
                        else:
                            if self.upper_limit1:
                                lidx1 = mapunit[self.mkey1][zlidx:zhidx] < self.mbins1[j1]
                            else:
                                lidx1 = (self.mbins1[j1] <= mapunit[self.mkey1][zlidx:zhidx]) & (mapunit[self.mkey1][zlidx:zhidx] < self.mbins1[j1+1])
                    else:
                        lidx1 = np.ones(zhidx-zlidx, dtype=np.bool)

                    self.nd1[self.jcount,j,j1,i] = len(mapunit['azim_ang'][zlidx:zhidx][lidx])
                    self.nd2[self.jcount,j,j1,i] = len(mapunit['azim_ang'][zlidx:zhidx][lidx1])

                    print("Number of cat1 in this z/lum bin: {0}".format(np.sum(lidx)))
                    print("Number of cat2 in this z/lum bin: {0}".format(np.sum(lidx1)))

                    cat1 = treecorr.Catalog(g1=mapunit['gamma1'][zlidx:zhidx][lidx], g2=-mapunit['gamma2'][zlidx:zhidx][lidx], 
                                           ra=mapunit['azim_ang'][zlidx:zhidx][lidx], dec=mapunit['polar_ang'][zlidx:zhidx][lidx],
                                           ra_units='deg', dec_units='deg')

                    cat2 = treecorr.Catalog(g1=mapunit['gamma1'][zlidx:zhidx][lidx1], g2=-mapunit['gamma2'][zlidx:zhidx][lidx1], 
                                           ra=mapunit['azim_ang'][zlidx:zhidx][lidx1], dec=mapunit['polar_ang'][zlidx:zhidx][lidx1],
                                           ra_units='deg', dec_units='deg')

                    self.varg1[self.jcount,j,j1,i] = cat1.varg
                    self.varg2[self.jcount,j,j1,i] = cat2.varg

                    sys.stdout.flush()
                    if (self.nd1[self.jcount,j,j1,i]<1) | (self.nd2[self.jcount,j,j1,i]<1):
                        continue

                    print('processing shear-shear correlation')
                    gg = treecorr.GGCorrelation(min_sep=self.min_sep, max_sep=self.max_sep, nbins=self.nabins, 
                                                sep_units=self.sep_units, bin_slop=self.bin_slop)

                    gg.process_cross(cat1, cat2, num_threads=self.nthreads)
                    
                    if self.amean is None:
                        if (gg.meanlogr!=0.0).any():
                            self.amean = np.exp(gg.meanlogr)
                        else:
                            self.amean = np.exp(gg.logr)                        

                    self.unxi_p[self.jcount,:,j,j1,i] = gg.xip
                    self.unxi_m[self.jcount,:,j,j1,i] = gg.xim
                    self.weights[self.jcount,:,j,j1,i] = gg.weight


    def reduce(self, rank=None, comm=None):

        if rank is not None:
            gnd1 = comm.gather(self.nd1, root=0)
            gnd2 = comm.gather(self.nd2, root=0)
            gweights = comm.gather(self.weights, root=0)
            gvarg1 = comm.gather(self.varg1, root=0)
            gvarg2 = comm.gather(self.varg1, root=0)

            gunxi_p = comm.gather(self.unxi_p, root=0)
            gunxi_m = comm.gather(self.unxi_m, root=0)

            if rank==0:
                nd1shape = [self.nd1.shape[i] for i in range(len(self.nd1.shape))]
                xishape = [self.unxi_p.shape[i] for i in range(len(self.unxi_p.shape))]

                nd1shape.insert(1,1)

                nd1shape[0] = self.njacktot
                xishape[0] = self.njacktot


                self.nd1 = np.zeros(nd1shape)
                self.nd2 = np.zeros(nd1shape)
                self.varg1 = np.zeros(nd1shape)
                self.varg2 = np.zeros(nd1shape)

                self.unxi_p = np.zeros(xishape)
                self.unxi_m = np.zeros(xishape)
                self.weights = np.zeros(xishape)

                jc = 0
                for i, g in enumerate(gnd1):
                    if g is None: continue
                    nj = g.shape[0]
                    self.nd1[jc:jc+nj,0,:,:,:] = g
                    self.nd2[jc:jc+nj,0,:,:,:] = gnd2[i]
                    self.varg1[jc:jc+nj,0,:,:,:] = gvarg1[i]
                    self.varg2[jc:jc+nj,0,:,:,:] = gvarg2[i]


                    self.weights[jc:jc+nj,:,:,:,:] = gdd[i]
                    self.unxi_p[jc:jc+nj,:,:,:,:] = gdr[i]
                    self.unxi_m[jc:jc+nj,:,:,:,:] = grd[i]

                    jc += nj

                self.jnd1     = self.jackknife(self.nd1, reduce_jk=False)
                self.jnd2     = self.jackknife(self.nd2, reduce_jk=False)
                self.jweights = self.jackknife(self.weights, reduce_jk=False)
                
                self.jxip     = self.jackknife(self.unxi_p, reduce_jk=False) / self.jweights
                self.jxim     = self.jackknife(self.unxi_m, reduce_jk=False) / self.jweights

                self.xip     = np.sum(self.jxip, axis=0) / self.njacktot
                self.xim     = np.sum(self.jxim, axis=0) / self.njacktot
                self.var_xip = np.sum(self.jxip - self.xip, axis=0) / self.njacktot 
                self.var_xim = np.sum(self.jxim - self.xim, axis=0) / self.njacktot

                self.var_xip += np.sum(self.varg1 * self.nd1) / np.sum(self.nd1)
                self.var_xim += np.sum(self.varg2 * self.nd2) / np.sum(self.nd2)

        else:
            self.jnd1     = self.jackknife(self.nd1, reduce_jk=False).reshape(-1,1,self.nmbins,self.nmbins1,self.nzbins)
            self.jnd2     = self.jackknife(self.nd2, reduce_jk=False).reshape(-1,1,self.nmbins,self.nmbins1,self.nzbins)
            self.jweights = self.jackknife(self.weights, reduce_jk=False)
                
            self.jxip     = self.jackknife(self.unxi_p, reduce_jk=False) / self.jweights
            self.jxim     = self.jackknife(self.unxi_m, reduce_jk=False) / self.jweights

            self.xip     = np.sum(self.jxip, axis=0) / self.njacktot
            self.xim     = np.sum(self.jxim, axis=0) / self.njacktot
            self.var_xip = np.sum(self.jxip - self.xip, axis=0) / self.njacktot 
            self.var_xim = np.sum(self.jxim - self.xim, axis=0) / self.njacktot

            self.var_xip += np.sum(self.varg1 * self.nd1) / np.sum(self.nd1)
            self.var_xim += np.sum(self.varg2 * self.nd2) / np.sum(self.nd2)


    def visualize(self, plotname=None, f=None, ax=None, usecols=None,
                    usez=None, compare=False, usecols1=None,**kwargs):

        if usecols is None:
            usecols = range(self.nmbins)

        if usecols1 is None:
            usecols1 = range(self.nmbins1)

        if usez is None:
            usez = range(self.nzbins)

        if f is None:
            f, ax = plt.subplots(len(usez), len(usecols)*len(usecols1), sharex=True,
                                    sharey=True, figsize=(8,8))
            ax = np.array(ax)
            ax = ax.reshape(len(usez), len(usecols))
            newaxes = True
        else:
            newaxes = False

        if hasattr(self, 'rmean'):
            if self.rmean is not None:
                rmean = self.rmean
        else:
            rmean = (self.abins[1:]+self.abins[:-1]) / 2

        for i, l in enumerate(usecols):
            for i1, l1 in enumerate(usecols1):
                for j, z in enumerate(usez):
                    xipe = np.sqrt(self.var_xip[:,l,l1,z])
                    xime = np.sqrt(self.var_xim[:,l,l1,z])
                    lp = ax[j][i].plot(rmean, self.xip[:,l,l1,z], **kwargs)
                    ax[j][i].fill_between(rmean, self.xip[:,l,l1,z]-xipe, self.xip[:,l,l1,z]+xipe, alpha=0.5, **kwargs)
                    lm = ax[j][i].plot(rmean, self.xim[:,l,l1,z], **kwargs)
                    ax[j][i].fill_between(rmean, self.xim[:,l,l1,z]-xime, self.xim[:,l,l1,z]+xime, alpha=0.5, **kwargs)

            ax[j][i].set_xscale('log')
            ax[j][i].set_yscale('log')


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
            sax.set_ylabel(r'$\xi_{+/i}(\theta)$', labelpad=20, fontsize=16)
            sax.set_xlabel(r'$\theta \, [ degrees ]$',labelpad=20, fontsize=16)

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax, l1[0]


    def compare(self, othermetrics, plotname=None, usecols=None,
                 usecols1=None, usez=None, labels=None, **kwargs):

        tocompare = [self]
        tocompare.extend(othermetrics)

        if usecols is not None:
            if not hasattr(usecols[0], '__iter__'):
                usecols = [usecols]*len(tocompare)
            else:
                assert(len(usecols)==len(tocompare))
        else:
            usecols = [None]*len(tocompare)

        if usecols1 is not None:
            if not hasattr(usecols1[0], '__iter__'):
                usecols1 = [usecols1]*len(tocompare)
            else:
                assert(len(usecols1)==len(tocompare1))
        else:
            usecols1 = [None]*len(tocompare)


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

        for i, m in enumerate(tocompare):
            if usecols[i] is not None:
                assert(len(usecols[0])==len(usecols[i]))
            if i==0:
                f, ax, l1 = m.visualize(usecols=usecols[i], usez=usez[i], usecols1=usecols1[i],
                                          compare=True, color=Metric._color_list[i],
                                          **kwargs)
            else:
                f, ax, l1 = m.visualize(usecols=usecols[i], usez=usez[i], usecols1=usecols1[i],
                                          compare=True, color=Metric._color_list[i],
                                          f=f, ax=ax, **kwargs)
            lines.append(l1)

        if labels[0]!=None:
            f.legend(lines, labels)

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax
