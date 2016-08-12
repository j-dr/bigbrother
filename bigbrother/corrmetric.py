from __future__ import print_function, division
#if __name__=='__main__':
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import sys

try:
    import treecorr as tc
    hastreecorr = True
except:
    hastreecorr = False

try:
    import Corrfunc._countpairs_mocks as countpairs_mocks
    import Corrfunc._countpairs as countpairs
    hascorrfunc = True
except:
    hascorrfunc = False

import numpy as np
import healpy as hp

from .metric import Metric, GMetric

class CorrelationFunction(Metric):

    def __init__(self, ministry, zbins=None, lumbins=None,
                   nrbins=None, subjack=False, lightcone=True,
                   catalog_type=None, lcutind=None,
                   tag=None, same_rand=False, inv_lum=True,
                   **kwargs):
        """
        Generic correlation function.
        """
        Metric.__init__(self, ministry, tag=tag, **kwargs)

        if catalog_type is None:
            self.catalog_type = ['galaxycatalog']
        else:
            self.catalog_type = catalog_type

        self.lightcone = lightcone

        if (zbins is None) & lightcone:
            self.zbins = np.linspace(self.ministry.minz, self.ministry.maxz, 4)
            self.nzbins = len(self.zbins)-1
        elif lightcone:
            self.zbins = zbins
            self.zbins = np.array(self.zbins)
            self.nzbins = len(self.zbins)-1
        else:
            self.nzbins = 1
            self.zbins = None

        if lumbins is None:
            self.lumbins = np.array([-22, -21, -20, -19])
        else:
            self.lumbins = lumbins

        self.nlumbins = len(self.lumbins)-1

        if same_rand & inv_lum:
            self.luminds = np.arange(self.nlumbins)[::-1]
        else:
            self.luminds = np.arange(self.nlumbins)

        self.same_rand = same_rand
        self.inv_lum = inv_lum

        if nrbins is None:
            self.nrbins = 15
        else:
            self.nrbins = nrbins

        if lcutind is None:
            self.lcutind = 0
        else:
            self.lcutind = lcutind

        self.subjack = subjack

        if 'galaxycatalog' in self.catalog_type:
            self.aschema = 'galaxygalaxy'
        else:
            self.aschema = 'halohalo'

        if self.subjack:
            raise NotImplementedError

        self.jsamples = 0

    def generateAngularRandoms(self, aza, pla, z=None, rand_factor=20, nside=8, nest=True):
       """
       Generate a set of randoms from a catalog by pixelating the input
       catalog and uniformly distributing random points within the pixels
       occupied by galaxies.

       Also option to assign redshifts with the same distribution as the input
       catalog to allow for generation of randoms once for all z bins.
       """

       if z is not None:
           rdtype = np.dtype([('azim_ang', np.float32), ('polar_ang', np.float32),
                              ('redshift', np.float32)])
       else:
           rdtype = np.dtype([('azim_ang', np.float32), ('polar_ang', np.float32)])

       rsize = len(aza)*rand_factor

       #randomly generate angles within region bounded by catalog angles
       grand = np.zeros(rsize, dtype=rdtype)
       grand['azim_ang'] = np.random.uniform(low=np.min(aza),
                                             high=np.max(aza),
                                             size=rsize)
       grand['polar_ang'] = np.random.uniform(low=np.min(pla),
                                              high=np.max(pla),
                                              size=rsize)
       if z is not None:
           grand['redshift'] = np.random.choice(z, size=rsize)
           zidx = grand['redshift'].argsort()
           grand = grand[zidx]

       #only keep points which fall within the healpix cells overlapping the catalog
       cpix = hp.ang2pix(nside, (pla+90)*np.pi/180., aza*np.pi/180., nest=nest)
       ucpix = np.unique(cpix)
       rpix = hp.ang2pix(nside, (pla+90)*np.pi/180, aza*np.pi/180., nest=nest)
       inarea = np.in1d(rpix, ucpix)

       grand = grand[inarea]

       return grand

    def genbins(self, minb, maxb, nb):

        if self.logbins:
            bins = np.logspace(np.log10(minb), np.log10(maxb), nb+1)
        else:
            bins = np.linspace(minb, maxb, nb+1)

        return bins

    def writeCorrfuncBinFile(self, binedges,
      binfilename='bb_corrfunc_rbins.txt'):
        """
        Write a bin file for corrfunc.

        inputs
        ---------
        binedges -- np.array
            If 1d, should be array of length nbins+1 of all binedges
            If 2d, should be array of length nbins where the 1st column
            is the left edge of the bins and the 2nd column is the right
            edge.
        """
        if len(binedges.shape)==1:
            binarray = np.array([[binedges[i], binedges[i+1]]
                                  for i in range(len(binedges)-1)])
        elif len(binedges.shape)==2:
            binarray = binedges

        np.savetxt(binfilename, binarray, fmt='%.12f', delimiter='\t')
        self.binfilename = binfilename

class AngularCorrelationFunction(CorrelationFunction):

    def __init__(self, ministry, zbins=None, lumbins=None, mintheta=None,
                 maxtheta=None, nabins=None, subjack=False,
                 catalog_type=None, tag=None, lcutind=None, **kwargs):
        """
        Angular correlation function, w(theta), for use with non-periodic
        data. All angles should be specified in degrees.
        """
        CorrelationFunction.__init__(self, ministry, zbins=zbins,
                                      lumbins=lumbins, nrbins=nabins,
                                      subjack=subjack, lcutind=lcutind,
                                      catalog_type=catalog_type, tag=tag,
                                      **kwargs)

        if mintheta is None:
            self.mintheta = 1e-2
        else:
            self.mintheta = mintheta

        if maxtheta is None:
            self.maxtheta = 1
        else:
            self.maxtheta = maxtheta

        self.mapkeys = ['luminosity', 'redshift', 'polar_ang', 'azim_ang']
        self.unitmap = {'luminosity':'mag', 'polar_ang':'dec', 'azim_ang':'ra'}

    def map(self, mapunit):
        if not hastreecorr:
            return

        self.jsamples += 1

        if not hasattr(self, 'wthetaj'):
            self.wthetaj = np.zeros((self.nabins, self.nlumbins, self.nzbins))
            self.varwthetaj = np.zeros((self.nabins, self.nlumbins, self.nzbins))


        #putting this outside loop maybe faster, inside loop
        #lower memory usage
        rand = self.generateAngularRandoms(mapunit['azim_ang'], mapunit['polar_ang'], nside=128)

        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])
            zrlidx = rand['redshift'].searchsorted(self.zbins[i])
            zrhidx = rand['redshift'].searchsorted(self.zbins[i+1])

            for j in range(self.nlumbins):
                #luminosity should be at least 2d
                lidx = np.where((self.lumbins[i] < mapunit['luminosity'][zlidx:zhidx,0]) &
                                (mapunit['luminosity'][zlidx:zhidx,0] <= self.lumbins[i+1]))
                cat  = {key:mapunit[key][zlidx:zhidx][lidx] for key in mapunit.keys()}


                d  = treecorr.Catalog(x=cat['azim_ang'], y=cat['polar_ang'])
                r = treecorr.Catalog(x=rand[zrlidx:zrhidx]['azim_ang'], y=rand['polar_ang'])

                dd = treecorr.NNCorrelation(nbins=self.nabins, min_sep=self.mintheta,
                                            max_sep=self.maxtheta, sep_units='degrees',
                                            bin_slop=0.001)
                dr = treecorr.NNCorrelation(nbins=self.nabins, min_sep=self.mintheta,
                                            max_sep=self.maxtheta, sep_units='degrees',
                                            bin_slop=0.001)
                rr = treecorr.NNCorrelation(nbins=self.nabins, min_sep=self.mintheta,
                                            max_sep=self.maxtheta, sep_units='degrees',
                                            bin_slop=0.001)
                dd.process(d)
                dr.process(d,r)
                rr.process(r)
                xi,varXi = dd.calculateXi(rr,dr)
                self.wthetaj[:,j,i] = xi
                self.varwthetaj[:,j,i] = varXi

    def reduce(self):
       pass

    def visualize(self):
        pass

    def compare(self):
        pass


class WPrpLightcone(CorrelationFunction):

    def __init__(self, ministry, zbins=None, lumbins=None, rbins=None,
                  minr=None, maxr=None, logbins=True, nrbins=None,
                  pimax=None, subjack=False, catalog_type=None, tag=None,
                  njack=None, lcutind=None, same_rand=False, inv_lum=True,
                  cosmology_flag=None, **kwargs):
        """
        Angular correlation function, w(theta), for use with non-periodic
        data. All angles should be specified in degrees.
        """
        CorrelationFunction.__init__(self, ministry, zbins=zbins,
                                      lightcone=True, lumbins=lumbins,
                                      nrbins=nrbins, subjack=subjack,
                                      lcutind=lcutind, same_rand=same_rand,
                                      inv_lum=inv_lum,catalog_type=catalog_type,
                                      tag=tag, **kwargs)

        self.logbins = logbins
        self.c = 299792.458

        if (rbins is None) & ((minr is None) | (maxr is None) | (nrbins is None)):
            self.minr = 1e-1
            self.maxr = 25
            self.nrbins = 14
            self.rbins = self.genbins(self.minr, self.maxr, self.nrbins)
        elif ((minr is not None) & (maxr is not None) & (nrbins is not None)):
            self.minr = minr
            self.maxr = maxr
            self.nrbins = nrbins
            self.rbins = self.genbins(minr, maxr, nrbins)
        else:
            self.rbins = rbins
            self.minr = rbins[0]
            self.maxr = rbins[1]
            self.nrbins = len(rbins)-1

        if cosmology_flag is None:
            self.cosmology_flag = 2
        else:
            self.cosmology_flag = cosmology_flag

        if pimax is None:
            self.pimax = 80.0
        else:
            self.pimax = pimax

        if njack is None:
            self.njack = 1

        self.jcount = 0

        self.writeCorrfuncBinFile(self.rbins)
        #self.binfilename = '/anaconda/lib/python2.7/site-packages/Corrfunc/xi_mocks/tests/bins'

        self.mapkeys = ['luminosity', 'redshift', 'polar_ang', 'azim_ang']
        self.unitmap = {'luminosity':'mag', 'polar_ang':'dec', 'azim_ang':'ra'}

    def map(self, mapunit):

        if not hascorrfunc:
            raise(ImportError("CorrFunc is required to calculate wp(rp)"))

        if not hasattr(self, 'wthetaj'):
            self.dd = np.zeros((self.nrbins, self.nlumbins, self.nzbins, self.njack))
            self.dr = np.zeros((self.nrbins, self.nlumbins, self.nzbins, self.njack))
            self.rr = np.zeros((self.nrbins, self.nlumbins, self.nzbins, self.njack))
            self.nd = np.zeros((self.nlumbins, self.nzbins, self.njack))
            self.nr = np.zeros((self.nlumbins, self.nzbins, self.njack))

        #calculate DD
        for i in range(self.nzbins):
            print('Finding redshift indices')

            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

            #zrlidx = rands['redshift'].searchsorted(self.zbins[i])
            #zrhidx = rands['redshift'].searchsorted(self.zbins[i+1])

            for li, j in enumerate(self.luminds):
                print('Finding luminosity indices')
                lidx = (self.lumbins[j] <= mapunit['luminosity'][zlidx:zhidx,self.lcutind]) & (mapunit['luminosity'][zlidx:zhidx,self.lcutind] < self.lumbins[j+1])

                if (li==0) | (not self.same_rand):
                    print('Generating Randoms')
                    print(li)
                    print(self.same_rand)
                    rands = self.generateAngularRandoms(mapunit['azim_ang'][zlidx:zhidx][lidx], mapunit['polar_ang'][zlidx:zhidx][lidx], selectz=True, nside=128)

                self.nd[j,i,self.jcount] = len(mapunit[zlidx:zhidx][lidx])
                self.nr[j,i,self.jcount] = len(rands)

                print("Number of galaxies in this z/lum bin: {0}".format(self.nd[j,i,self.jcount]))
                print("Number of randoms in this z/lum bin: {0}".format(self.nr[j,i,self.jcount]))

                #data data
                print('calculating data data pairs')
                sys.stdout.flush()

                ddresults = countpairs_mocks.countpairs_rp_pi_mocks(1,
                                        self.cosmology_flag, 1,
                                        self.pimax,
                                        self.binfilename,
                                        mapunit['azim_ang'][zlidx:zhidx][lidx],
                                        mapunit['polar_ang'][zlidx:zhidx][lidx],
                                        mapunit['redshift'][zlidx:zhidx][lidx]*self.c,
                                        mapunit['azim_ang'][zlidx:zhidx][lidx],
                                        mapunit['polar_ang'][zlidx:zhidx][lidx],
                                        mapunit['redshift'][zlidx:zhidx][lidx]*self.c)

                self.dd[:,j,i,self.jcount] = np.array([ddresults[k][4] for k in range(self.nrbins)])

                #data randoms
                print('calculating data random pairs')
                drresults = countpairs_mocks.countpairs_rp_pi_mocks(0, 1, 1,
                                        self.pimax,
                                        self.binfilename,
                                        mapunit['azim_ang'][zlidx:zhidx][lidx],
                                        mapunit['polar_ang'][zlidx:zhidx][lidx],
                                        mapunit['redshift'][zlidx:zhidx][lidx]*self.c,
                                        rands['azim_ang'],
                                        rands['polar_ang'],
                                        rands['redshift']*self.c)

                self.dr[:,j,i,self.jcount] = np.array([drresults[k][4] for k in range(self.nrbins)])

                #randoms randoms
                print('calculating random random pairs')
                if (li==0) | (not self.same_rand):
                    rrresults = countpairs_mocks.countpairs_rp_pi_mocks(1, 1, 1,
                                        self.pimax,
                                        self.binfilename,
                                        rands['azim_ang'],
                                        rands['polar_ang'],
                                        rands['redshift']*self.c,
                                        rands['azim_ang'],
                                        rands['polar_ang'],
                                        rands['redshift']*self.c)

                self.rr[:,j,i,self.jcount] = np.array([rrresults[k][4] for k in range(self.nrbins)])

        self.jcount += 1

    def reduce(self):

        self.jwprp = np.zeros(self.dd.shape)

        if self.njack>1:
            for i in range(self.njack):
                idx = [j for j in range(self.njack) if i!=j]
                nd = np.sum(self.nd[:,:,idx],axis=-1)
                nr = np.sum(self.nr[:,:,idx],axis=-1)

                DD = np.sum(self.dd[:,:,:,idx],axis=-1) / (nd * (nd - 1) / 2)
                RR = np.sum(self.rr[:,:,:,idx],axis=-1) / (nr * (nr - 1) / 2)
                DR = np.sum(self.dr[:,:,:,idx],axis=-1) / (nd * nr)

                self.jwprp[:,:,:,i] = (DD - 2 * DR + RR) / RR

            self.wprp = np.sum(self.jwprp, axis=-1)/self.njack
            self.varwprp = np.sum((self.jwprp-self.wprp)**2, axis=-1) * (self.njack - 1)/self.njack
        else:
            #no jackknife
            nd = np.sum(self.nd,axis=-1)
            nr = np.sum(self.nr,axis=-1)

            DD = np.sum(self.dd,axis=-1) / (nd * (nd - 1) / 2)
            RR = np.sum(self.rr,axis=-1) / (nr * (nr - 1) / 2)
            DR = np.sum(self.dr,axis=-1) / (nd * nr)

            self.jwprp = None
            self.varwprp = None

            self.wprp = (DD - 2 * DR + RR) / RR



    def visualize(self, plotname=None, f=None, ax=None, usecols=None,
                    usez=None, compare=False, **kwargs):

        if usecols is None:
            usecols = range(self.nlumbins)

        if usez is None:
            usez = range(self.nzbins)

        if f is None:
            f, ax = plt.subplots(len(usez), len(usecols), sharex=True,
                                    sharey=True, figsize=(8,8))
            ax = np.array(ax)
            ax = ax.reshape(len(usez), len(usecols))
            newaxes = True
        else:
            newaxes = False

        rmean = self.rbins[1:]+self.rbins[:-1]

        for i, l in enumerate(usecols):
            for j, z in enumerate(usez):
                l1 = ax[j][i].loglog(rmean, self.wprp[:,i,j])

        if newaxes:
            sax = f.add_subplot(111)
            sax.patch.set_alpha(0.0)
            sax.patch.set_facecolor('none')
            sax.spines['top'].set_color('none')
            sax.spines['bottom'].set_color('none')
            sax.spines['left'].set_color('none')
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            sax.set_xlabel(r'$w_{p}(r_{p})$')
            sax.set_ylabel(r'$r_{p} \, [ Mpc h^{-1}]$')

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax, l1


    def compare(self, othermetrics, plotname=None, usecols=None,
                 usez=None, labels=None, **kwargs):

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

        for i, m in enumerate(tocompare):
            if usecols[i] is not None:
                assert(len(usecols[0])==len(usecols[i]))
            if i==0:
                f, ax, l1 = m.visualize(usecols=usecols[i], usez=usez[i],
                                          compare=True,
                                          **kwargs)
            else:
                f, ax, l1 = m.visualize(usecols=usecols[i], usez=usez[i],
                                          compare=True,
                                          f=f, ax=ax, **kwargs)
            lines.append(l1)

        if labels[0]!=None:
            f.legend(lines, labels)

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax

class WPrpSnapshot(CorrelationFunction):

    def __init__(self, ministry, lumbins=None, rbins=None,
                  minr=None, maxr=None, logbins=True, nrbins=None,
                  pimax=None, catalog_type=None, tag=None,
                  lcutind=None, same_rand=False, inv_lum=True,
                  **kwargs):

        """
        Angular correlation function, w(theta), for use with non-periodic
        data. All angles should be specified in degrees.
        """
        CorrelationFunction.__init__(self, ministry, lightcone=False,
                                      lumbins=lumbins, nrbins=nrbins,
                                      lcutind=lcutind,
                                      same_rand=same_rand, inv_lum=inv_lum,
                                      catalog_type=catalog_type, tag=tag,
                                      **kwargs)

        self.logbins = logbins
        self.c = 299792.458

        if (rbins is None) & ((minr is None) | (maxr is None) | (nrbins is None)):
            self.minr = 1e-1
            self.maxr = 10
            self.nrbins = 15
            self.rbins = self.genbins(self.minr, self.maxr, self.nrbins)
        elif ((minr is not None) & (maxr is not None) & (nrbins is not None)):
            self.minr = minr
            self.maxr = maxr
            self.nrbins = nrbins
            self.rbins = self.genbins(minr, maxr, nrbins)
        else:
            self.rbins = rbins
            self.minr = rbins[0]
            self.maxr = rbins[1]
            self.nrbins = len(rbins)-1

        if pimax is None:
            self.pimax = 80.0
        else:
            self.pimax = pimax

        self.writeCorrfuncBinFile(self.rbins)
        #self.binfilename = '/anaconda/lib/python2.7/site-packages/Corrfunc/xi_mocks/tests/bins'

        self.mapkeys = ['px', 'py', 'pz', 'luminosity']
        self.unitmap = {'luminosity':'mag', 'px':'mpch', 'py':'mpch', 'pz':'mpch'}

    def map(self, mapunit):

        if not hascorrfunc:
            raise(ImportError("CorrFunc is required to calculate wp(rp)"))

        if not hasattr(self, 'wthetaj'):
            self.wprp = np.zeros((self.nrbins, self.nlumbins))

        for li, j in enumerate(self.luminds):
            print('Finding luminosity indices')
            lidx = (self.lumbins[j] <= mapunit['luminosity'][:,self.lcutind]) & (mapunit['luminosity'][:,self.lcutind] < self.lumbins[j+1])

            if not lidx.any():
                print("No galaxies in magnitude bin [{0},{1})".format(self.lumbins[j], self.lumbins[j+1]))
                continue

            wprp = countpairs.countpairs_wp(self.ministry.boxsize,
                                        self.pimax,
                                        1,
                                        self.binfilename,
                                        mapunit['px'][lidx],
                                        mapunit['py'][lidx],
                                        mapunit['pz'][lidx])

            self.wprp[:,li] = np.array([wprp[k][3] for k in range(self.nrbins)])

    def reduce(self):
        pass


    def visualize(self, plotname=None, f=None, ax=None, usecols=None,
                    usez=None, compare=False, **kwargs):

        if usecols is None:
            usecols = range(self.nlumbins)

        if usez is None:
            usez = [0]

        if f is None:
            f, ax = plt.subplots(len(usez), len(usecols), sharex=True,
                                    sharey=True, figsize=(8,8))
            ax = np.array(ax)
            ax = ax.reshape(len(usez), len(usecols))
            newaxes = True
        else:
            newaxes = False

        rmeans = self.rbins[1:]-self.rbins[:-1]

        for i, l in enumerate(usecols):
            print(len(rmeans))
            print(len(self.wprp[:,i]))
            l1 = ax[usez[0]][i].loglog(rmeans, self.wprp[:,i])

        if newaxes:
            sax = f.add_subplot(111)
            sax.patch.set_alpha(0.0)
            sax.patch.set_facecolor('none')
            sax.spines['top'].set_color('none')
            sax.spines['bottom'].set_color('none')
            sax.spines['left'].set_color('none')
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            sax.set_xlabel(r'$w_{p}(r_{p})$')
            sax.set_ylabel(r'$r_{p} \, [ Mpc h^{-1}]$')

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax, l1


    def compare(self, othermetrics, plotname=None, usecols=None,
                 labels=None, usez=None, **kwargs):

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

        for i, m in enumerate(tocompare):
            if usecols[i] is not None:
                assert(len(usecols[0])==len(usecols[i]))
            if i==0:
                f, ax, l1 = m.visualize(usecols=usecols[i], compare=True,
                                    **kwargs)
            else:
                f, ax, l1 = m.visualize(usecols=usecols[i], compare=True,
                                    f=f, ax=ax, **kwargs)
            lines.append(l1)

        if labels[0]!=None:
            f.legend(lines, labels)

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax

class TabulatedWPrpLightcone(CorrelationFunction):

    def __init__(self, fname, *args, **kwargs):

        self.fname = fname

        if 'rmeancol' in kwargs:
            self.rmeancol = int(kwargs.pop('rmeancol'))
        else:
            self.rmeancol = 0

        if 'wprpcol' in kwargs:
            self.wprpcol = int(kwargs.pop('wprpcol'))
        else:
            self.wprpcol = 1

        if 'wprperr' in kwargs:
            self.wprperr = int(kwargs.pop('wprperr'))
        else:
            self.wprperr = 2

        if 'ncuts' in kwargs:
            self.ncuts = int(kwargs.pop('ncuts'))
        else:
            self.ncuts = 1

        WPrpLightcone.__init__(self, *args, **kwargs)

        self.nomap = True
        self.loadWPrp()


    def loadWPrp(self):

        tab = np.genfromtxt(self.fname)
        self.wprp = np.zeros((tab.shape[0], self.ncuts))
        self.varwprp = np.zeros((tab.shape[0], self.ncuts))
        self.rmean = tab[:,self.rmeancol]


class GalaxyRadialProfileBCC(Metric):

    def __init__(self, ministry, zbins=None, lumbins=None, rbins=None,
                 massbins=None, subjack=False, catalog_type=['galaxycatalog'],
                 tag=None, **kwargs):
        """
        Radial profile of galaxies around their nearest halos.
        """

        Metric.__init__(self, ministry, tag=tag, **kwargs)

        self.catalog_type = catalog_type

        if zbins is None:
            self.zbins = [0.0, 0.2]
        else:
            self.zbins = zbins
            self.zbins = np.array(self.zbins)

        self.nzbins = len(self.zbins)-1

        if lumbins is None:
            self.lumbins = np.array([-22, -21, -20, -19])
        else:
            self.lumbins = lumbins

        self.nlumbins = len(self.lumbins)-1

        if rbins is None:
            self.rbins = np.logspace(-2, 1, 21)
        else:
            self.rbins = rbins

        self.nrbins = len(self.rbins)-1

        self.aschema = 'galaxyonly'

        self.mapkeys = ['luminosity', 'redshift', 'rhalo']
        self.unitmap = {'luminosity':'mag', 'polar_ang':'dec', 'azim_ang':'ra'}

    def map(self, mapunit):

        if not hasattr(self, 'rprof'):
            self.rprof = np.zeros((self.nrbins, self.nlumbins, self.nzbins))

        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])
            for j, l in enumerate(self.lumbins[:-1]):
                lidx = (self.lumbins[j]<mapunit['luminosity'][zlidx:zhidx,0]) & (mapunit['luminosity'][zlidx:zhidx,0]<self.lumbins[j+1])
                c, e = np.histogram(mapunit['rhalo'][zlidx:zhidx][lidx], bins=self.rbins)
                self.rprof[:,j,i] += c

    def reduce(self):
        if not hascorrfunc:
            return

        self.rmean = (self.rbins[1:]+self.rbins[:-1])/2
        vol = 4*np.pi*(self.rmean**3)/3

        for i in range(self.nzbins):
            for j in range(self.nlumbins):
                self.rprof[:,j,i] /= vol


    def visualize(self, plotname=None, f=None, ax=None, compare=False, **kwargs):
        if not hascorrfunc:
            return
        if f is None:
            f, ax = plt.subplots(self.nlumbins, self.nzbins,
                                 sharex=True, sharey=True,
                                 figsize=(8,8))
            newaxes = True
        else:
            newaxes = False

        if self.nzbins>1:
            for i in range(self.nlumbins):
                for j in range(self.nzbins):
                    ax[i][j].semilogx(self.rmean, self.rprof[:,i,j],
                                      **kwargs)
        else:
            for i in range(self.nlumbins):
                for j in range(self.nzbins):
                    ax[i].semilogx(self.rmean, self.rprof[:,i,j],
                                   **kwargs)

        if newaxes:
            sax = f.add_subplot(111)
            sax.patch.set_alpha(0.0)
            sax.patch.set_facecolor('none')
            sax.spines['top'].set_color('none')
            sax.spines['bottom'].set_color('none')
            sax.spines['left'].set_color('none')
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            sax.set_xlabel(r'$r\, [Mpc \, h^{-1}]$')
            sax.set_ylabel(r'$n \, [Mpc^{3} \, h^{-1}]$')

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax


    def compare(self):
        pass
