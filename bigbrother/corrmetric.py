from __future__ import print_function, division
#if __name__=='__main__':
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import fitsio
import warnings
import sys

try:
    import treecorr as tc
    hastreecorr = True
except:
    hastreecorr = False

try:
    import Corrfunc._countpairs_mocks as countpairs_mocks
    from Corrfunc.theory import DDrppi, DD
    from Corrfunc.mocks import DDtheta_mocks
    from Corrfunc.utils import convert_3d_counts_to_cf
    hascorrfunc = True
except:
    hascorrfunc = False

import numpy as np
import healpy as hp

from .metric import Metric, GMetric, jackknifeMap

class CorrelationFunction(Metric):

    def __init__(self, ministry, zbins=None, mbins=None,
                   nrbins=None, subjack=False, lightcone=True,
                   catalog_type=None, mcutind=None,
                   tag=None, same_rand=False, inv_m=True,
                   randname=None, upper_limit=False, **kwargs):
        """
        Generic correlation function.
        """
        Metric.__init__(self, ministry, tag=tag, **kwargs)

        if catalog_type is None:
            self.catalog_type = ['galaxycatalog']
        else:
            self.catalog_type = catalog_type

        self.lightcone = lightcone
        self.upper_limit = upper_limit

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


        if (mbins is None) & (self.catalog_type == ['galaxycatalog']):
            self.mbins = np.array([-30, 0])
        elif (mbins is None) & (self.catalog_type == ['halocatalog']):
            self.mbins = np.array([10**7, 10**17])
        else:
            self.mbins = mbins

        if self.upper_limit:
            self.nmbins = len(self.mbins)
        else:
            self.nmbins = len(self.mbins)-1

        if inv_m:
            self.minds = np.arange(self.nmbins)[::-1]
        else:
            self.minds = np.arange(self.nmbins)

        self.rand_ind = 0

        self.same_rand = same_rand
        self.inv_m = inv_m
        self.randname = randname

        if nrbins is None:
            self.nrbins = 15
        else:
            self.nrbins = nrbins

        self.subjack = subjack

        if 'galaxycatalog' in self.catalog_type:
            self.aschema = 'galaxygalaxy'
            self.mkey = 'luminosity'
        else:
            self.mkey = 'halomass'
            self.aschema = 'halohalo'

        self.mcutind = mcutind

        if self.subjack:
            raise NotImplementedError

        self.jsamples = 0

    def getRandoms(self, aza, pla, z=None):

        if self.randname is None:
            rand = self.generateAngularRandoms(aza, pla, z=z, nside=self.randnside)
        else:
            rand = self.readAngularRandoms(self.randname, len(aza), z=z)

        return rand

    def getCartesianRandoms(self, x, y, z, rand_fact=10):

        rsize = len(x)*rand_fact
        rdtype = np.dtype([('px', np.float32), ('py', np.float32),
                           ('pz', np.float32)])

        gr = np.zeros(rsize, dtype=rdtype)
        gr['px'] = np.random.uniform(low=np.min(x),
                                     high=np.max(x),
                                     size=rsize)
        gr['py'] = np.random.uniform(low=np.min(y),
                                     high=np.max(y),
                                     size=rsize)
        gr['pz'] = np.random.uniform(low=np.min(z),
                                     high=np.max(z),
                                     size=rsize)

        return gr


    def generateAngularRandoms(self, aza, pla, z=None, urand_factor=20,
                               rand_factor=10, nside=8, nest=True):
        """
        Generate a set of randoms from a catalog by pixelating the input
        catalog and uniformly distributing random points within the pixels
        occupied by galaxies.

        Also option to assign redshifts with the same distribution as the input
        catalog to allow for generation of randoms once for all z bins.
        """

        if z is not None:
            rdtype = np.dtype([('azim_ang', np.float64), ('polar_ang', np.float64),
                               ('redshift', np.float64)])
        else:
            rdtype = np.dtype([('azim_ang', np.float64), ('polar_ang', np.float64)])

        rsize = len(aza)*urand_factor
        rlen = 0
        ncycles = 0

        while rlen < len(aza)*rand_factor:

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
            rpix = hp.ang2pix(nside, (grand['polar_ang']+90)*np.pi/180, grand['azim_ang']*np.pi/180., nest=nest)
            inarea = np.in1d(rpix, ucpix)

            if ncycles == 0:
                gr = grand[inarea]
                rlen = len(gr)

            else:
                gr = np.hstack([gr, grand[inarea]])
                rlen = len(gr)

            ncycles +=1

        ridx = np.arange(rlen)
        ridx = np.random.choice(ridx, size=len(aza)*rand_factor)
        gr = gr[ridx]

        return gr

    def readAngularRandoms(self, fname, ngal, z=None, rand_factor=10):
        """
        Use randoms from a file
        """

        print('Reading randoms from file: {0}'.format(fname))

        try:
            r = np.genfromtxt(fname)
            if r.shape[1]<2:
                raise ValueError("Not enough columns in {0} to be a random catalog".format(fname))
            elif r.shape[1]>2:
                warnings.warn("More than 2 columns in {0}, assuming first two are ra and dec respectively.")

            rand = np.zeros(ngal*rand_factor, dtype=np.dtype([('azim_ang', np.float64), ('polar_ang', np.float64), ('redshift', np.float64)]))
            rand['azim_ang'] = np.random.choice(r[:,0], size=ngal*rand_factor)
            rand['polar_ang'] = np.random.choice(r[:,1], size=ngal*rand_factor)

        except Exception as e:
            print(e)
            try:
                r = fitsio.read(fname, columns=['RA', 'DEC'])

                rand = np.array(ngal*rand_factor, dtype=np.array([('azim_ang', np.float64), ('polar_ang', np.float64), ("redshift")]))
                rand['azim_ang'] = np.random.choice(r['RA'], size=ngal*rand_factor)
                rand['polar_ang'] = np.random.choice(r['DEC'], size=ngal*rand_factor)

            except ValueError as e:
                print(e)
                r = fitsio.read(fname, columns=['azim_ang', 'polar_ang'])

                rand = np.array(ngal*rand_factor, dtype=np.array([('azim_ang', np.float64), ('polar_ang', np.float64), ("redshift")]))
                rand['azim_ang'] = np.random.choice(r['azim_ang'], size=ngal*rand_factor)
                rand['polar_ang'] = np.random.choice(r['polar_ang'], size=ngal*rand_factor)

        if z is not None:
            rand['redshift'] = np.random.choice(z, size=ngal*rand_factor)

        return rand


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

    def __init__(self, ministry, zbins=None, mbins=None,
                  abins=None, mintheta=None, maxtheta=None,
                  logbins=True, nabins=None, subjack=False,
                  catalog_type=None, tag=None, mcutind=None,
                  same_rand=False, inv_m=True, cosmology_flag=None,
                  bimodal_ccut=False, percentile_ccut=None,
                  precompute_color=False, upper_limit=False,
                  centrals_only=False, rsd=False,
                  randnside=None, **kwargs):
        """
        Angular correlation function, w(theta), for use with non-periodic
        data.
        """
        CorrelationFunction.__init__(self, ministry, zbins=zbins,
                                      lightcone=True, mbins=mbins,
                                      nrbins=nabins, subjack=subjack,
                                      mcutind=mcutind, same_rand=same_rand,
                                      inv_m=inv_m,catalog_type=catalog_type,
                                      tag=tag, upper_limit=upper_limit, **kwargs)

        self.bimodal_ccut = bimodal_ccut
        self.percentile_ccut = percentile_ccut
        self.splitcolor = None

        if self.bimodal_ccut:
            self.hcbins = 100
            self.ncbins = 2
        elif self.percentile_ccut is not None:
            self.ncbins = 2
        else:
            self.ncbins = 1

        self.pccolor = precompute_color
        self.centrals_only = centrals_only

        self.logbins = logbins
        self.c = 299792.458

        if (abins is None) & ((mintheta is None) | (maxtheta is None) | (nabins is None)):
            self.mintheta = 1e-2
            self.maxtheta = 1
            self.nabins = 30
            self.abins = self.genbins(self.mintheta, self.maxtheta, self.nabins)
        elif ((mintheta is not None) & (maxtheta is not None) & (nabins is not None)):
            self.mintheta = mintheta
            self.maxtheta = maxtheta
            self.nabins = nabins
            self.abins = self.genbins(mintheta, maxtheta, nabins)
        else:
            self.abins = abins
            self.mintheta = abins[0]
            self.maxtheta = abins[-1]
            self.nabins = len(abins)-1

        if randnside is None:
            self.randnside = 128
        else:
            self.randnside = randnside

        if cosmology_flag is None:
            self.cosmology_flag = 2
        else:
            self.cosmology_flag = cosmology_flag

        self.jcount = 0

        self.writeCorrfuncBinFile(self.abins, binfilename='angular_bins')
        #self.binfilename = '/anaconda/lib/python2.7/site-packages/Corrfunc/xi_mocks/tests/bins'

        self.mapkeys = [self.mkey, 'redshift', 'polar_ang', 'azim_ang']

        if self.catalog_type == ['galaxycatalog']:
            self.unitmap = {'luminosity':'mag', 'polar_ang':'dec', 'azim_ang':'ra', 'redshift':'z'}
        elif self.catalog_type == ['halocatalog']:
            self.unitmap = {'halomass':'msunh', 'polar_ang':'dec', 'azim_ang':'ra', 'redshift':'z'}

        self.rsd = rsd

        if self.rsd:
            self.mapkeys.append('velocity')
            self.unitmap['velocity'] = 'kms'

        if self.centrals_only:
            self.mapkeys.append('central')

        if self.pccolor:
            self.mapkeys.append('color')

        self.rand_ind = 0

        self.nd = None
        self.nr = None
        self.dd = None
        self.dr = None
        self.rr = None

    @jackknifeMap
    def map(self, mapunit):
        if not hastreecorr:
            return

        self.jsamples += 1

        if self.nd is None:
            self.nd = np.zeros((self.njack, self.ncbins, self.nmbins,
                                  self.nzbins))
            self.nr = np.zeros((self.njack, self.ncbins, self.nmbins,
                                  self.nzbins))
            self.dd = np.zeros((self.njack, self.nabins, self.ncbins, 
                                  self.nmbins, self.nzbins))
            self.dr = np.zeros((self.njack, self.nabins, self.ncbins, 
                                  self.nmbins, self.nzbins))
            self.rr = np.zeros((self.njack, self.nabins, self.ncbins, 
                                  self.nmbins, self.nzbins))

        if (mapunit['azim_ang'].dtype == '>f4') | (mapunit['azim_ang'].dtype == '>f8') | (mapunit['azim_ang'].dtype == np.float64):
            mu = {}
            mu['azim_ang'] = np.zeros(len(mapunit['azim_ang']), dtype=np.float64)
            mu['polar_ang'] = np.zeros(len(mapunit['polar_ang']), dtype=np.float64)
            mu['redshift'] = np.zeros(len(mapunit['redshift']), dtype=np.float64)

            mu['azim_ang'][:] = mapunit['azim_ang'][:]
            mu['polar_ang'][:] = mapunit['polar_ang'][:]
            mu['redshift'][:] = mapunit['redshift'][:]
            mu[self.mkey] = mapunit[self.mkey]

            if self.rsd:
                mu['velocity'] = np.zeros((len(mapunit['velocity']),3), dtype=np.float64)
                mu['velocity'][:] = mapunit['velocity'][:]
        else:
            mu = mapunit

        if self.rsd:
            z = self.addRSD(mu) / self.c
        else:
            z = mu['redshift']

        #calculate DD
        for i in range(self.nzbins):
            print('Finding redshift indices')

            zlidx = z.searchsorted(self.zbins[i])
            zhidx = z.searchsorted(self.zbins[i+1])

            if zlidx==zhidx:
                print("No galaxies in redshift bin {0} to {1}".format(self.zbins[i], self.zbins[i+1]))
                print("Min and max z: {0}, {1}".format(np.min(z), np.max(z)))
                print(z)
                continue

            if (self.splitcolor is None) & (self.bimodal_ccut):
                ccounts, cbins = np.histogram(clr[zlidx:zhidx], self.hcbins)
                self.splitcolor = self.splitBimodal(cbins[:-1], ccounts)

            for li, j in enumerate(self.minds):
                print('Finding luminosity indices')

                if self.mcutind is not None:
                    if self.upper_limit:
                        lidx = mu[self.mkey][zlidx:zhidx,self.mcutind] < self.mbins[j]
                    else:
                        lidx = (self.mbins[j] <= mu[self.mkey][zlidx:zhidx,self.mcutind]) & (mu[self.mkey][zlidx:zhidx,self.mcutind] < self.mbins[j+1])
                else:
                    if self.upper_limit:
                        lidx = mu[self.mkey][zlidx:zhidx] < self.mbins[j]
                    else:
                        lidx = (self.mbins[j] <= mu[self.mkey][zlidx:zhidx]) & (mu[self.mkey][zlidx:zhidx] < self.mbins[j+1])

                if self.centrals_only:
                    lidx = lidx & (mu['central'][zlidx:zhidx]==1)

                if (li==self.rand_ind) | (not self.same_rand):
                    print('Generating Randoms')
                    if len(z[zlidx:zhidx][lidx])==0:
                        self.rand_ind+=1
                        continue

                    rands = self.getRandoms(mu['azim_ang'][zlidx:zhidx][lidx], mu['polar_ang'][zlidx:zhidx][lidx])

                if (self.percentile_ccut is not None):
                    self.splitcolor = self.splitPercentile(clr[zlidx:zhidx], self.percentile_ccut)

                for k in range(self.ncbins):
                    if self.ncbins == 1:
                        cidx = lidx
                    else:
                        if k==0:
                            cidx = lidx & (self.splitcolor < clr[zlidx:zhidx])
                        else:
                            cidx = lidx & (self.splitcolor >= clr[zlidx:zhidx])

                    self.nd[self.jcount,k,j,i] = len(mu['azim_ang'][zlidx:zhidx][cidx])
                    self.nr[self.jcount,k,j,i] = len(rands)

                    print("Number of galaxies in this z/lum bin: {0}".format(self.nd[self.jcount,k,j,i]))
                    print("Number of randoms in this z/lum bin: {0}".format(self.nr[self.jcount,k,j,i]))

                    #data data
                    print('calculating data data pairs')
                    sys.stdout.flush()
                    if (self.nd[self.jcount,k,j,i]<2) | (self.nr[self.jcount,k,j,i]<2):
                        continue

                    ddresults = DDtheta_mocks(1, 1, self.binfilename,
                                            mu['azim_ang'][zlidx:zhidx][cidx],
                                            mu['polar_ang'][zlidx:zhidx][cidx],
                                            mu['azim_ang'][zlidx:zhidx][cidx],
                                            mu['polar_ang'][zlidx:zhidx][cidx])


                    self.dd[self.jcount,:,k,j,i] = ddresults['npairs']

                    #data randoms
                    print('calculating data random pairs')
                    sys.stdout.flush()
                    drresults = DDtheta_mocks(0, 1, self.binfilename,
                                            mu['azim_ang'][zlidx:zhidx][cidx],
                                            mu['polar_ang'][zlidx:zhidx][cidx],
                                            rands['azim_ang'],
                                            rands['polar_ang'])

                    self.dr[self.jcount,:,k,j,i] = drresults['npairs']

                    #randoms randoms
                    print('calculating random random pairs')
                    sys.stdout.flush()
                    if (li==0) | (not self.same_rand):
                        rrresults = DDtheta_mocks(1, 1, self.binfilename,
                                                rands['azim_ang'],
                                                rands['polar_ang'],
                                                rands['azim_ang'],
                                                rands['polar_ang'])


                    self.rr[self.jcount,:,k,j,i] = rrresults['npairs']


    def reduce(self, rank=None, comm=None):

        if rank is not None:
            gnd = comm.gather(self.nd, root=0)
            gnr = comm.gather(self.nr, root=0)
            gdd = comm.gather(self.dd, root=0)
            gdr = comm.gather(self.dr, root=0)
            grr = comm.gather(self.rr, root=0)


            if rank==0:
                ndshape = [self.nd.shape[i] for i in range(len(self.nd.shape))]
                nrshape = [self.nr.shape[i] for i in range(len(self.nr.shape))]
                ddshape = [self.dd.shape[i] for i in range(len(self.dd.shape))]
                drshape = [self.dr.shape[i] for i in range(len(self.dr.shape))]
                rrshape = [self.rr.shape[i] for i in range(len(self.rr.shape))]

                ndshape.insert(1,1)
                nrshape.insert(1,1)

                ndshape[0] = self.njacktot
                nrshape[0] = self.njacktot
                ddshape[0] = self.njacktot
                drshape[0] = self.njacktot
                rrshape[0] = self.njacktot

                self.nd = np.zeros(ndshape)
                self.nr = np.zeros(nrshape)
                self.dd = np.zeros(ddshape)
                self.dr = np.zeros(drshape)
                self.rr = np.zeros(rrshape)

                jc = 0
                for i, g in enumerate(gnd):
                    if g is None: continue
                    nj = g.shape[0]
                    self.nd[jc:jc+nj,0,:,:] = g
                    self.nr[jc:jc+nj,0,:,:] = gnr[i]
                    self.dd[jc:jc+nj,:,:,:] = gdd[i]
                    self.dr[jc:jc+nj,:,:,:] = gdr[i]
                    self.rr[jc:jc+nj,:,:,:] = grr[i]

                    jc += nj

                self.jwtheta = np.zeros(self.dd.shape)

                self.jnd = self.jackknife(self.nd, reduce_jk=False)
                self.jnr = self.jackknife(self.nr, reduce_jk=False)
                self.jdd = self.jackknife(self.dd, reduce_jk=False)
                self.jdr = self.jackknife(self.dr, reduce_jk=False)
                self.jrr = self.jackknife(self.rr, reduce_jk=False)

                fnorm = self.jnr / self.jnd

                self.jwtheta = (fnorm ** 2 * self.jdd - 2 * fnorm * self.jdr + self.jrr) / self.jrr

                self.wtheta = np.sum(self.jwtheta, axis=0) / self.njacktot

                self.varwtheta = np.sum((self.jwtheta - self.wtheta)**2, axis=0) * (self.njacktot - 1) / self.njacktot

        else:
            self.jwprp = np.zeros(self.dd.shape)

            self.jnd = self.jackknife(self.nd, reduce_jk=False)
            self.jnr = self.jackknife(self.nr, reduce_jk=False)
            self.jdd = self.jackknife(self.dd, reduce_jk=False)
            self.jdr = self.jackknife(self.dr, reduce_jk=False)
            self.jrr = self.jackknife(self.rr, reduce_jk=False)

            fnorm = self.jnr / self.jnd

            self.jwtheta = (fnorm ** 2 * self.jdd - 2 * fnorm * self.jdr + self.jrr) / self.jrr

            self.wtheta = np.sum(self.jwtheta, axis=0) / self.njacktot

            self.varwtheta = np.sum((self.jwtheta - self.wtheta)**2, axis=0) * (self.njacktot - 1) / self.njacktot

    def visualize(self, plotname=None, f=None, ax=None, usecols=None,
                    usez=None, compare=False, usecolors=None, **kwargs):

        if usecols is None:
            usecols = range(self.nmbins)

        if usez is None:
            usez = range(self.nzbins)

        if usecolors is None:
            usecolors = range(self.ncbins)

        if f is None:
            f, ax = plt.subplots(len(usez), len(usecols), sharex=True,
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
            for j, z in enumerate(usez):
                for k, c in enumerate(usecolors):
                    ye = np.sqrt(self.varwtheta[:,c,l,z])
                    l1 = ax[j][i].plot(rmean, self.wtheta[:,c,l,z], **kwargs)
                    ax[j][i].fill_between(rmean, self.wtheta[:,c,l,z]-ye, self.wtheta[:,c,l,z]+ye, alpha=0.5, **kwargs)

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
            sax.set_ylabel(r'$w(\theta)$', labelpad=20, fontsize=16)
            sax.set_xlabel(r'$\theta \, [ degrees ]$',labelpad=20, fontsize=16)

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax, l1[0]


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
                                          compare=True, color=Metric._color_list[i],
                                          **kwargs)
            else:
                f, ax, l1 = m.visualize(usecols=usecols[i], usez=usez[i],
                                          compare=True, color=Metric._color_list[i],
                                          f=f, ax=ax, **kwargs)
            lines.append(l1)

        if labels[0]!=None:
            f.legend(lines, labels)

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax


class WPrpLightcone(CorrelationFunction):

    def __init__(self, ministry, zbins=None, mbins=None,
                  rbins=None, minr=None, maxr=None, logbins=True,
                  nrbins=None, pimax=None, subjack=False,
                  catalog_type=None, tag=None, mcutind=None,
                  same_rand=False, inv_m=True, cosmology_flag=None,
                  bimodal_ccut=False, percentile_ccut=None,
                  precompute_color=False, upper_limit=False,
                  centrals_only=False, rsd=False,
                  randnside=None,
                  **kwargs):
        """
        Projected correlation function, wp(rp), for use with non-periodic
        data.
        """
        CorrelationFunction.__init__(self, ministry, zbins=zbins,
                                      lightcone=True, mbins=mbins,
                                      nrbins=nrbins, subjack=subjack,
                                      mcutind=mcutind, same_rand=same_rand,
                                      inv_m=inv_m,catalog_type=catalog_type,
                                      tag=tag, upper_limit=upper_limit, **kwargs)

        self.bimodal_ccut = bimodal_ccut
        self.percentile_ccut = percentile_ccut
        self.splitcolor = None

        if self.bimodal_ccut:
            self.hcbins = 100
            self.ncbins = 2
        elif self.percentile_ccut is not None:
            self.ncbins = 2
        else:
            self.ncbins = 1

        self.pccolor = precompute_color
        self.centrals_only = centrals_only

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

        if randnside is None:
            self.randnside = 128
        else:
            self.randnside = randnside

        if cosmology_flag is None:
            self.cosmology_flag = 2
        else:
            self.cosmology_flag = cosmology_flag

        if pimax is None:
            self.pimax = 80.0
        else:
            self.pimax = pimax

        self.jcount = 0

        self.writeCorrfuncBinFile(self.rbins)
        #self.binfilename = '/anaconda/lib/python2.7/site-packages/Corrfunc/xi_mocks/tests/bins'

        self.mapkeys = [self.mkey, 'redshift', 'polar_ang', 'azim_ang']

        if self.catalog_type == ['galaxycatalog']:
            self.unitmap = {'luminosity':'mag', 'polar_ang':'dec', 'azim_ang':'ra', 'redshift':'z'}
        elif self.catalog_type == ['halocatalog']:
            self.unitmap = {'halomass':'msunh', 'polar_ang':'dec', 'azim_ang':'ra', 'redshift':'z'}

        self.rsd = rsd

        if self.rsd:
            self.mapkeys.append('velocity')
            self.unitmap['velocity'] = 'kms'

        if self.centrals_only:
            self.mapkeys.append('central')

        if self.pccolor:
            self.mapkeys.append('color')

        self.rand_ind = 0

        self.nd = None
        self.nr = None
        self.dd = None
        self.dr = None
        self.rr = None


    def addRSD(self, mapunit):

        rvec = hp.ang2vec(-(mapunit['polar_ang'] - 90) * np.pi / 180.,
                            mapunit['azim_ang'] * np.pi / 180 )
        vr   = np.sum(rvec * mapunit['velocity'], axis=1)

        return vr + mapunit['redshift']*self.c


    @jackknifeMap
    def map(self, mapunit):

        if not hascorrfunc:
            raise(ImportError("CorrFunc is required to calculate wp(rp)"))

        if (self.ncbins > 1) & (~self.pccolor):
            clr = mapunit['luminosity'][:,0] - mapunit['luminosity'][:,1]
        elif self.pccolor:
            clr = mapunit['color']

        if self.dd is None:
            self.dd = np.zeros((self.njack, self.nrbins, int(self.pimax), self.ncbins, self.nmbins, self.nzbins))
            self.dr = np.zeros((self.njack, self.nrbins, int(self.pimax), self.ncbins, self.nmbins, self.nzbins))
            self.rr = np.zeros((self.njack, self.nrbins, int(self.pimax), self.ncbins, self.nmbins, self.nzbins))
            self.nd = np.zeros((self.njack, self.ncbins, self.nmbins, self.nzbins))
            self.nr = np.zeros((self.njack, self.ncbins, self.nmbins, self.nzbins))


        if (mapunit['azim_ang'].dtype == '>f4') | (mapunit['azim_ang'].dtype == '>f8') | (mapunit['azim_ang'].dtype == np.float64):
            mu = {}
            mu['azim_ang'] = np.zeros(len(mapunit['azim_ang']), dtype=np.float64)
            mu['polar_ang'] = np.zeros(len(mapunit['polar_ang']), dtype=np.float64)
            mu['redshift'] = np.zeros(len(mapunit['redshift']), dtype=np.float64)

            mu['azim_ang'][:] = mapunit['azim_ang'][:]
            mu['polar_ang'][:] = mapunit['polar_ang'][:]
            mu['redshift'][:] = mapunit['redshift'][:]
            mu[self.mkey] = mapunit[self.mkey]

            if self.rsd:
                mu['velocity'] = np.zeros((len(mapunit['velocity']),3), dtype=np.float64)
                mu['velocity'][:] = mapunit['velocity'][:]
        else:
            mu = mapunit

        if self.rsd:
            cz = self.addRSD(mu)
        else:
            cz = mu['redshift'] * self.c


        #calculate DD
        for i in range(self.nzbins):
            print('Finding redshift indices')

            zlidx = mu['redshift'].searchsorted(self.zbins[i])
            zhidx = mu['redshift'].searchsorted(self.zbins[i+1])

            if zlidx==zhidx:
                print("No galaxies in redshift bin {0} to {1}".format(self.zbins[i], self.zbins[i+1]))
                print("Min and max z: {0}, {1}".format(np.min(mu['redshift']), np.max(mu['redshift'])))
                print(mu['redshift'])
                continue

            if (self.splitcolor is None) & (self.bimodal_ccut):
                ccounts, cbins = np.histogram(clr[zlidx:zhidx], self.hcbins)
                self.splitcolor = self.splitBimodal(cbins[:-1], ccounts)

            for li, j in enumerate(self.minds):
                print('Finding luminosity indices')

                if self.mcutind is not None:
                    if self.upper_limit:
                        lidx = mu[self.mkey][zlidx:zhidx,self.mcutind] < self.mbins[j]
                    else:
                        lidx = (self.mbins[j] <= mu[self.mkey][zlidx:zhidx,self.mcutind]) & (mu[self.mkey][zlidx:zhidx,self.mcutind] < self.mbins[j+1])
                else:
                    if self.upper_limit:
                        lidx = mu[self.mkey][zlidx:zhidx] < self.mbins[j]
                    else:
                        lidx = (self.mbins[j] <= mu[self.mkey][zlidx:zhidx]) & (mu[self.mkey][zlidx:zhidx] < self.mbins[j+1])

                if self.centrals_only:
                    lidx = lidx & (mu['central'][zlidx:zhidx]==1)

                if (li==self.rand_ind) | (not self.same_rand):
                    print('Generating Randoms')
                    if len(cz[zlidx:zhidx][lidx])==0:
                        self.rand_ind+=1
                        continue

                    rands = self.getRandoms(mu['azim_ang'][zlidx:zhidx][lidx], mu['polar_ang'][zlidx:zhidx][lidx], z=cz[zlidx:zhidx][lidx])

                if (self.percentile_ccut is not None):
                    self.splitcolor = self.splitPercentile(clr[zlidx:zhidx], self.percentile_ccut)


                for k in range(self.ncbins):
                    if self.ncbins == 1:
                        cidx = lidx
                    else:
                        if k==0:
                            cidx = lidx & (self.splitcolor < clr[zlidx:zhidx])
                        else:
                            cidx = lidx & (self.splitcolor >= clr[zlidx:zhidx])

                    self.nd[self.jcount,k,j,i] = len(mu['azim_ang'][zlidx:zhidx][cidx])
                    self.nr[self.jcount,k,j,i] = len(rands)

                    print("Number of galaxies in this z/lum bin: {0}".format(self.nd[self.jcount,k,j,i]))
                    print("Number of randoms in this z/lum bin: {0}".format(self.nr[self.jcount,k,j,i]))

                    #data data
                    print('calculating data data pairs')
                    sys.stdout.flush()
                    if (self.nd[self.jcount,k,j,i]<2) | (self.nr[self.jcount,k,j,i]<2):
                        continue

                    ddresults = countpairs_mocks.countpairs_rp_pi_mocks(1,
                                            self.cosmology_flag, 1,
                                            self.pimax,
                                            self.binfilename,
                                            mu['azim_ang'][zlidx:zhidx][cidx],
                                            mu['polar_ang'][zlidx:zhidx][cidx],
                                            cz[zlidx:zhidx][cidx],
                                            mu['azim_ang'][zlidx:zhidx][cidx],
                                            mu['polar_ang'][zlidx:zhidx][cidx],
                                            cz[zlidx:zhidx][cidx])

                    ddresults = np.array(ddresults[0]).reshape(-1,int(self.pimax),5)
                    self.dd[self.jcount,:,:,k,j,i] = ddresults[:,:,4]

                    #data randoms
                    print('calculating data random pairs')
                    sys.stdout.flush()
                    drresults = countpairs_mocks.countpairs_rp_pi_mocks(0, 1, 1,
                                            self.pimax,
                                            self.binfilename,
                                            mu['azim_ang'][zlidx:zhidx][cidx],
                                            mu['polar_ang'][zlidx:zhidx][cidx],
                                            cz[zlidx:zhidx][cidx],
                                            rands['azim_ang'],
                                            rands['polar_ang'],
                                            rands['redshift'])

                    drresults = np.array(drresults[0]).reshape(-1,int(self.pimax),5)
                    self.dr[self.jcount,:,:,k,j,i] = drresults[:,:,4]

                    #randoms randoms
                    print('calculating random random pairs')
                    sys.stdout.flush()
                    if (li==0) | (not self.same_rand):
                        rrresults = countpairs_mocks.countpairs_rp_pi_mocks(1, 1, 1,
                                            self.pimax,
                                            self.binfilename,
                                            rands['azim_ang'],
                                            rands['polar_ang'],
                                            rands['redshift'],
                                            rands['azim_ang'],
                                            rands['polar_ang'],
                                            rands['redshift'])

                        try:
                            rrresults = np.array(rrresults[0]).reshape(-1,int(self.pimax),5)
                        except:
                            raise

                    self.rr[self.jcount,:,:,k,j,i] = rrresults[:,:,4]

    def reduce(self, rank=None, comm=None):

        if rank is not None:
            gnd = comm.gather(self.nd, root=0)
            gnr = comm.gather(self.nr, root=0)
            gdd = comm.gather(self.dd, root=0)
            gdr = comm.gather(self.dr, root=0)
            grr = comm.gather(self.rr, root=0)


            if rank==0:
                ndshape = [self.nd.shape[i] for i in range(len(self.nd.shape))]
                nrshape = [self.nr.shape[i] for i in range(len(self.nr.shape))]
                ddshape = [self.dd.shape[i] for i in range(len(self.dd.shape))]
                drshape = [self.dr.shape[i] for i in range(len(self.dr.shape))]
                rrshape = [self.rr.shape[i] for i in range(len(self.rr.shape))]

                ndshape.insert(1,1)
                ndshape.insert(1,1)
                nrshape.insert(1,1)
                nrshape.insert(1,1)

                ndshape[0] = self.njacktot
                nrshape[0] = self.njacktot
                ddshape[0] = self.njacktot
                drshape[0] = self.njacktot
                rrshape[0] = self.njacktot

                self.nd = np.zeros(ndshape)
                self.nr = np.zeros(nrshape)
                self.dd = np.zeros(ddshape)
                self.dr = np.zeros(drshape)
                self.rr = np.zeros(rrshape)

                jc = 0
                for i, g in enumerate(gnd):
                    if g is None: continue
                    nj = g.shape[0]
                    self.nd[jc:jc+nj,0,0,:,:,:] = g
                    self.nr[jc:jc+nj,0,0,:,:,:] = gnr[i]
                    self.dd[jc:jc+nj,:,:,:,:,:] = gdd[i]
                    self.dr[jc:jc+nj,:,:,:,:,:] = gdr[i]
                    self.rr[jc:jc+nj,:,:,:,:,:] = grr[i]

                    jc += nj

                self.jwprp = np.zeros(self.dd.shape)

                self.jnd = self.jackknife(self.nd, reduce_jk=False)
                self.jnr = self.jackknife(self.nr, reduce_jk=False)
                self.jdd = self.jackknife(self.dd, reduce_jk=False)
                self.jdr = self.jackknife(self.dr, reduce_jk=False)
                self.jrr = self.jackknife(self.rr, reduce_jk=False)

                fnorm = self.jnr / self.jnd

                self.jwprppi = (fnorm ** 2 * self.jdd - 2 * fnorm * self.jdr + self.jrr) / self.jrr
                self.jwprp = 2 * np.sum(self.jwprppi, axis=2)

                self.wprppi = np.sum(self.jwprppi, axis=0) / self.njacktot
                self.wprp = np.sum(self.jwprp, axis=0) / self.njacktot

                self.varwprppi = np.sum((self.jwprppi - self.wprppi)**2, axis=0) * (self.njacktot - 1) / self.njacktot
                self.varwprp = np.sum((self.jwprp - self.wprp)**2, axis=0) * (self.njacktot - 1) / self.njacktot
        else:
            self.jwprp = np.zeros(self.dd.shape)

            self.jnd = self.jackknife(self.nd, reduce_jk=False)
            self.jnr = self.jackknife(self.nr, reduce_jk=False)
            self.jdd = self.jackknife(self.dd, reduce_jk=False)
            self.jdr = self.jackknife(self.dr, reduce_jk=False)
            self.jrr = self.jackknife(self.rr, reduce_jk=False)

            fnorm = self.jnr / self.jnd

            self.jwprppi = (fnorm ** 2 * self.jdd - 2 * fnorm * self.jdr + self.jrr) / self.jrr
            self.jwprp = 2 * np.sum(self.jwprppi, axis=2)

            self.wprppi = np.sum(self.jwprppi, axis=0) / self.njacktot
            self.wprp = np.sum(self.jwprp, axis=0) / self.njacktot

            self.varwprppi = np.sum((self.jwprppi - self.wprppi)**2, axis=0) * (self.njacktot - 1) / self.njacktot
            self.varwprp = np.sum((self.jwprp - self.wprp)**2, axis=0) * (self.njacktot - 1) / self.njacktot

    def visualize(self, plotname=None, f=None, ax=None, usecols=None,
                    usez=None, compare=False, usecolors=None, **kwargs):

        if usecols is None:
            usecols = range(self.nmbins)

        if usez is None:
            usez = range(self.nzbins)

        if usecolors is None:
            usecolors = range(self.ncbins)

        if f is None:
            f, ax = plt.subplots(len(usez), len(usecols), sharex=True,
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
            rmean = (self.rbins[1:]+self.rbins[:-1]) / 2

        for i, l in enumerate(usecols):
            for j, z in enumerate(usez):
                for k, c in enumerate(usecolors):
                    ye = np.sqrt(self.varwprp[:,c,l,z])
                    l1 = ax[j][i].plot(rmean, self.wprp[:,c,l,z], **kwargs)
                    ax[j][i].fill_between(rmean, self.wprp[:,c,l,z]-ye, self.wprp[:,c,l,z]+ye, alpha=0.5, **kwargs)

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
            sax.set_ylabel(r'$w_{p}(r_{p})$', labelpad=20)
            sax.set_xlabel(r'$r_{p} \, [ Mpc h^{-1}]$',labelpad=20)

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax, l1[0]


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
                                          compare=True, color=Metric._color_list[i],
                                          **kwargs)
            else:
                f, ax, l1 = m.visualize(usecols=usecols[i], usez=usez[i],
                                          compare=True, color=Metric._color_list[i],
                                          f=f, ax=ax, **kwargs)
            lines.append(l1)

        if labels[0]!=None:
            f.legend(lines, labels)

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax

class WPrpSnapshot(CorrelationFunction):

    def __init__(self, ministry, mbins=None, rbins=None,
                  minr=None, maxr=None, logbins=True, nrbins=None,
                  pimax=None, catalog_type=None, tag=None,
                  mcutind=None, same_rand=False, inv_m=True,
                  rsd=False, upper_limit=False, **kwargs):

        """
        Angular correlation function, w(theta), for use with non-periodic
        data. All angles should be specified in degrees.
        """
        CorrelationFunction.__init__(self, ministry, lightcone=False,
                                      mbins=mbins, nrbins=nrbins,
                                      mcutind=mcutind, upper_limit=upper_limit,
                                      same_rand=same_rand, inv_m=inv_m,
                                      catalog_type=catalog_type, tag=tag,
                                      **kwargs)

        self.logbins = logbins
        self.c = 299792.458

        if (rbins is None) & ((minr is None) | (maxr is None) | (nrbins is None)):
            self.minr = 1e-1
            self.maxr = 25
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

        self.rsd = rsd

        self.mapkeys = ['px', 'py', 'pz', self.mkey]
        self.unitmap = {'px':'mpch', 'py':'mpch', 'pz':'mpch'}
        if self.mkey == 'luminosity':
            self.unitmap[self.mkey] = 'mag'
        else:
            self.unitmap[self.mkey] = 'msunh'

        if self.rsd:
            self.mapkeys.append('velocity')
            self.unitmap['velocity'] = 'kms'

        self.nd = None
        self.nr = None
        self.dd = None
        self.dr = None
        self.rr = None


    def addRSD(self, mapunit):

        vr = mapunit['velocity'][:,2]

        return vr + mapunit['pz']


    @jackknifeMap
    def map(self, mapunit):

        if not hascorrfunc:
            raise(ImportError("CorrFunc is required to calculate wp(rp)"))

        if self.dd is None:
            self.dd = np.zeros((self.njack, self.nrbins, int(self.pimax), self.nmbins))
            self.dr = np.zeros((self.njack, self.nrbins, int(self.pimax), self.nmbins))
            self.rr = np.zeros((self.njack, self.nrbins, int(self.pimax), self.nmbins))
            self.nd = np.zeros((self.njack, self.nmbins))
            self.nr = np.zeros((self.njack, self.nmbins))

        if (mapunit['px'].dtype == '>f4') | (mapunit['px'].dtype == '>f8') | (mapunit['px'].dtype == np.float64):
            mu = {}
            mu['px'] = np.zeros(len(mapunit['px']), dtype=np.float32)
            mu['py'] = np.zeros(len(mapunit['py']), dtype=np.float32)
            mu['pz'] = np.zeros(len(mapunit['pz']), dtype=np.float32)

            mu['px'][:] = mapunit['px'][:]
            mu['py'][:] = mapunit['py'][:]
            mu['pz'][:] = mapunit['pz'][:]
            mu[self.mkey] = mapunit[self.mkey]

            if self.rsd:
                mu['velocity'] = np.zeros((len(mapunit['velocity']),3), dtype=np.float32)
                mu['velocity'][:] = mapunit['velocity'][:]
        else:
            mu = mapunit

        if self.rsd:
            cz = self.addRSD(mu)
        else:
            cz = mu['pz']

        for li, i in enumerate(self.minds):
            print('Finding luminosity indices')
            if self.mcutind is not None:
                if self.upper_limit:
                    lidx = mu[self.mkey][:,self.mcutind] < self.mbins[i]
                else:
                    lidx = (self.mbins[i] <= mu[self.mkey][:,self.mcutind]) & (mu[self.mkey][:,self.mcutind] < self.mbins[i+1])
            else:
                if self.upper_limit:
                    lidx = mu[self.mkey] < self.mbins[i]
                else:
                    lidx = (self.mbins[i] <= mu[self.mkey]) & (mu[self.mkey] < self.mbins[i+1])

            if (li==self.rand_ind) | (not self.same_rand):
                print('Generating Randoms')
                if len(cz[lidx])==0:
                    self.rand_ind+=1
                    continue

                rands = self.getCartesianRandoms(mu['px'][lidx], mu['py'][lidx], cz[lidx])

            self.nd[self.jcount,i] = len(cz[lidx])
            self.nr[self.jcount,i] = len(rands)

            print("Number of galaxies in this z/lum bin: {0}".format(self.nd[self.jcount,i]))
            print("Number of randoms in this z/lum bin: {0}".format(self.nr[self.jcount,i]))

            if self.nd[self.jcount,i]<2: continue
            #data data
            print('calculating data data pairs')
            sys.stdout.flush()

            ddout = DDrppi(1,1,self.pimax,
                              self.binfilename,
                              mu['px'][lidx],
                              mu['py'][lidx],
                              mu['pz'][lidx],
                              False,
                              mu['px'][lidx],
                              mu['py'][lidx],
                              mu['pz'][lidx])

            ddout = np.array(ddout)
            ddout = ddout.reshape(-1,int(self.pimax))
            self.dd[self.jcount,:,:,i] = ddout['npairs']

            print('calculating data random pairs')
            sys.stdout.flush()

            drout = DDrppi(0,1,self.pimax,
                              self.binfilename,
                              mu['px'][lidx],
                              mu['py'][lidx],
                              mu['pz'][lidx],
                              False,
                              rands['px'],
                              rands['py'],
                              rands['pz'])

            drout = np.array(drout)
            drout = drout.reshape(-1,int(self.pimax))
            self.dr[self.jcount,:,:,i] = drout['npairs']

            print('calculating random random pairs')
            sys.stdout.flush()

            if (li==0) | (not self.same_rand):
                rrout = DDrppi(1,1,self.pimax,
                               self.binfilename,
                               rands['px'],
                               rands['py'],
                               rands['pz'],
                               False,
                               rands['px'],
                               rands['py'],
                               rands['pz'])

                rrout = np.array(rrout)
                rrout = rrout.reshape(-1,int(self.pimax))

            self.rr[self.jcount,:,:,i] = rrout['npairs']


    def reduce(self, rank=None, comm=None):

        if rank is not None:
            gnd = comm.gather(self.nd, root=0)
            gnr = comm.gather(self.nr, root=0)
            gdd = comm.gather(self.dd, root=0)
            gdr = comm.gather(self.dr, root=0)
            grr = comm.gather(self.rr, root=0)


            if rank==0:
                ndshape = [self.nd.shape[i] for i in range(len(self.nd.shape))]
                nrshape = [self.nr.shape[i] for i in range(len(self.nr.shape))]
                ddshape = [self.dd.shape[i] for i in range(len(self.dd.shape))]
                drshape = [self.dr.shape[i] for i in range(len(self.dr.shape))]
                rrshape = [self.rr.shape[i] for i in range(len(self.rr.shape))]

                ndshape.insert(1,1)
                ndshape.insert(1,1)
                nrshape.insert(1,1)
                nrshape.insert(1,1)

                ndshape[0] = self.njacktot
                nrshape[0] = self.njacktot
                ddshape[0] = self.njacktot
                drshape[0] = self.njacktot
                rrshape[0] = self.njacktot

                self.nd = np.zeros(ndshape)
                self.nr = np.zeros(nrshape)
                self.dd = np.zeros(ddshape)
                self.dr = np.zeros(drshape)
                self.rr = np.zeros(rrshape)

                jc = 0
                for i, g in enumerate(gnd):
                    if g is None: continue
                    nj = g.shape[0]
                    self.nd[jc:jc+nj,0,0,:] = g
                    self.nr[jc:jc+nj,0,0,:] = gnr[i]
                    self.dd[jc:jc+nj,:,:,:] = gdd[i]
                    self.dr[jc:jc+nj,:,:,:] = gdr[i]
                    self.rr[jc:jc+nj,:,:,:] = grr[i]

                    jc += nj

                self.jwprp = np.zeros(self.dd.shape)

                self.jnd = self.jackknife(self.nd, reduce_jk=False)
                self.jnr = self.jackknife(self.nr, reduce_jk=False)
                self.jdd = self.jackknife(self.dd, reduce_jk=False)
                self.jdr = self.jackknife(self.dr, reduce_jk=False)
                self.jrr = self.jackknife(self.rr, reduce_jk=False)

                fnorm = self.jnr / self.jnd

                self.jwprppi = (fnorm ** 2 * self.jdd - 2 * fnorm * self.jdr + self.jrr) / self.jrr
                self.jwprp = 2 * np.sum(self.jwprppi, axis=2)

                self.wprppi = np.sum(self.jwprppi, axis=0) / self.njacktot
                self.wprp = np.sum(self.jwprp, axis=0) / self.njacktot

                self.varwprppi = np.sum((self.jwprppi - self.wprppi)**2, axis=0) * (self.njacktot - 1) / self.njacktot
                self.varwprp = np.sum((self.jwprp - self.wprp)**2, axis=0) * (self.njacktot - 1) / self.njacktot
        else:
            self.jwprp = np.zeros(self.dd.shape)
            self.nd = self.nd.reshape(self.njacktot, 1, 1, self.nmbins)
            self.nr = self.nr.reshape(self.njacktot, 1, 1, self.nmbins)

            self.jnd = self.jackknife(self.nd, reduce_jk=False)
            self.jnr = self.jackknife(self.nr, reduce_jk=False)
            self.jdd = self.jackknife(self.dd, reduce_jk=False)
            self.jdr = self.jackknife(self.dr, reduce_jk=False)
            self.jrr = self.jackknife(self.rr, reduce_jk=False)

            fnorm = self.jnr / self.jnd

            self.jwprppi = (fnorm ** 2 * self.jdd - 2 * fnorm * self.jdr + self.jrr) / self.jrr
            self.jwprp = 2 * np.sum(self.jwprppi, axis=2)

            self.wprppi = np.sum(self.jwprppi, axis=0) / self.njacktot
            self.wprp = np.sum(self.jwprp, axis=0) / self.njacktot

            self.varwprppi = np.sum((self.jwprppi - self.wprppi)**2, axis=0) * (self.njacktot - 1) / self.njacktot
            self.varwprp = np.sum((self.jwprp - self.wprp)**2, axis=0) * (self.njacktot - 1) / self.njacktot


        self.wprp = self.wprp.reshape(-1, 1, self.nmbins, 1)
        self.varwprp = self.varwprp.reshape(-1, 1, self.nmbins, 1)


    def visualize(self, plotname=None, f=None, ax=None, usecols=None,
                    usez=None, compare=False, **kwargs):

        if usecols is None:
            usecols = range(self.nmbins)

        if f is None:
            f, ax = plt.subplots(len(usecols), sharex=True,
                                    sharey=True, figsize=(8,8))
            ax = np.array(ax)
            ax = ax.reshape(1, len(usecols))
            newaxes = True
        else:
            newaxes = False

        if hasattr(self, 'rmean'):
            if self.rmean is not None:
                rmean = self.rmean
        else:
            rmean = (self.rbins[1:]+self.rbins[:-1]) / 2

        for i, l in enumerate(usecols):
            ye = np.sqrt(self.varwprp[:,0,l,0])
            l1 = ax[0][i].plot(rmean, self.wprp[:,0,l,0], **kwargs)
            ax[0][i].fill_between(rmean, self.wprp[:,0,l,0]-ye, self.wprp[:,0,l,0]+ye, alpha=0.5, **kwargs)

            ax[0][i].set_xscale('log')
            ax[0][i].set_yscale('log')

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
            sax.set_ylabel(r'$w_{p}(r_{p})$', labelpad=20, fontsize=16)
            sax.set_xlabel(r'$r_{p} \, [ Mpc h^{-1}]$', labelpad=20, fontsize=16)

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax, l1[0]


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
                                          compare=True, color=Metric._color_list[i],
                                          **kwargs)
            else:
                f, ax, l1 = m.visualize(usecols=usecols[i], usez=usez[i],
                                          compare=True, color=Metric._color_list[i],
                                          f=f, ax=ax, **kwargs)
            lines.append(l1)

        if labels[0]!=None:
            f.legend(lines, labels)

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax


class XiofR(CorrelationFunction):

    def __init__(self, ministry, mbins=None, zbins=None, rbins=None,
                  minr=None, maxr=None, logbins=True, nrbins=None,
                  lightcone=True, catalog_type=None, tag=None,
                  mcutind=None, same_rand=False, inv_m=True,
                  **kwargs):

        """
        Real space 3-d correlation function, xi(r), for use with non-periodic
        data.
        """
        CorrelationFunction.__init__(self, ministry, lightcone=lightcone,
                                      mbins=mbins, nrbins=nrbins,
                                      mcutind=mcutind, zbins=zbins,
                                      same_rand=same_rand, inv_m=inv_m,
                                      catalog_type=catalog_type, tag=tag,
                                      **kwargs)

        self.logbins = logbins
        self.c = 299792.458

        if (rbins is None) & ((minr is None) | (maxr is None) | (nrbins is None)):
            self.minr = 1e-1
            self.maxr = 25
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

        self.writeCorrfuncBinFile(self.rbins)

        self.mapkeys = ['px', 'py', 'pz', self.mkey]
        self.unitmap = {'px':'mpch', 'py':'mpch', 'pz':'mpch'}

        if self.lightcone:
            self.mapkeys.append('redshift')
            self.unitmap['redshift'] = 'z'

        if self.mkey == 'luminosity':
            self.unitmap[self.mkey] = 'mag'
        else:
            self.unitmap[self.mkey] = 'msunh'

        self.nd = None
        self.nr = None
        self.dd = None
        self.dr = None
        self.rr = None

    @jackknifeMap
    def map(self, mapunit):

        if not hascorrfunc:
            raise(ImportError("CorrFunc is required to calculate xi(r)"))

        if self.dd is None:
            self.dd = np.zeros((self.njack, self.nrbins, self.nmbins, self.nzbins))
            self.dr = np.zeros((self.njack, self.nrbins, self.nmbins, self.nzbins))
            self.rr = np.zeros((self.njack, self.nrbins, self.nmbins, self.nzbins))
            self.nd = np.zeros((self.njack, self.nmbins, self.nzbins))
            self.nr = np.zeros((self.njack, self.nmbins, self.nzbins))

        if (mapunit['px'].dtype == '>f4') | (mapunit['px'].dtype == '>f8') | (mapunit['px'].dtype == np.float64):
            mu = {}
            mu['px'] = np.zeros(len(mapunit['px']), dtype=np.float32)
            mu['py'] = np.zeros(len(mapunit['py']), dtype=np.float32)
            mu['pz'] = np.zeros(len(mapunit['pz']), dtype=np.float32)

            mu['px'][:] = mapunit['px'][:]
            mu['py'][:] = mapunit['py'][:]
            mu['pz'][:] = mapunit['pz'][:]
            mu[self.mkey] = mapunit[self.mkey]

            if self.lightcone:
                mu['redshift'] = mapunit['redshift']

        else:
            mu = mapunit

        if self.lightcone:
            for i in range(self.nzbins):
                print('Finding redshift indices')

                zlidx = mu['redshift'].searchsorted(self.zbins[i])
                zhidx = mu['redshift'].searchsorted(self.zbins[i+1])

                if zlidx==zhidx:
                    print("No galaxies in redshift bin {0} to {1}".format(self.zbins[i], self.zbins[i+1]))
                    print("Min and max z: {0}, {1}".format(np.min(mu['redshift']), np.max(mu['redshift'])))
                    print(mu['redshift'])
                    continue

                for lj, j in enumerate(self.minds):
                    print('Finding luminosity indices')
                    if self.mcutind is not None:
                        lidx = (self.mbins[j] <= mu[self.mkey][zlidx:zhidx,self.mcutind]) & (mu[self.mkey][zlidx:zhidx,self.mcutind] < self.mbins[j+1])
                    else:
                        lidx = (self.mbins[j] <= mu[self.mkey][zlidx:zhidx]) & (mu[self.mkey][zlidx:zhidx] < self.mbins[j+1])

                    if (lj==self.rand_ind) | (not self.same_rand):
                        print('Generating Randoms')
                        if len(mu['pz'][zlidx:zhidx][lidx])==0:
                            self.rand_ind+=1
                            continue

                        rands = self.getCartesianRandoms(mu['px'][zlidx:zhidx][lidx],
                                                         mu['py'][zlidx:zhidx][lidx],
                                                         mu['pz'][zlidx:zhidx][lidx])


                    self.nd[self.jcount,j,i] = len(mu['pz'][zlidx:zhidx][lidx])
                    self.nr[self.jcount,j,i] = len(rands)

                    print("Number of galaxies in this z/lum bin: {0}".format(self.nd[self.jcount,j,i]))
                    print("Number of randoms in this z/lum bin: {0}".format(self.nr[self.jcount,j,i]))

                    if self.nd[self.jcount,j,i]<2: continue
                    #data data
                    print('calculating data data pairs')
                    sys.stdout.flush()

                    ddout = DD(1,1,
                                 self.binfilename,
                                 mu['px'][zlidx:zhidx][lidx],
                                 mu['py'][zlidx:zhidx][lidx],
                                 mu['pz'][zlidx:zhidx][lidx],
                                 False,
                                 mu['px'][zlidx:zhidx][lidx],
                                 mu['py'][zlidx:zhidx][lidx],
                                 mu['pz'][zlidx:zhidx][lidx])

                    ddout = np.array(ddout)

                    self.dd[self.jcount,:,j,i] = ddout['npairs']

                    print('calculating data random pairs')
                    sys.stdout.flush()

                    drout = DD(0,1,
                                 self.binfilename,
                                 mu['px'][zlidx:zhidx][lidx],
                                 mu['py'][zlidx:zhidx][lidx],
                                 mu['pz'][zlidx:zhidx][lidx],
                                 False,
                                 rands['px'],
                                 rands['py'],
                                 rands['pz'])

                    drout = np.array(drout)
                    self.dr[self.jcount,:,j,i] = drout['npairs']

                    print('calculating random random pairs')
                    sys.stdout.flush()

                    if (lj==0) | (not self.same_rand):
                        rrout = DD(1,1,
                                     self.binfilename,
                                     rands['px'],
                                     rands['py'],
                                     rands['pz'],
                                     False,
                                     rands['px'],
                                     rands['py'],
                                     rands['pz'])

                        rrout = np.array(rrout)

                    self.rr[self.jcount,:,j,i] = rrout['npairs']

        else:
            i=0
            for lj, j in enumerate(self.minds):
                print('Finding luminosity indices')
                if self.mcutind is not None:
                    lidx = (self.mbins[j] <= mu[self.mkey][:,self.mcutind]) & (mu[self.mkey][:,self.mcutind] < self.mbins[j+1])
                else:
                    lidx = (self.mbins[j] <= mu[self.mkey]) & (mu[self.mkey] < self.mbins[j+1])

                if (lj==self.rand_ind) | (not self.same_rand):
                    print('Generating Randoms')
                    if len(mu['pz'][lidx])==0:
                        self.rand_ind+=1
                        continue

                    rands = self.getCartesianRandoms(mu['px'][lidx],
                                                     mu['py'][lidx],
                                                     mu['pz'][lidx])


                self.nd[self.jcount,j,i] = len(mu['pz'][lidx])
                self.nr[self.jcount,j,i] = len(rands)

                print("Number of galaxies in this z/lum bin: {0}".format(self.nd[self.jcount,j,i]))
                print("Number of randoms in this z/lum bin: {0}".format(self.nr[self.jcount,j,i]))

                if self.nd[self.jcount,j,i]<2: continue
                #data data
                print('calculating data data pairs')
                sys.stdout.flush()

                ddout = DD(1,1,
                             self.binfilename,
                             mu['px'][lidx],
                             mu['py'][lidx],
                             mu['pz'][lidx],
                             False,
                             mu['px'][lidx],
                             mu['py'][lidx],
                             mu['pz'][lidx])

                ddout = np.array(ddout)

                self.dd[self.jcount,:,j,i] = ddout['npairs']
                print('calculating data random pairs')
                sys.stdout.flush()

                drout = DD(0,1,
                             self.binfilename,
                             mu['px'][lidx],
                             mu['py'][lidx],
                             mu['pz'][lidx],
                             False,
                             rands['px'],
                             rands['py'],
                             rands['pz'])

                drout = np.array(drout)
                self.dr[self.jcount,:,j,i] = drout['npairs']

                print('calculating random random pairs')
                sys.stdout.flush()

                if (lj==0) | (not self.same_rand):
                    rrout = DD(1,1,
                                 self.binfilename,
                                 rands['px'],
                                 rands['py'],
                                 rands['pz'],
                                 False,
                                 rands['px'],
                                 rands['py'],
                                 rands['pz'])

                    rrout = np.array(rrout)
                self.rr[self.jcount,:,j,i] = rrout['npairs']


    def reduce(self, rank=None, comm=None):

        if rank is not None:
            gnd = comm.gather(self.nd, root=0)
            gnr = comm.gather(self.nr, root=0)
            gdd = comm.gather(self.dd, root=0)
            gdr = comm.gather(self.dr, root=0)
            grr = comm.gather(self.rr, root=0)


            if rank==0:
                ndshape = [self.nd.shape[i] for i in range(len(self.nd.shape))]
                nrshape = [self.nr.shape[i] for i in range(len(self.nr.shape))]
                ddshape = [self.dd.shape[i] for i in range(len(self.dd.shape))]
                drshape = [self.dr.shape[i] for i in range(len(self.dr.shape))]
                rrshape = [self.rr.shape[i] for i in range(len(self.rr.shape))]

                ndshape.insert(1,1)
                nrshape.insert(1,1)

                ndshape[0] = self.njacktot
                nrshape[0] = self.njacktot
                ddshape[0] = self.njacktot
                drshape[0] = self.njacktot
                rrshape[0] = self.njacktot

                self.nd = np.zeros(ndshape)
                self.nr = np.zeros(nrshape)
                self.dd = np.zeros(ddshape)
                self.dr = np.zeros(drshape)
                self.rr = np.zeros(rrshape)

                jc = 0
                for i, g in enumerate(gnd):
                    if g is None: continue
                    nj = g.shape[0]
                    self.nd[jc:jc+nj,0,:,:] = g
                    self.nr[jc:jc+nj,0,:,:] = gnr[i]
                    self.dd[jc:jc+nj,:,:,:] = gdd[i]
                    self.dr[jc:jc+nj,:,:,:] = gdr[i]
                    self.rr[jc:jc+nj,:,:,:] = grr[i]

                    jc += nj

                self.jxi = np.zeros(self.dd.shape)

                self.jnd = self.jackknife(self.nd, reduce_jk=False)
                self.jnr = self.jackknife(self.nr, reduce_jk=False)
                self.jdd = self.jackknife(self.dd, reduce_jk=False)
                self.jdr = self.jackknife(self.dr, reduce_jk=False)
                self.jrr = self.jackknife(self.rr, reduce_jk=False)

                for i in xrange(self.njacktot):
                    for j in xrange(self.nmbins):
                        for k in xrange(self.nzbins):
                            self.jxi[i,:,j,k] = convert_3d_counts_to_cf(self.jnd[i,0,j,k],
                                                                        self.jnd[i,0,j,k],
                                                                        self.jnr[i,0,j,k],
                                                                        self.jnr[i,0,j,k],
                                                                        self.dd[i,:,j,k],
                                                                        self.dr[i,:,j,k],
                                                                        self.dr[i,:,j,k],
                                                                        self.rr[i,:,j,k])

                self.xi    = np.sum(self.jxi, axis=0) / self.njacktot
                self.varxi = np.sum((self.jxi - self.xi)**2, axis=0) * (self.njacktot - 1) / self.njacktot

                self.xi = self.xi.reshape(self.nrbins, 1, self.nmbins, self.nzbins)
                self.varxi = self.varxi.reshape(self.nrbins, 1, self.nmbins, self.nzbins)

        else:
            self.jxi = np.zeros(self.dd.shape)
            self.jnd = self.jackknife(self.nd, reduce_jk=False)
            self.jnr = self.jackknife(self.nr, reduce_jk=False)
            self.jdd = self.jackknife(self.dd, reduce_jk=False)
            self.jdr = self.jackknife(self.dr, reduce_jk=False)
            self.jrr = self.jackknife(self.rr, reduce_jk=False)

            for i in xrange(self.njacktot):
                for j in xrange(self.nmbins):
                    for k in xrange(self.nzbins):
                        self.jxi[i,:,j,k] = convert_3d_counts_to_cf(self.jnd[i,0,j,k],
                                                                      self.jnd[i,0,j,k],
                                                                      self.jnr[i,0,j,k],
                                                                      self.jnr[i,0,j,k],
                                                                      self.dd[i,:,j,k],
                                                                      self.dr[i,:,j,k],
                                                                      self.dr[i,:,j,k],
                                                                      self.rr[i,:,j,k])

            self.xi    = np.sum(self.jxi, axis=0) / self.njacktot
            self.varxi = np.sum((self.jxi - self.xi)**2, axis=0) * (self.njacktot - 1) / self.njacktot

            self.xi = self.xi.reshape(self.nrbins, 1, self.nmbins, self.nzbins)
            self.varxi = self.varxi.reshape(self.nrbins, 1, self.nmbins, self.nzbins)

    def visualize(self, plotname=None, f=None, ax=None, usecols=None,
                    usez=None, compare=False, **kwargs):

        if usecols is None:
            usecols = range(self.nmbins)
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

        if hasattr(self, 'rmean'):
            if self.rmean is not None:
                rmean = self.rmean
        else:
            rmean = (self.rbins[1:]+self.rbins[:-1]) / 2

        for i, l in enumerate(usecols):
            for j, z in enumerate(usez):
                ye = np.sqrt(self.varxi[:,0,l,z])
                l1 = ax[j][i].plot(rmean, self.xi[:,0,l,z], **kwargs)
                ax[j][i].fill_between(rmean, self.xi[:,0,l,z]-ye, self.xi[:,0,l,z]+ye, alpha=0.5, **kwargs)

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
            sax.set_ylabel(r'$\xi(r)$', fontsize=16, labelpad=20)
            sax.set_xlabel(r'$r \, [ Mpc h^{-1}]$', fontsize=16, labelpad=20)

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax, l1[0]


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
                                          compare=True, color=Metric._color_list[i],
                                          **kwargs)
            else:
                f, ax, l1 = m.visualize(usecols=usecols[i], usez=usez[i],
                                          compare=True, color=Metric._color_list[i],
                                          f=f, ax=ax, **kwargs)
            lines.append(l1)

        if labels[0]!=None:
            f.legend(lines, labels)

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax


class TabulatedWPrpLightcone(WPrpLightcone):

    def __init__(self, ministry, fname, *args, **kwargs):

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

        WPrpLightcone.__init__(self, ministry, *args, **kwargs)

        self.nomap = True
        self.loadWPrp()


    def loadWPrp(self):
        tab = np.loadtxt(self.fname)
        self.wprp = np.zeros((tab.shape[0], self.ncuts, 1, 1))
        self.varwprp = np.zeros((tab.shape[0], self.ncuts, 1, 1))
        self.rmean = tab[:,self.rmeancol]
        self.wprp[:,:,0,0] = tab[:,self.wprpcol:self.wprpcol+self.ncuts]
        self.varwprp[:,:,0,0] = tab[:,self.wprperr:self.wprperr+self.ncuts]


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
        self.unitmap = {'luminosity':'mag', 'polar_ang':'dec', 'azim_ang':'ra', 'redshift':'z'}

        self.rcounts = None

    @jackknifeMap
    def map(self, mapunit):

        if self.rcounts is None:
            self.rcounts = np.zeros((self.njack, self.nrbins,
                                    self.nlumbins,
                                    self.nzbins))

        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

            for j, l in enumerate(self.lumbins[:-1]):
                lidx = (self.lumbins[j]<mapunit['luminosity'][zlidx:zhidx,0]) & (mapunit['luminosity'][zlidx:zhidx,0]<self.lumbins[j+1])
                c, e = np.histogram(mapunit['rhalo'][zlidx:zhidx][lidx], bins=self.rbins)
                self.rcounts[self.jcount,:,j,i] += c

    def reduce(self, rank=None, comm=None):

        if rank is not None:
            gdata = comm.gather(self.rcounts, root=0)

            if rank==0:
                dshape = self.rcounts.shape
                dshape = [dshape[i] for i in range(len(dshape))]
                dshape[0] = self.njacktot
                self.rcounts = np.zeros(dshape)

                jc = 0
                #iterate over gathered arrays, filling in arrays of rank==0
                #process
                for g in gdata:
                    if g is None: continue
                    nj = g.shape[0]
                    self.rcounts[jc:jc+nj,:,:,:] = g

                    jc += nj

                self.rmean = (self.rbins[1:]+self.rbins[:-1])/2
                vol = 4*np.pi*(self.rmean**3)/3

                self.jrprof = np.zeros(self.rcounts.shape)

                for i in range(self.nzbins):
                    for j in range(self.nlumbins):
                        self.jrprof[:,:,j,i] /= self.rcounts[:,:,j,i] / vol

                self.jrprof, self.rprof, self.varrprof = self.jackknife(self.jrprof)

        else:
            self.rmean = (self.rbins[1:]+self.rbins[:-1])/2
            vol = 4*np.pi*(self.rmean**3)/3

            self.jrprof = np.zeros(self.rcounts.shape)

            for i in range(self.nzbins):
                for j in range(self.nlumbins):
                    self.jrprof[:,:,j,i] /= self.rcounts[:,:,j,i] / vol

            self.jrprof, self.rprof, self.varrprof = self.jackknife(self.jrprof)


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
