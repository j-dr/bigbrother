from __future__ import print_function, division
#if __name__=='__main__':
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

try:
    import treecorr as tc
    hastreecorr = True
except:
    hastreecorr = False

try:
    import Corrfunc._countpairs_mocks as countpairs
    hascorrfunc = True
except:
    hascorrfunc = False

import numpy as np
import healpy as hp

from .metric import Metric, GMetric

class CorrelationFunction(Metric):

    def __init__(self, ministry, zbins=None, lumbins=None,
                   nrbins=None, subjack=False,
                   catalog_type=None, lcutind=None,
                   tag=None):
        """
        Generic correlation function.
        """
        Metric.__init__(self, ministry, tag=tag)

        if catalog_type is None:
            self.catalog_type = ['galaxycatalog']
        else:
            self.catalog_type = catalog_type

        if zbins is None:
            self.zbins = np.array([0.0, 0.2])
        else:
            self.zbins = zbins
            self.zbins = np.array(self.zbins)

        self.nzbins = len(self.zbins)-1

        if lumbins is None:
            self.lumbins = np.array([-22, -21, -20, -19])
        else:
            self.lumbins = lumbins

        self.nlumbins = len(self.lumbins)-1

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

    def generateAngularRandoms(self, cat, rand_factor=10, nside=8, nest=True, selectz=False):
       """
       Generate a set of randoms from a catalog by pixelating the input
       catalog and uniformly distributing random points within the pixels
       occupied by galaxies.

       Also option to assign redshifts with the same distribution as the input
       catalog to allow for generation of randoms once for all z bins.
       """

       if selectz:
           rdtype = np.dtype([('azim_ang', np.float32), ('polar_ang', np.float32),
                              ('redshift', np.float32)])
       else:
           rdtype = np.dtype([('azim_ang', np.float32), ('polar_ang', np.float32)])

       rsize = len(cat)*rand_factor

       #randomly generate angles within region bounded by catalog angles
       grand = np.zeros(rsize, dtype=rdtype)
       grand['azim_ang'] = np.random.uniform(low=np.min(cat['azim_ang']),
                                             high=np.max(cat['azim_ang']),
                                             size=rsize)
       grand['polar_ang'] = np.random.uniform(low=np.min(cat['polar_ang']),
                                              high=np.max(cat['polar_ang']),
                                              size=rsize)
       if selectz:
           grand['redshift'] = np.random.choice(cat['redshift'], size=rsize)
           zidx = grand['redshift'].argsort()
           grand = grand[zidx]

       #only keep points which fall within the healpix cells overlapping the catalog
       cpix = hp.ang2pix(nside, (cat['polar_ang']+90)*np.pi/180., cat['azim_ang']*np.pi/180., nest=nest)
       ucpix = np.unique(cpix)
       rpix = hp.ang2pix(nside, (grand['polar_ang']+90)*np.pi/180, grand['azim_ang']*np.pi/180., nest=nest)
       inarea = np.in1d(rpix, ucpix)

       grand = grand[inarea]

       return grand

    def genbins(self, minb, maxb, nb):

        if self.logbins:
            bins = np.logspace(np.log10(minb), np.log10(maxb), nb)
        else:
            bins = np.linspace(minb, maxb, nb)

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

class AngularCorrelationFunction(CorrelationFunction):

    def __init__(self, ministry, zbins=None, lumbins=None, mintheta=None,
                 maxtheta=None, nabins=None, subjack=False,
                 catalog_type=None, tag=None, lcutind=None):
        """
        Angular correlation function, w(theta), for use with non-periodic
        data. All angles should be specified in degrees.
        """
        CorrelationFunction.__init__(self, ministry, zbins=zbins,
                                      lumbins=lumbins, nrbins=nabins,
                                      subjack=subjack, lcutind=lcutind,
                                      catalog_type=catalog_type, tag=tag)

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
        rand = self.generateAngularRandoms(cat, nside=128)

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
                  njack=None, lcutind=None):
        """
        Angular correlation function, w(theta), for use with non-periodic
        data. All angles should be specified in degrees.
        """
        CorrelationFunction.__init__(self, ministry, zbins=zbins,
                                      lumbins=lumbins, nrbins=nrbins,
                                      subjack=subjack, lcutind=lcutind,
                                      catalog_type=catalog_type, tag=tag)

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
            self.pimax = 40.0

        if njack is None:
            self.njack = 1

        self.jcount = 0

        self.writeCorrfuncBinFile(self.rbins)
        self.binfilename = '/anaconda/lib/python2.7/site-packages/Corrfunc/xi_mocks/tests/bins'

        self.mapkeys = ['luminosity', 'redshift', 'polar_ang', 'azim_ang']
        self.unitmap = {'luminosity':'mag', 'polar_ang':'dec', 'azim_ang':'ra'}

    def map(self, mapunit):

        if not hascorrfunc:
            raise(ImportError("CorrFunc is required to calculate wp(rp)"))

        if not hasattr(self, 'wthetaj'):
            self.DD = np.zeros((self.nrbins, self.nlumbins, self.nzbins, self.njack))
            self.DR = np.zeros((self.nrbins, self.nlumbins, self.nzbins, self.njack))
            self.RR = np.zeros((self.nrbins, self.nlumbins, self.nzbins, self.njack))




        #calculate DD
        for i in range(self.nzbins):
            print('Finding redshift indices')

            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

            print('Generating randoms')
            rands = self.generateAngularRandoms(mapunit[zlidx:zhidx], selectz=True, nside=128)

            #zrlidx = rands['redshift'].searchsorted(self.zbins[i])
            #zrhidx = rands['redshift'].searchsorted(self.zbins[i+1])

            for j in range(self.nlumbins):
                print('Finding luminosity indices')
                lidx = (self.lumbins[j] <= mapunit['luminosity'][zlidx:zhidx,self.lcutind]) & (mapunit['luminosity'][zlidx:zhidx,self.lcutind] < self.lumbins[j+1])
                print(len(mapunit[zlidx:zhidx][lidx]))
                print(len(rands))
                print('Calculating pairs')

                #data data
                print('calculating data data pairs')
                results = countpairs.countpairs_rp_pi_mocks(1, 1, 1, self.pimax,
                                        self.binfilename,
                                        mapunit[zlidx:zhidx][lidx]['azim_ang'],
                                        mapunit[zlidx:zhidx][lidx]['polar_ang'],
                                        mapunit[zlidx:zhidx][lidx]['redshift']*self.c,
                                        mapunit[zlidx:zhidx][lidx]['azim_ang'],
                                        mapunit[zlidx:zhidx][lidx]['polar_ang'],
                                        mapunit[zlidx:zhidx][lidx]['redshift']*self.c)

                self.DD[:,j,i,self.jcount] = np.array([results[k][4] for k in range(self.nrbins)])

                #data randoms
                print('calculating data random pairs')
                results = countpairs.countpairs_rp_pi_mocks(0, 1, 1, self.pimax,
                                        self.binfilename,
                                        mapunit[zlidx:zhidx][lidx]['azim_ang'],
                                        mapunit[zlidx:zhidx][lidx]['polar_ang'],
                                        mapunit[zlidx:zhidx][lidx]['redshift']*self.c,
                                        rands['azim_ang'],
                                        rands['polar_ang'],
                                        rands['redshift']*self.c)

                self.DR[:,j,i,self.jcount] = np.array([results[k][4] for k in range(self.nrbins)])

                #randoms randoms
                print('calculating random random pairs')
                results = countpairs.countpairs_rp_pi_mocks(1, 1, 1, self.pimax,
                                        self.binfilename,
                                        rands['azim_ang'],
                                        rands['polar_ang'],
                                        rands['redshift']*self.c,
                                        rands['azim_ang'],
                                        rands['polar_ang'],
                                        rands['redshift']*self.c)

                self.RR[:,j,i,self.jcount] = np.array([results[k][4] for k in range(self.nrbins)])

                self.jcount += 1

    def reduce(self):
        pass

    def visualize(self):
        pass

    def compare(self):
        pass




class GalaxyRadialProfileBCC(Metric):

    def __init__(self, ministry, zbins=None, lumbins=None, rbins=None,
                 massbins=None, subjack=False, catalog_type=['galaxycatalog'],
                 tag=None):
        """
        Radial profile of galaxies around their nearest halos.
        """

        Metric.__init__(self, ministry, tag=tag)

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
