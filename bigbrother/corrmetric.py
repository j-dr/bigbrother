from __future__ import print_function, division
if __name__=='__main__':
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pylab as plt
import treecorr as tc
import numpy as np

from .metric import Metric, GMetric


class AngularCorrelationFunction(Metric):

    def __init__(self, ministry, zbins=None, lumbins=None, mintheta=1e-2,
                 maxtheta=1, nabins=15, subjack=False,
                 catalog_type=['galaxycatalog']):
        """
        Angular correlation function, w(theta), for use with non-periodic
        data. All angles should be specified in degrees.
        """
        Metric.__init__(self, ministry)

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

        self.mintheta = mintheta
        self.maxtheta = maxtheta
        self.nabins = 15

        self.subjack = subjack
        self.aschema = 'galaxygalaxy'

        if self.subjack:
            raise NotImplementedError

        self.jsamples = 0
        self.mapkeys = ['luminosity', 'redshift', 'polar_ang', 'azim_ang']
        self.unitmap = {'luminosity':'mag', 'polar_ang':'dec', 'azim_ang':'ra'}

    def map(self, mapunit):

        self.jsamples += 1

        if not hasattr(self, 'wthetaj'):
            self.wthetaj = np.zeros((self.nabins, self.nlumbins, self.nzbins))
            self.varwthetaj = np.zeros((self.nabins, self.nlumbins, self.nzbins))

        #putting this outside loop maybe faster, inside loop
        #lower memory usage
            
        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

            for j in range(self.nlumbins):
                #luminosity should be at least 2d
                lidx = np.where((self.lumbins[i] < mapunit['luminosity'][zlidx:zhidx,0]) &
                                (mapunit['luminosity'][zlidx:zhidx,0] <= self.lumbins[i+1]))
                cat  = {key:mapunit[key][zlidx:zhidx][lidx] for key in mapunit.keys()}
                rand = self.generate_randoms(cat, nside=128)

                d  = treecorr.Catalog(x=cat['azim_ang'], y=cat['polar_ang'])
                r = treecorr.Catalog(x=rand['azim_ang'], y=rand['polar_ang'])

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
    
    def generate_randoms(self, cat, rand_factor=10, nside=8, nest=True, selectz=False):
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
       grand['azim_ang'] = np.random.uniform(low=np.min(cat['RA']),
                                             high=np.max(cat['RA']),
                                             size=rsize)
       grand['polar_ang'] = np.random.uniform(low=np.min(cat['DEC']),
                                              high=np.max(cat['DEC']),
                                              size=rsize)
       if selectz:
           grand['redshift'] = np.random.choice(cat['redshift'], size=rsize)
           zidx = grand['redshift'].argsort()
           grand = grand[zidx]
    
       #only keep points which fall within the healpix cells overlapping the catalog
       cpix = hp.ang2pix(nside, *radec2rad(cat['DEC'], cat['RA']), nest=nest)
       ucpix = np.unique(cpix)
       rpix = hp.ang2pix(nside, *radec2rad(grand['DEC'], grand['RA']), nest=nest)
       inarea = np.in1d(rpix, ucpix)
    
       grand = grand[inarea]
    
       return grand

        
class GalaxyRadialProfileBCC(Metric):

    def __init__(self, ministry, zbins=None, lumbins=None, rbins=None,
                 massbins=None, subjack=False, catalog_type=['galaxycatalog']):
        """
        Radial profile of galaxies around their nearest halos.
        """

        Metric.__init__(self, ministry)

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
            zidx = (self.zbins[i]<mapunit['redshift']) & (mapunit['redshift']<self.zbins[i+1])
            for j, l in enumerate(self.lumbins[:-1]):
                lidx = (self.lumbins[i]<mapunit['luminosity']) & (mapunit['luminosity']<self.lumbins[i+1])

                c, e = np.histogram(mapunit['rhalo'][zidx&lidx], bins=self.rbins)
                self.rprof[:,j,i] = c

    def reduce(self):

        self.rmean = (self.rbins[1:]+self.rbins[:-1])/2
        vol = 4*np.pi*self.rmean/3

        self.rprof /= vol


    def visualize(self, plotname=None, f=None, ax=None, **kwargs):

        if f is None:
            f, ax = plt.subplots(self.nlumbins, self.nzbins,
                                 sharex=True, sharey=True,
                                 figsize=(8,8))
            newaxes = True
        else:
            newaxes = False

        if self.nzbins>1:
            for i, in range(self.nlbins):
                for j in range(self.nzbins):
                    ax[i][j].semilogx(self.rmean, self.rprof[:,i,j], 
                                      **kwargs)
        else:
            for i, in range(self.nlbins):
                for j in range(self.nzbins):
                    ax[i].semilogx(self.rmean, self.rprof[:,i,j], 
                                   **kwargs)

        if newaxes:
            sax = f.add_subplot(111)
            sax.spines['top'].set_color('none')
            sax.spines['bottom'].set_color('none')
            sax.spines['left'].set_color('none')
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            sax.set_xlabel(r'$r\, [Mpc \, h^{-1}]$')
            sax.set_ylabel(r'$n \, [Mpc^{3} \, h^{-1}]$')

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax


    def compare(self):
        pass

        
                
                

