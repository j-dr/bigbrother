
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from .metric import Metric


class DensityMagnitudePDF(Metric):
    """
    Calculate the joint probability density of environmental
    density and magnitude.
    """

    def __init__(self, ministry, zbins=None, densbins=None,
                  magbins=None, catalog_type=None,
                  tag=None, central_only=False, normed=False,
                  **kwargs):

        if catalog_type is None:
            catalog_type = ['galaxycatalog']

        Metric.__init__(self, ministry, catalog_type=catalog_type, tag=tag, **kwargs)

        if zbins is None:
            self.zbins = np.linspace(ministry.minz, ministry.maxz, 6)
        else:
            self.zbins = zbins

        if densbins is None:
            self.densbins = np.logspace(0.1, 1, 61)
        else:
            self.densbins = densbins

        if magbins is None:
            self.magbins = np.linspace(-24, -19, 61)
        else:
            self.magbins = magbins

        self.nzbins = len(self.zbins) - 1
        self.nmagbins = len(self.magbins) - 1
        self.ndensbins = len(self.densbins) - 1
        self.central_only = central_only
        self.normed = normed

        if self.central_only:
            self.mapkeys = ['luminosity', 'density', 'redshift', 'central']
        else:
            self.mapkeys = ['luminosity', 'density', 'redshift']

        self.aschema = 'galaxyonly'
        self.unitmap = {'luminosity':'mag', 'density':'mpch'}

        self.densmagcounts = None

    def map(self, mapunit):

        if self.densmagcounts is None:
            self.densmagcounts = np.zeros((self.ndensbins, self.nmagbins,
                                            self.nzbins))

        for i, z in enumerate(self.zbins[:-1]):
            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

            c, e0, e1 = np.histogram2d(mapunit['density'][zlidx:zhidx],
                                        mapunit['luminosity'][zlidx:zhidx],
                                        bins=[self.densbins, self.magbins],
                                        normed=self.normed)

            self.densmagcounts[:,:,i] += c

    def reduce(self, rank=None, comm=None):
        area = self.ministry.galaxycatalog.getArea()

        if self.normed:
            dd = self.densbins[1:] - self.densbins[:-1]
            dm = self.magbins[1:] - self.magbins[:-1]
            dddm = np.outer(dd, dm).reshape((len(dd), len(dm), 1))
            self.densmagpdf = self.densmagcounts / dddm / np.sum(self.densmagcounts, axis=(0,1))
        else:
            self.densmagpdf = self.densmagcounts / area

    def visualize(self, compare=False, plotname=None, f=None, ax=None,
                  usez=None, colors=None, ncont=None, **kwargs):

        mdens = (self.densbins[1:]+self.densbins[:-1])/2
        mmag = (self.magbins[1:]+self.magbins[:-1])/2

        X, Y = np.meshgrid(mdens, mmag)

        if usez is None:
            usez = list(range(self.densmagpdf.shape[2]))

        nz = len(usez)

        if ncont is None:
            ncont = 5

        if f is None:
            f, ax = plt.subplots(self.nzbins,
                                   sharex=True, sharey=True, figsize=(8,8))
            ax = np.array(ax)
            newaxes = True
        else:
            newaxes = False

        for i in range(nz):
            try:
                l1 = ax[i].contour(X, Y, self.densmagpdf[:,:,i].T, ncont,
                                   colors=colors, **kwargs)
            except ValueError as e:
                print('Caught error {0}'.format(e))
                pass

        if newaxes:
            sax = f.add_subplot(111)
            sax.patch.set_alpha(0.0)
            sax.patch.set_facecolor('none')
            sax.spines['top'].set_color('none')
            sax.spines['bottom'].set_color('none')
            sax.spines['left'].set_color('none')
            sax.spines['right'].set_color('none')
            sax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

            if 'xlabel' in kwargs:
                sax.set_xlabel(kwargs['xlabel'])
            else:
                sax.set_xlabel(r'Density')

            if 'ylabel' in kwargs:
                sax.set_ylabel(kwargs['ylabel'])
            else:
                sax.set_ylabel(r'Magnitude')

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)


        return f, ax, l1

    def compare(self, othermetrics, plotname=None, usez=None,
                  labels=None, colors=None, ncont=None, **kwargs):

        tocompare = [self]
        tocompare.extend(othermetrics)

        if usez!=None:
            if not hasattr(usez[0], '__iter__'):
                usez = [usez]*len(tocompare)
            else:
                assert(len(usez)==len(tocompare))
        else:
            usez = [None]*len(tocompare)

        if labels is None:
            labels = [None] * len(tocompare)

        if colors is None:
            colors = [None] * len(tocompare)

        lines = []

        for i, m in enumerate(tocompare):
            if i==0:
                f, ax, l = m.visualize(usez=usez[i], compare=True,
                                        colors=colors[i], ncont=ncont,
                                        **kwargs)
            else:
                f, ax, l = m.visualize(usez=usez[i], compare=True,
                                        colors=colors[i], ncont=ncont,
                                        ax=ax, f=f, **kwargs)

            lines.append(l)


        if labels[0]!=None:
            f.legend(lines, labels)

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax

class ConditionalDensityPDF(Metric):

    def __init__(self, ministry, magcuts=None, densbins=None,
                  zbins=None, catalog_type=None, tag=None,
                  magcutind=None, normed=True, centrals=None,
                  **kwargs):

        if catalog_type is None:
            catalog_type = ['galaxycatalog']

        Metric.__init__(self, ministry, tag=tag, catalog_type=catalog_type, **kwargs)

        if centrals is None:
            self.centrals = 0
        else:
            self.centrals = centrals

        if magcuts is None:
            self.magcuts = np.linspace(-24, -18, 6)
        else:
            self.magcuts = magcuts

        if densbins is None:
            self.densbins = np.logspace(-3., np.log10(15.), 50)
        else:
            self.densbins = densbins

        if zbins is None:
            self.zbins = np.linspace(ministry.minz, ministry.maxz, 6)
        else:
            self.zbins = zbins

        self.magcutind = magcutind
        self.normed    = normed


        self.nmagcuts  = len(self.magcuts) - 1
        self.nzbins    = len(self.zbins) - 1
        self.ndensbins = len(self.densbins) - 1


        self.aschema = 'galaxyonly'
        if self.centrals != 0:
            self.mapkeys = ['luminosity', 'density', 'redshift', 'central']
        else:
            self.mapkeys = ['luminosity', 'density', 'redshift']

        self.unitmap = {'luminosity':'mag', 'density':'mpch'}


    def map(self, mapunit):

        mu = {}

        if self.centrals == 1:
            for k in list(mapunit.keys()):
                mu[k] = mapunit[k][mapunit['central']==1]
        elif self.centrals == 2:
            for k in list(mapunit.keys()):
                mu[k] = mapunit[k][mapunit['central']==0]
        else:
            for k in list(mapunit.keys()):
                mu[k] = mapunit[k]


        if not hasattr(self, 'cdcounts'):
            self.cdcounts = np.zeros((self.ndensbins, self.nmagcuts,
                                        self.nzbins))

        for i in range(self.nzbins):
            zlidx = mu['redshift'].searchsorted(self.zbins[i])
            zhidx = mu['redshift'].searchsorted(self.zbins[i+1])

            jrmdist, e0, e1        = np.histogram2d(mu['density'][zlidx:zhidx].reshape(-1),
                                                    mu['luminosity'][zlidx:zhidx].reshape(-1),
                                                    bins=[self.densbins, self.magcuts])
            self.cdcounts[:,:,i]   += jrmdist


    def reduce(self, rank=None, comm=None):
        dr                = (self.densbins[1:] - self.densbins[:-1]).reshape(-1,1,1)
        crmcounts         = np.cumsum(self.cdcounts, axis=1)
        self.rmcounts     = np.sum(crmcounts, axis=0).reshape(1,self.nmagcuts,1)
        self.cdenspdf     = crmcounts / self.rmcounts / dr
        #rmerr      = np.sqrt(rmdist)

    def visualize(self, compare=False, plotname=None, f=None, ax=None,
                  usez=None, usecuts=None, **kwargs):

        mdens = (self.densbins[1:]+self.densbins[:-1])/2

        if usez is None:
            usez = list(range(self.cdenspdf.shape[2]))

        if usecuts is None:
            usecuts = list(range(self.cdenspdf.shape[1]))

        ncuts = len(usecuts)
        nz    = len(usez)

        if f is None:
            f, ax = plt.subplots(ncuts, nz,
                                   sharex=True, sharey=False, figsize=(8,8))
            ax = np.array(ax)
            ax = ax.reshape((ncuts,nz))
            newaxes = True
        else:
            newaxes = False

        for i in range(nz):
            for j in range(ncuts):
                l1 = ax[j,i].semilogx(mdens, self.cdenspdf[:,j,i],
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

            if 'xlabel' in kwargs:
                sax.set_xlabel(kwargs['xlabel'])
            else:
                sax.set_xlabel(r'Density')

            if 'ylabel' in kwargs:
                sax.set_ylabel(kwargs['ylabel'])
            else:
                sax.set_ylabel(r'p(density|magnitude>M)')

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax, l1

    def compare(self, othermetrics, plotname=None, usez=None,
                  labels=None, usecuts=None, ncont=None, **kwargs):

        tocompare = [self]
        tocompare.extend(othermetrics)

        if usez!=None:
            if not hasattr(usez[0], '__iter__'):
                usez = [usez]*len(tocompare)
            else:
                assert(len(usez)==len(tocompare))
        else:
            usez = [None]*len(tocompare)

        if labels is None:
            labels = [None] * len(tocompare)

        if usecuts is None:
            usecuts = [None] * len(tocompare)

        lines = []

        for i, m in enumerate(tocompare):
            if i==0:
                f, ax, l = m.visualize(usez=usez[i], usecuts=usecuts[i],
                                        compare=True, **kwargs)
            else:
                f, ax, l = m.visualize(usez=usez[i], usecuts=usecuts[i],
                                        compare=True, ax=ax, f=f, **kwargs)

            lines.append(l)

        if labels[0]!=None:
            f.legend(lines, labels)

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax

class AnalyticConditionalDensityPDF(ConditionalDensityPDF):

    def __init__(self, ministry, params, form, magcuts=None,
                  densbins=None, zbins=None, **kwargs):

        self.form   = form
        self.params = params

        ConditionalDensityPDF.__init__(self, ministry, magcuts=magcuts,
                                        densbins=densbins, zbins=zbins,
                                        **kwargs)
        self.generateCDPDF()

    def generateCDPDF(self):

        if self.form == 'addgals':
            self.generateADDGALSCDPDF()
        else:
            raise ValueError

    def generateADDGALSCDPDF(self):

        if not hasattr(self, 'cdenspdf'):
            self.cdenspdf = np.zeros((self.ndensbins, self.nmagcuts,
                                        self.nzbins))
            self.dpdfpars = np.zeros((5,self.nmagcuts, self.nzbins))

        mag_ref = -20.5
        bright_mag_lim = -22.5-mag_ref

        self.meandens = (self.densbins[1:] + self.densbins[:-1]) / 2
        self.meanz = (self.zbins[1:] + self.zbins[:-1]) / 2

        for i in range(self.nzbins):
            for j in range(self.nmagcuts):
                mag = self.magcuts[j]-mag_ref
                z   = self.meanz[i]
                if mag<bright_mag_lim: mag = bright_mag_lim

                pmag = np.sum(np.array([self.params['pmag'][k]*mag**k for k in range(len(self.params['pmag']))]))
                pz   = np.sum(np.array([self.params['pz'][k]*z**k for k in range(len(self.params['pz']))]))
                p = pmag + pz

                mucmag = np.sum(np.array([self.params['mucmag'][k]*mag**k for k in range(len(self.params['mucmag']))]))
                mucz   = np.sum(np.array([self.params['mucz'][k]*z**k for k in range(len(self.params['mucz']))]))
                muc = mucmag + mucz

                mufmag = np.sum(np.array([self.params['mufmag'][k]*mag**k for k in range(len(self.params['mufmag']))]))
                mufz   = np.sum(np.array([self.params['mufz'][k]*z**k for k in range(len(self.params['mufz']))]))
                muf = mufmag + mufz

                sigmacmag = np.sum(np.array([self.params['sigmacmag'][k]*mag**k for k in range(len(self.params['sigmacmag']))]))
                sigmacz   = np.sum(np.array([self.params['sigmacz'][k]*z**k for k in range(len(self.params['sigmacz']))]))
                sigmac = sigmacmag + sigmacz

                sigmafmag = np.sum(np.array([self.params['sigmafmag'][k]*mag**k for k in range(len(self.params['sigmafmag']))]))
                sigmafz   = np.sum(np.array([self.params['sigmafz'][k]*z**k for k in range(len(self.params['sigmafz']))]))
                sigmaf = sigmafmag + sigmafz


                self.dpdfpars[:,j,i] = np.array([p, muc, sigmac, muf, sigmaf])

                self.cdenspdf[:,j,i] = (1 - p) * np.exp(-(np.log(self.meandens) - muc) ** 2 / (2 * sigmac ** 2)) / ( self.meandens * np.sqrt(2 * np.pi ) * sigmac ) + p * np.exp(-(self.meandens - muf) ** 2 / (2 * sigmaf ** 2)) / (np.sqrt(2 * np.pi ) * sigmaf )
