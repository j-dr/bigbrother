from __future__ import print_function, division
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pylab as plt
import numpy as np

from .metric import Metric


class DensityMagnitudePDF(Metric):
    """
    Calculate the joint probability density of environmental
    density and magnitude.
    """

    def __init__(self, ministry, zbins=None, densbins=None,
                  magbins=None, catalog_type=None,
                  tag=None, central_only=False, normed=False):

        if catalog_type is None:
            catalog_type = ['galaxycatalog']

        Metric.__init__(self, ministry, catalog_type=catalog_type, tag=tag)

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

    def map(self, mapunit):

        if not hasattr(self, 'densmagcounts'):
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

    def reduce(self):
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
            usez = range(self.densmagpdf.shape[2])

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
                  magcutind=None, normed=True):

        if catalog_type is None:
            catalog_type = ['galaxycatalog']

        Metric.__init__(self, ministry, tag=tag, catalog_type=catalog_type)

        if magcuts is None:
            self.magcuts = np.linspace(-24, -18, 6)
        else:
            self.magcuts = magcuts

        if densbins is None:
            self.densbins = np.logspace(-1, 1, 61)
        else:
            self.densbins = densbins

        if zbins is None:
            self.zbins = np.linspace(ministry.minz, ministry.maxz, 6)
        else:
            self.zbins = zbins

        self.magcutind = magcutind
        self.normed    = normed

        self.nmagcuts  = len(self.magcuts)
        self.nzbins    = len(self.zbins) - 1
        self.ndensbins = len(self.densbins) - 1

        self.aschema = 'galaxyonly'
        self.mapkeys = ['luminosity', 'density', 'redshift']
        self.unitmap = {'luminosity':'mag', 'density':'mpch'}


    def map(self, mapunit):

        if not hasattr(self, 'cdcounts'):
            self.cdcounts = np.zeros((self.ndensbins, self.nmagcuts,
                                        self.nzbins))

        for i in range(self.nzbins):
            zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
            zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

            for j in range(self.nmagcuts):
                if self.magcutind is None:
                    lidx = mapunit['luminosity'][zlidx:zhidx]<self.magcuts[j]
                else:
                    lidx = mapunit['luminosity'][zlidx:zhidx,self.magcutind]<self.magcuts[j]

                c, e = np.histogram(mapunit['density'][zlidx:zhidx][lidx],
                                bins=self.densbins)

                self.cdcounts[:,j,i] = c

    def reduce(self):
        area = self.ministry.galaxycatalog.getArea()

        if self.normed:
            #reshape to allow broadcasting
            dd = (self.densbins[1:] - self.densbins[:-1]).reshape((self.ndensbins, 1, 1))
            self.cdenspdf = self.cdcounts / dd / np.sum(self.cdcounts, axis=0)
        else:
            self.cdenspdf = self.cdcounts / area


    def visualize(self, compare=False, plotname=None, f=None, ax=None,
                  usez=None, usecuts=None, **kwargs):

        mdens = (self.densbins[1:]+self.densbins[:-1])/2

        if usez is None:
            usez = range(self.cdenspdf.shape[2])

        if usecuts is None:
            usecuts = range(self.cdenspdf.shape[1])

        ncuts = len(usecuts)
        nz    = len(usez)

        if f is None:
            f, ax = plt.subplots(ncuts, nz,
                                   sharex=True, sharey=True, figsize=(8,8))
            ax = np.array(ax)
            ax = ax.reshape((ncuts,nz))
            newaxes = True
        else:
            newaxes = False

        for i in range(nz):
            for j in range(ncuts):
                l1 = ax[j,i].plot(mdens, self.cdenspdf[:,j,i],
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

def analyticConditionalDensityPDF(ConditionalDensityPDF):

    def __init__(self, ministry, params, form, magcuts=None,
                  densbins=None, zbins=None):

        self.form   = form
        self.params = params

        ConditionalDensityPDF.__init__(self, ministry, magcuts=magcuts,
                                        densbins=densbins, zbins=zbins)
        self.generateCDPDF()

    def generateCDPDF(self):

        if self.form == 'addgals':
            self.generateADDGALSCDPDF()
        else:
            raise(ValueError("Analytic form {0} is not implemented!".format(self.form)))

    def generateADDGALSCDPDF(self):

        if not hasattr(self, 'cdenspdf'):
            self.cdenspdf = np.zeros((self.ndensbins, self.nmagcuts,
                                        self.nzbins))

        mag_ref = -20.5
        bright_mag_lim = -22.5-mag_ref

        meandens = (self.densbins[1:] + self.densbins[:-1]) / 2
        meanz = (self.zbins[1:] + self.zbins[:-1]) / 2

        for i in range(self.nzbins):
            for j in range(self.nmagcuts):
                mag = self.magcuts[j]
                z   = self.meanz[i]
                if mag<bright_mag_lim: mag = bright_mag_lim

                pmag = np.sum(np.array([mag**i for i in range(len(self.params['pmag']))]) * self.params['pmag'])
                pz   = np.sum(np.array([z**i for i in range(len(self.params['pz']))]) * self.params['pz'])
                p = pmag + pz

                mucmag = np.sum(np.array([mag**i for i in range(len(self.params['mucmag']))]) * self.params['mucmag'])
                mucz   = np.sum(np.array([z**i for i in range(len(self.params['mucz']))]) * self.params['mucz'])
                muc = mucmag + mucz

                mufmag = np.sum(np.array([mag**i for i in range(len(self.params['mufmag']))]) * self.params['mufmag'])
                mufz   = np.sum(np.array([z**i for i in range(len(self.params['mufz']))]) * self.params['mufz'])
                muf = mufmag + mufz

                sigmacmag = np.sum(np.array([mag**i for i in range(len(self.params['sigmacmag']))]) * self.params['sigmacmag'])
                sigmacz   = np.sum(np.array([z**i for i in range(len(self.params['sigmacz']))]) * self.params['sigmacz'])
                sigmac = sigmacmag + sigmacz

                sigmafmag = np.sum(np.array([mag**i for i in range(len(self.params['sigmafmag']))]) * self.params['sigmafmag'])
                sigmafz   = np.sum(np.array([z**i for i in range(len(self.params['sigmafz']))]) * self.params['sigmafz'])
                sigmaf = sigmafmag + sigmafz

                self.cdenspdf[:,j,i] = (1 - p) * np.exp((np.log(meandens) - muc) ** 2 / (2 * sigmac ** 2)) / ( meandens * np.sqrt(2 * np.pi ) * sigmac ) + p * np.exp((meandens - muf) ** 2 / (2 * sigmaf ** 2)) / ( meandens * np.sqrt(2 * np.pi ) * sigmaf )
                
