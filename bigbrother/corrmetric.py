from __future__ import print_function, division
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import fitsio
import warnings
import seaborn as sns
import sys

try:
    import treecorr as tc
    hastreecorr = True
except:
    hastreecorr = False

try:
    from Corrfunc.theory import DDrppi, DD, wp, xi
    from Corrfunc.mocks import DDtheta_mocks, DDrppi_mocks
    from Corrfunc.utils import convert_3d_counts_to_cf
    hascorrfunc = True
except:
    hascorrfunc = False

import numpy as np
import healpy as hp
import healpix_util as hu

from .metric import Metric, GMetric, jackknifeMap

class CorrelationFunction(Metric):

    def __init__(self, ministry, zbins=None, mbins=None,
                   nrbins=None, subjack=False, lightcone=True,
                   catalog_type=None, mcutind=None,
                   tag=None, same_rand=False, inv_m=True,
                   randname=None, rand_factor=10,
                   upper_limit=False, appmag=False,
                   rand_azim_ang_field='RA', rand_polar_ang_field='DEC',
                   rand_px_field='PX', rand_py_field='PY', rand_pz_field='PZ',
                   weight_field=None, redshift_field=None,
                   clustercatalog=None, nthreads=1,
                   weights=False, no_mkey=False,
                   rand_nside=None, rand_nest=True,
                   rr_rand_factor=1.0, rand_ra_lims=None,
                   rand_dec_lims=None, ra_lims=None,
                   dec_lims=None, calc_weighted_x = False,
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
        self.upper_limit = upper_limit
        self.appmag    = appmag

        self.nthreads = nthreads
        self.rand_nside = rand_nside
        self.rand_nest  = rand_nest

        #random file stuff
        self.rand_azim_ang_field  = rand_azim_ang_field
        self.rand_polar_ang_field = rand_polar_ang_field
        self.weight_field         = weight_field
        self.redshift_field       = redshift_field

        self.rand_px_field = rand_px_field
        self.rand_py_field = rand_py_field
        self.rand_pz_field = rand_pz_field

        self.rand_ra_lims  = rand_ra_lims
        self.rand_dec_lims = rand_dec_lims

        self.ra_lims  = ra_lims
        self.dec_lims = dec_lims

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
            if self.appmag:
                self.mbins = np.array([0,40])
            else:
                self.mbins = np.array([-30, 0])

        elif (mbins is None) & (self.catalog_type == ['halocatalog']):
            self.mbins = np.array([10**7, 10**17])
        elif self.catalog_type[0] == 'particlecatalog':
            self.mbins = [0,1]
        else:
            self.mbins = mbins

        if self.mbins is not None:
            if self.upper_limit:
                self.nmbins = len(self.mbins)
            else:
                self.nmbins = len(self.mbins) - 1
        else:
            self.nmbins = 1

        if inv_m:
            self.minds = np.arange(self.nmbins)[::-1]
        else:
            self.minds = np.arange(self.nmbins)

        self.rand_ind = 0

        self.same_rand = same_rand
        self.inv_m = inv_m
        self.randname = randname
        self.rand_factor = rand_factor
        self.rr_rand_factor = rr_rand_factor
        self.calc_weighted_x = calc_weighted_x

        if nrbins is None:
            self.nrbins = 15
        else:
            self.nrbins = nrbins

        self.subjack = subjack

        if ('galaxycatalog' == self.catalog_type[0]):
            self.aschema = 'galaxygalaxy'
            if not no_mkey:
                if not self.appmag:
                    self.mkey = 'luminosity'
                else:
                    self.mkey = 'appmag'
            else:
                self.mkey = None

        elif 'halocatalog' == self.catalog_type[0]:
            if not no_mkey:
                if not hasattr(self, 'clustercatalog'):
                    self.mkey = 'halomass'
                elif not self.clustercatalog:
                    self.mkey = 'halomass'
                else:
                    self.mkey = 'richness'
            else:
                self.mkey = None
            self.aschema = 'halohalo'
        else:
            self.mkey = None
            self.aschema = 'particleparticle'

        self.mcutind = mcutind

        if self.subjack:
            raise NotImplementedError

        self.weights = weights

        self.jsamples = 0


    def getRandoms(self, aza, pla, z=None, sample=0, zmin=None, zmax=None):

        if self.randname is None:
            rand = self.generateAngularRandoms(aza, pla, z=z)

        elif sample==0:
            rand = self.readAngularRandoms(self.randname, aza, pla, z=z,
                                            dt=aza.dtype,
                                            rand_factor=self.rand_factor,
                                            polar_ang_field=self.rand_polar_ang_field,
                                            azim_ang_field=self.rand_azim_ang_field,
                                            weight_field=self.weight_field,
                                            redshift_field=self.redshift_field,
                                            zmin=zmin, zmax=zmax)
        else:
            rand = self.readAngularRandoms(self.randname1, aza, pla, z=z,
                                            dt=aza.dtype,
                                            rand_factor=self.rand_factor,
                                            polar_ang_field=self.rand_polar_ang_field1,
                                            azim_ang_field=self.rand_azim_ang_field1,
                                            weight_field=self.weight_field1,
                                            redshift_field=self.redshift_field1,
                                            zmin=zmin, zmax=zmax)

        return rand

    def generateCartesianRandoms(self, x, y, z, rand_fact=10):
        rsize = len(x)*rand_fact

        if self.rand_nside:
            rdtype = np.dtype([('px', np.float32), ('py', np.float32),
                               ('pz', np.float32), ('redshift', np.float32)])

        else:
            rdtype = np.dtype([('px', np.float32), ('py', np.float32),
                               ('pz', np.float32)])

        gr = np.zeros(rsize, dtype=rdtype)

        if self.rand_nside is None:

            gr['px'] = np.random.uniform(low=np.min(x),
                                         high=np.max(x),
                                         size=rsize)
            gr['py'] = np.random.uniform(low=np.min(y),
                                         high=np.max(y),
                                         size=rsize)
            gr['pz'] = np.random.uniform(low=np.min(z),
                                         high=np.max(z),
                                         size=rsize)
        else:
            print('generating randoms using nside {}'.format(self.rand_nside))
            print('Nest ordering : {}'.format(self.rand_nest))
            r  = np.sqrt(x**2 + y**2 + z**2)
            rr = np.random.choice(r, size=rsize)

            pix  = hp.vec2pix(self.rand_nside, x, y, z, nest=self.rand_nest)
            pix  = np.unique(pix)
            hmap = np.zeros(12*self.rand_nside**2)

            hmap[pix] = 1

            if self.rand_nest:
                hmap      = hu.DensityMap('nest', hmap)
            else:
                hmap      = hu.DensityMap('ring', hmap)

            theta, phi = hmap.genrand(rsize, system='ang')

            if (self.rand_ra_lims is not None):
                ra = phi * 180. / np.pi
                idx = (self.rand_ra_lims[0] < ra ) & (ra < self.rand_ra_lims[1])

                theta = theta[idx]
                phi   = phi[idx]
                rr    = rr[idx]
                gr    = gr[idx]

                del ra, idx

            if (self.rand_dec_lims is not None):
                dec = 90. - theta * 180. / np.pi
                idx = (self.rand_dec_lims[0] < dec ) & (dec < self.rand_dec_lims[1])
                print(gr.shape)
                theta = theta[idx]
                phi   = phi[idx]
                rr    = rr[idx]
                gr    = gr[idx]

                del dec, idx
                print(gr.shape)

            pos        = hp.ang2vec(theta, phi)
            pos       *= rr.reshape(-1,1)

            gr['px']   = pos[:,0]
            gr['py']   = pos[:,1]
            gr['pz']   = pos[:,2]
            gr['redshift'] = rr

        return gr

    def getCartesianRandoms(self, x, y, z, rand_fact=10, sample=0):

        if self.randname is None:
            gr = self.generateCartesianRandoms(x,y,z,rand_fact=rand_fact)
            return gr

        elif sample==0:
            px_field = self.rand_px_field
            py_field = self.rand_py_field
            pz_field = self.rand_pz_field
            weight_field = self.weight_field
            randname     = self.randname
        elif sample==1:
            px_field = self.rand_px_field1
            py_field = self.rand_py_field1
            pz_field = self.rand_pz_field1
            weight_field = self.weight_field1
            randname     = self.randname1

        if weight_field is None:
            gr = fitsio.read(randname, columns=[px_field, py_field, pz_field])
        else:
            gr = fitsio.read(randname, columns=[px_field, py_field, pz_field, weight_field])

        rr = np.sqrt(gr[px_field]**2 + gr[py_field]**2 + gr[pz_field]**2)
        pr = np.sqrt(x**2 + y**2 + z**2)

        rmin = np.min(pr)
        rmax = np.max(pr)

        print('rmin: {}'.format(rmin))
        print('rmax: {}'.format(rmax))

        if (np.max(rr)<rmax) | (np.min(rr)>rmin):
            print("Randoms dont span same radial range as data! Not computing CF")
            return None

        gr = gr[(rmin<rr) & (rr<=rmax)]

        if len(gr)<(len(x)*rand_fact):
            print('Not enough randoms in this bin! Not computing CF')
            return None

        if weight_field:
            rdtype = np.dtype([('px', np.float32), ('py', np.float32),
                               ('pz', np.float32), ('weight',np.float32)])
        else:
            rdtype = np.dtype([('px', np.float32), ('py', np.float32),
                               ('pz', np.float32)])

        rands = np.zeros(len(gr), dtype=rdtype)

        rands['px'] = gr[px_field]
        rands['py'] = gr[py_field]
        rands['pz'] = gr[pz_field]

        if weight_field:
            rands['weight'] = gr[weight_field]

        rands = rands[np.random.choice(np.arange(len(rands)), size=len(x)*rand_fact)]

        return rands


    def generateAngularRandoms(self, aza, pla, z=None, urand_factor=20,
                               rand_factor=10, nside=8,nest=True):
        """
        Generate a set of randoms from a catalog by pixelating the input
        catalog and uniformly distributing random points within the pixels
        occupied by galaxies.

        Also option to assign redshifts with the same distribution as the input
        catalog to allow for generation of randoms once for all z bins.
        """

        dt = aza.dtype

        if self.rand_nside is None:
            nside = 128
        else:
            nside = self.rand_nside

        if z is not None:
            rdtype = np.dtype([('azim_ang', dt), ('polar_ang', dt),
                               ('redshift', dt)])
        else:
            rdtype = np.dtype([('azim_ang', dt), ('polar_ang', dt)])

        rsize = len(aza)*urand_factor

        cpix = hp.ang2pix(nside, (90-pla)*np.pi/180., aza*np.pi/180., nest=self.rand_nest)
        cpix = np.unique(cpix)
        pmap = np.zeros(12 * nside ** 2)

        pmap[cpix] = 1
        if self.rand_nest:
            pmap       = hu.DensityMap('nest', pmap)
        else:
            pmap       = hu.DensityMap('ring', pmap)

        grand = np.zeros(len(aza)*rand_factor, dtype=rdtype)
        grand['azim_ang'], grand['polar_ang'] = pmap.genrand(len(aza)*rand_factor, system='eq')

        if z is not None:
            grand['redshift'] = np.random.choice(z, size=len(aza)*rand_factor)
            zidx = grand['redshift'].argsort()
            grand = grand[zidx]

        return grand

    def addRSD(self, mapunit, cat1=False):

        if not cat1:
            rvec = hp.ang2vec(-(mapunit['polar_ang'] - 90) * np.pi / 180.,
                                mapunit['azim_ang'] * np.pi / 180 )
            vr   = np.sum(rvec * mapunit['velocity'], axis=1)

            return vr + mapunit['redshift']*self.c
        else:
            rvec = hp.ang2vec(-(mapunit['polar_ang1'] - 90) * np.pi / 180.,
                                mapunit['azim_ang1'] * np.pi / 180 )
            vr   = np.sum(rvec * mapunit['velocity1'], axis=1)

            return vr + mapunit['redshift1']*self.c

    def readAngularRandoms(self, fname, aza, pla, z=None, zmin=None, zmax=None,
                            rand_factor=10, dt='>f8',
                            azim_ang_field='RA', polar_ang_field='DEC',
                            weight_field=None, redshift_field=None):
        """
        Use randoms from a file
        """

        ngal = len(aza)
        if self.rand_nside:
            cpix = hp.ang2pix(self.rand_nside, (90 - pla) * np.pi / 180., aza * np.pi / 180,
                              nest=self.rand_nest)
            cpix = np.unique(cpix)

        print('Reading randoms from file: {0}'.format(fname))

        nfields = 2
        if weight_field:
            nfields += 1
        if redshift_field:
            nfields +=1

        try:
            r = np.genfromtxt(fname)

            if r.shape[1]<nfields:
                raise ValueError("Not enough columns in {0} to be a random catalog".format(fname))
            elif (r.shape[1]>nfields):
                warnings.warn("More than {} columns in {}, assuming first two are ra and dec respectively.".format(nfields, fname))

            if weight_field:
                dtype = np.dtype([('azim_ang', dt), ('polar_ang', dt), ('redshift', dt), ('weight', dt)])
            else:
                dtype = np.dtype([('azim_ang', dt), ('polar_ang', dt), ('redshift', dt)])

            rand = np.zeros(ngal*rand_factor, dtype=dtype)

            if redshift_field:
                if zmin is None:
                    zmin = np.min(z)

                if zmax is None:
                    zmax = np.max(z)

                idx = (zmin<r[:,redshift_field]) & (r[:,redshift_field]<=zmax)
                r = r[idx]

            if self.rand_nside:
                pix = hp.ang2pix(self.rand_nside, (90 - r[:,0]) * np.pi / 180.,
                                  r[:,1] * np.pi / 180, nest=self.rand_nest)
                idx = np.in1d(pix, cpix)
                r   = r[idx]


            idx = np.random.choice(np.arange(len(r)), size=ngal*rand_factor,
                                    replace=False)

            rand['azim_ang'] = r[idx,0]
            rand['polar_ang'] = r[idx,1]
            if redshift:
                rand['redshift'] = r[idx,redshift_field]
            if weight:
                rand['redshift'] = r[idx,weight_field]

        except Exception as e:
            cols = [azim_ang_field, polar_ang_field]
            if redshift_field:
                cols.append(redshift_field)
            if weight_field:
                cols.append(weight_field)

            r = fitsio.read(fname, columns=cols)

            if weight_field:
                dtype = np.dtype([('azim_ang', dt), ('polar_ang', dt), ('redshift', dt), ('weight', dt)])
            else:
                dtype = np.dtype([('azim_ang', dt), ('polar_ang', dt), ('redshift', dt)])

            if redshift_field:
                if zmin is None:
                    zmin = np.min(z)

                if zmax is None:
                    zmax = np.max(z)

                idx = (zmin<r[redshift_field]) & (r[redshift_field]<=zmax)
                r = r[idx]

            if self.rand_nside:
                pix = hp.ang2pix(self.rand_nside, (90 - r[polar_ang_field]) * np.pi / 180.,
                                  r[azim_ang_field] * np.pi / 180, nest=self.rand_nest)
                idx = np.in1d(pix, cpix)
                r   = r[idx]

            if len(r)>=ngal*rand_factor:
                nrand = ngal*rand_factor
            else:
                nrand = len(r)
                print('Not enough randoms. Wanted {}, using {}'.format(ngal*rand_factor, nrand))

            rand = np.zeros(nrand, dtype=dtype)

            idx = np.random.choice(np.arange(len(r)), size=nrand, replace=False)
            rand['azim_ang']  = r[azim_ang_field][idx]
            rand['polar_ang'] = r[polar_ang_field][idx]

            if weight_field:
                rand['weight']    = r[weight_field][idx]
            if redshift_field:
                rand['redshift']  = r[redshift_field][idx]

        if (z is not None) & (redshift_field is None):
            rand['redshift'] = np.random.choice(z, size=nrand)

        return rand


    def genbins(self, minb, maxb, nb):

        if self.logbins:
            bins = np.logspace(np.log10(minb), np.log10(maxb), nb+1)
        else:
            bins = np.linspace(minb, maxb, nb+1)

        return bins

    def computeAngularBinsFromRadii(self, rbins, z):
        """
        Computes angular binning appropriate for a particular comoving radial binning
        in Mpc/h
        """

        d_a = self.ministry.cosmo.comoving_distance(z).value * self.ministry.h

        return np.arctan( rbins / d_a )* 180 / np.pi


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

    def deevolve_gal(self, mapunit, Q, faber=False):

        if faber:
            mag = mapunit[self.mkey] - Q * (np.log10(mapunit['redshift'].reshape(len(mapunit['redshift']),1)+1) + 1)
        else:
            mag = mapunit[self.mkey] - Q * (1/(mapunit['redshift'].reshape(len(mapunit['redshift']),1)+1) - 1./1.1)

        return mag

class CrossCorrelationFunction(CorrelationFunction):

    def __init__(self, ministry, zbins1=None, mbins1=None, mcutind1=None,
                    upper_limit1=False, randname1=None, inv_m1=False,
                    clustercatalog=False, rand_azim_ang_field1=None,
                    rand_polar_ang_field1=None, weight_field1=None,
                    rand_px_field1='px', rand_py_field1='py', rand_pz_field1='pz',
                    redshift_field1=None, lightcone=True, weights1=False,
                    no_mkey1=False, **kwargs):

        CorrelationFunction.__init__(self, ministry, lightcone=lightcone, **kwargs)

        self.mcutind1 = mcutind1
        self.mbins1   = mbins1
        self.randname1 = randname1
        self.inv_m1 = inv_m1
        self.upper_limit1 = upper_limit1
        self.no_mkey1 = no_mkey1

        if zbins1 is None:
            self.zbins1 = self.zbins
            self.same_zbins = True
        else:
            self.zbins1 = zbins1
            self.same_zbins = False

        if lightcone & (self.zbins1 is not None) & (not self.same_zbins):
            self.nzbins1 = len(self.zbins1)-1
        else:
            self.nzbins1 = 1

        self.clustercatalog = clustercatalog

        if rand_azim_ang_field1:
            self.rand_azim_ang_field1 = rand_azim_ang_field1
        else:
            self.rand_azim_ang_field1 = self.rand_azim_ang_field

        if rand_polar_ang_field1:
            self.rand_polar_ang_field1 = rand_polar_ang_field1
        else:
            self.rand_polar_ang_field1 = self.rand_polar_ang_field


        if rand_px_field1:
            self.rand_px_field1 = rand_px_field1
        else:
            self.rand_px_field1 = self.rand_px_field

        if rand_py_field1:
            self.rand_py_field1 = rand_py_field1
        else:
            self.rand_py_field1 = self.rand_py_field

        if rand_pz_field1:
            self.rand_pz_field1 = rand_pz_field1
        else:
            self.rand_pz_field1 = self.rand_pz_field

        #weights for the second sample
        self.weights1 = weights1

        #weights and redshifts for the second sample's randoms
        self.weight_field1 = weight_field1
        self.redshift_field1 = redshift_field1

        if self.mbins1 is not None:
            if self.upper_limit1:
                self.nmbins1 = len(self.mbins1)
            else:
                self.nmbins1 = len(self.mbins1) - 1
        else:
            self.nmbins1 = 1

        self.minds1 = np.arange(self.nmbins1)
        if self.inv_m1:
            self.minds1 = self.minds1[::-1]

        if self.no_mkey1:
            self.mkey1 = None
        else:
            if self.catalog_type[1] == 'galaxycatalog':
                if self.appmag:
                    self.mkey1 = 'appmag'
                else:
                    self.mkey1 = 'luminosity'

            elif self.catalog_type[1] == 'halocatalog':
                if self.clustercatalog & (self.mkey !='richness'):
                    self.mkey1 = 'richness'
                else:
                    self.mkey1 = 'halomass'

            else:
                self.mkey1 = None

        self.aschema = ''.join([c.split('catalog')[0] for c in self.catalog_type])


class WxyTheta(CrossCorrelationFunction):

    def __init__(self, ministry,rsd=False,rsd1=False,centrals_only=False,
                    centrals_only1=False,colorcat=None,pccolor=False,
                    cbins=None, cinds=None,abins=None, mintheta=None,
                    maxtheta=None,nabins=None,same_rand1=False,
                    cosmology_flag=None,logbins=True, compute_sigma_g=False,
                    rbins=None,**kwargs):

        CrossCorrelationFunction.__init__(self,ministry,lightcone=True,**kwargs)

        self.compute_sigma_g = compute_sigma_g

        self.c = 299792.458
        self.logbins = logbins

        if not self.compute_sigma_g:

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

        else:

            if rbins is None:
                self.rbins = np.logspace(-1, 1.3, 16)
            else:
                self.rbins = rbins

            self.nabins = len(self.rbins)-1
            self.abins  = np.zeros((self.nrbins+1,self.nzbins1))

        self.same_rand1 = same_rand1
        self.cbins  = cbins
        self.cinds  = cinds
        self.pccolor  = pccolor
        self.colorcat = colorcat

        if self.cbins is None:
            self.ncbins = 1
        else:
            self.ncbins = len(cbins) - 1
            if self.colorcat is None:
                if ('galaxy' in self.catalog_type[0]) & ('galaxy' not in self.catalog_type[1]):
                    self.colorcat = 0
                elif ('galaxy' not in self.catalog_type[0]) & ('galaxy' in self.catalog_type[1]):
                    self.colorcat = 1
                else:
                    #if both galaxy catalogs, then use first
                    self.colorcat = 0


        if cosmology_flag is None:
            self.cosmology_flag = 2
        else:
            self.cosmology_flag = cosmology_flag

        self.jcount = 0

        self.mapkeys = [self.mkey, self.mkey1, 'redshift', 'polar_ang',
                        'azim_ang', 'redshift1', 'polar_ang1', 'azim_ang1']

        self.unitmap = {'polar_ang':'dec', 'azim_ang':'ra', 'redshift':'z',
                        'polar_ang1':'dec', 'azim_ang1':'ra', 'redshift1':'z'}

        if (self.mkey == 'appmag') | (self.mkey=='luminosity'):
            self.unitmap[self.mkey] = 'mag'
        if (self.mkey == 'halomass'):
            self.unitmap[self.mkey] = 'msunh'

        if (self.mkey1 == 'appmag') | (self.mkey1=='luminosity'):
            self.unitmap[self.mkey1] = 'mag'
        if (self.mkey1 == 'halomass'):
            self.unitmap[self.mkey1] = 'msunh'

        self.rsd  = rsd
        self.rsd1 = rsd1

        self.centrals_only  = centrals_only
        self.centrals_only1 = centrals_only1

        if self.rsd:
            self.mapkeys.append('velocity')
            self.unitmap['velocity'] = 'kms'

        if self.rsd1:
            self.mapkeys.append('velocity1')
            self.unitmap['velocity1'] = 'kms'

        if self.centrals_only:
            self.mapkeys.append('central')
            self.unitmap['central'] = 'binary'

        if self.centrals_only1:
            self.mapkeys.append('central1')
            self.unitmap['central1'] = 'binary'

        if self.pccolor:
            self.mapkeys.append('color')

        self.rand_ind = 0
        self.rand_ind1 = 0

        self.nd1 = None
        self.nr1 = None
        self.nd2 = None
        self.nr2 = None

        self.dd = None
        self.dr = None
        self.rd = None
        self.rr = None

    @jackknifeMap
    def map(self, mapunit):
        if (self.ncbins > 1) & (~self.pccolor):
            if self.colorcat==0:
                clr = mapunit[self.mkey][:,self.cinds[0]] - mapunit[self.mkey][:,self.cinds[1]]
            else:
                clr = mapunit[self.mkey1][:,self.cinds[0]] - mapunit[self.mkey1][:,self.cinds[1]]
        elif self.pccolor:
            clr = mapunit['color']

        if self.nd1 is None:
            self.writeCorrfuncBinFile(self.abins, binfilename='angular_bins')
            self.nd1 = np.zeros((self.njack, self.ncbins, self.nmbins,
                                  self.nmbins1, self.nzbins * self.nzbins1))
            self.nr1 = np.zeros((self.njack, self.ncbins, self.nmbins,
                                  self.nmbins1, self.nzbins * self.nzbins1))
            self.nd2 = np.zeros((self.njack, self.ncbins, self.nmbins,
                                  self.nmbins1, self.nzbins * self.nzbins1))
            self.nr2 = np.zeros((self.njack, self.ncbins, self.nmbins,
                                  self.nmbins1, self.nzbins * self.nzbins1))

            self.dd = np.zeros((self.njack, self.nabins, self.ncbins,
                                  self.nmbins, self.nmbins1, self.nzbins * self.nzbins1))
            self.dr = np.zeros((self.njack, self.nabins, self.ncbins,
                                  self.nmbins, self.nmbins1, self.nzbins * self.nzbins1))
            self.rd = np.zeros((self.njack, self.nabins, self.ncbins,
                                  self.nmbins, self.nmbins1, self.nzbins * self.nzbins1))
            self.rr = np.zeros((self.njack, self.nabins, self.ncbins,
                                  self.nmbins, self.nmbins1, self.nzbins * self.nzbins1))

        mu = {}
        mu['azim_ang'] = np.zeros(len(mapunit['azim_ang']), dtype=np.float64)
        mu['polar_ang'] = np.zeros(len(mapunit['polar_ang']), dtype=np.float64)
        mu['redshift'] = np.zeros(len(mapunit['redshift']), dtype=np.float64)

        mu['azim_ang1'] = np.zeros(len(mapunit['azim_ang1']), dtype=np.float64)
        mu['polar_ang1'] = np.zeros(len(mapunit['polar_ang1']), dtype=np.float64)
        mu['redshift1'] = np.zeros(len(mapunit['redshift1']), dtype=np.float64)

        mu['azim_ang'][:] = mapunit['azim_ang'][:].astype(np.float64)
        mu['polar_ang'][:] = mapunit['polar_ang'][:].astype(np.float64)
        mu['redshift'][:] = mapunit['redshift'][:].astype(np.float64)

        mu['azim_ang1'][:] = mapunit['azim_ang1'][:].astype(np.float64)
        mu['polar_ang1'][:] = mapunit['polar_ang1'][:].astype(np.float64)
        mu['redshift1'][:] = mapunit['redshift1'][:].astype(np.float64)


        for f in self.mapkeys:
            if ('polar_ang' in f) | ('azim_ang' in f) | ('redshift' in f) : continue
            mu[f] = mapunit[f]

        if self.rsd:
            z = self.addRSD(mu) / self.c
        else:
            z = mu['redshift']

        if self.rsd1:
            z1 = self.addRSD(mu, cat1=True) / self.c
        else:
            z1 = mu['redshift1']

        for i in range(self.nzbins):
            print('Finding redshift indices')

            zlidx = z.searchsorted(self.zbins[i])
            zhidx = z.searchsorted(self.zbins[i+1])


            for i1 in range(self.nzbins1):
                if self.same_zbins:
                    zlidx1 = z1.searchsorted(self.zbins[i])
                    zhidx1 = z1.searchsorted(self.zbins[i+1])
                else:
                    zlidx1 = z1.searchsorted(self.zbins1[i1])
                    zhidx1 = z1.searchsorted(self.zbins1[i1+1])

                if self.compute_sigma_g:
                    if self.same_zbins:
                        zm = (self.zbins[i] + self.zbins[i+1]) / 2
                    else:
                        zm = (self.zbins1[i] + self.zbins1[i+1]) / 2

                    self.abins[:,i] = self.computeAngularBinsFromRadii(self.rbins, zm)
                    self.writeCorrfuncBinFile(self.abins[:,i], binfilename='angular_bins_{}'.format(i))


                if (zlidx==zhidx):
                    print("No galaxies in redshift bin {0} to {1}".format(self.zbins[i], self.zbins[i+1]))
                    print('z: {}'.format(z))

                    print("Min and max z: {0}, {1}".format(np.min(z), np.max(z)))
                    continue
                elif (zlidx1==zhidx1):
                    print("No galaxies in redshift bin {0} to {1}".format(self.zbins1[i], self.zbins1[i+1]))
                    print('z1: {}'.format(z1))

                    print("Min and max z1: {0}, {1}".format(np.min(z1), np.max(z1)))
                    continue

                if self.ncbins>1:
                    if (self.splitcolor is None) & (self.bimodal_ccut):
                        if self.colorcat==0:
                            ccounts, cbins = np.histogram(clr[zlidx:zhidx], self.hcbins)
                        else:
                            ccounts, cbins = np.histogram(clr[zlidx1:zhidx1], self.hcbins)
                        self.splitcolor = self.splitBimodal(cbins[:-1], ccounts)

                    elif (self.splitcolor is None) & (self.percentile_ccut is not None):
                        if self.colorcat==0:
                            self.splitcolor = self.splitPercentile(clr[zlidx:zhidx], self.percentile_ccut)
                        else:
                            self.splitcolor = self.splitPercentile(clr[zlidx1:zhidx1], self.percentile_ccut)

                for li, j in enumerate(self.minds):
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

                        rands = self.getRandoms(mu['azim_ang'][zlidx:zhidx][lidx],
                                                mu['polar_ang'][zlidx:zhidx][lidx],
                                                z[zlidx:zhidx][lidx],
                                                zmin=self.zbins[i], zmax=self.zbins[i+1])

                    for li1, j1 in enumerate(self.minds1):
                        if self.mcutind1 is not None:
                            if self.upper_limit1:
                                lidx1 = mu[self.mkey1][zlidx1:zhidx1,self.mcutind1] < self.mbins1[j1]
                            else:
                                lidx1 = (self.mbins1[j1] <= mu[self.mkey1][zlidx1:zhidx1,self.mcutind1]) & (mu[self.mkey1][zlidx1:zhidx1,self.mcutind1] < self.mbins1[j1+1])
                        else:
                            if self.upper_limit1:
                                lidx1 = mu[self.mkey1][zlidx1:zhidx1] < self.mbins1[j1]
                            else:
                                lidx1 = (self.mbins1[j1] <= mu[self.mkey1][zlidx1:zhidx1]) & (mu[self.mkey1][zlidx1:zhidx1] < self.mbins1[j1+1])

                        if self.centrals_only1:
                            lidx1 = lidx1 & (mu['central1'][zlidx1:zhidx1]==1)

                        if (li1==self.rand_ind1) | (not self.same_rand1):
                            print('Generating Randoms')
                            if len(z1[zlidx1:zhidx1][lidx1])==0:
                                self.rand_ind1+=1
                                continue

                            rands1 = self.getRandoms(mu['azim_ang1'][zlidx1:zhidx1][lidx1],
                                                     mu['polar_ang1'][zlidx1:zhidx1][lidx1],
                                                     z1[zlidx1:zhidx1][lidx1], sample=1,
                                                     zmin=self.zbins1[i1], zmax=self.zbins1[i1+1])

                        for k in range(self.ncbins):
                            if self.ncbins == 1:
                                cidx = lidx
                            else:
                                if k==0:
                                    cidx = lidx & (self.splitcolor < clr[zlidx:zhidx])
                                else:
                                    cidx = lidx & (self.splitcolor >= clr[zlidx:zhidx])

                            print('i*self.nzbins1 + i1: {}'.format(i*self.nzbins1 + i1))

                            self.nd1[self.jcount,k,j,j1,i*self.nzbins1 + i1] = len(mu['azim_ang'][zlidx:zhidx][cidx])
                            self.nd2[self.jcount,k,j,j1,i*self.nzbins1 + i1] = len(mu['azim_ang1'][zlidx1:zhidx1][lidx1])

                            self.nr1[self.jcount,k,j,j1,i*self.nzbins1 + i1] = len(rands)
                            self.nr2[self.jcount,k,j,j1,i*self.nzbins1 + i1] = len(rands1)

                            print("Number of cat1 in this z/lum bin: {0}".format(self.nd1[self.jcount,k,j,j1,i*self.nzbins1 + i1]))
                            print("Number of cat2 in this z/lum bin: {0}".format(self.nd2[self.jcount,k,j,j1,i*self.nzbins1 + i1]))

                            print("Number of rands1 in this z/lum bin: {0}".format(self.nr1[self.jcount,k,j,j1,i*self.nzbins1 + i1]))
                            print("Number of rands2 in this z/lum bin: {0}".format(self.nr2[self.jcount,k,j,j1,i*self.nzbins1 + i1]))

                            #data data
                            print('calculating data data pairs')
                            sys.stdout.flush()
                            if (self.nd1[self.jcount,k,j,j1,i*self.nzbins1 +i1]<2) | (self.nd2[self.jcount,k,j,j1,i*self.nzbins1 + i1]<2):
                                continue

                            ddresults = DDtheta_mocks(0, self.nthreads, self.binfilename,
                                                    mu['azim_ang'][zlidx:zhidx][cidx],
                                                    mu['polar_ang'][zlidx:zhidx][cidx],
                                                    RA2=mu['azim_ang1'][zlidx1:zhidx1][lidx1],
                                                    DEC2=mu['polar_ang1'][zlidx1:zhidx1][lidx1])


                            self.dd[self.jcount,:,k,j,j1,i*self.nzbins1 + i1] = ddresults['npairs']

                            #data randoms
                            print('calculating data random pairs')
                            sys.stdout.flush()
                            if self.weight_field1:
                                drresults = DDtheta_mocks(0, self.nthreads, self.binfilename,
                                                          mu['azim_ang'][zlidx:zhidx][cidx],
                                                          mu['polar_ang'][zlidx:zhidx][cidx],
                                                          weights1=np.ones_like(mu['polar_ang'][zlidx:zhidx][cidx],dtype=np.float64),
                                                          RA2=rands1['azim_ang'],
                                                          DEC2=rands1['polar_ang'],
                                                          weights2=rands1['weight'],
                                                          weight_type='pair_product')
                            else:
                                drresults = DDtheta_mocks(0, self.nthreads, self.binfilename,
                                                          mu['azim_ang'][zlidx:zhidx][cidx],
                                                          mu['polar_ang'][zlidx:zhidx][cidx],
                                                          RA2=rands1['azim_ang'],
                                                          DEC2=rands1['polar_ang'])
                                drresults['weightavg'] = 1

                            print('weightavg: {}'.format(drresults['weightavg']))
                            self.dr[self.jcount,:,k,j,j1,i*self.nzbins1 + i1] = drresults['npairs'] * drresults['weightavg']

                            print('calculating random data pairs')
                            sys.stdout.flush()
                            if self.weight_field:
                                drresults = DDtheta_mocks(0, self.nthreads, self.binfilename,
                                                          mu['azim_ang1'][zlidx1:zhidx1][lidx1],
                                                          mu['polar_ang1'][zlidx1:zhidx1][lidx1],
                                                          weights1=np.ones_like(mu['polar_ang1'][zlidx1:zhidx1][lidx1],dtype=np.float64),
                                                          RA2=rands['azim_ang'],
                                                          DEC2=rands['polar_ang'],
                                                          weights2=rands['weight'],
                                                          weight_type='pair_product')

                            else:
                                drresults = DDtheta_mocks(0, self.nthreads, self.binfilename,
                                                          mu['azim_ang1'][zlidx1:zhidx1][lidx1],
                                                          mu['polar_ang1'][zlidx1:zhidx1][lidx1],
                                                          RA2=rands['azim_ang'],
                                                          DEC2=rands['polar_ang'])
                                drresults['weightavg'] = 1


                            print('weightavg: {}'.format(drresults['weightavg']))
                            self.rd[self.jcount,:,k,j,j1,i*self.nzbins1 + i1] = drresults['npairs'] * drresults['weightavg']


                            #randoms randoms
                            print('calculating random random pairs')
                            sys.stdout.flush()
                            if (li==self.rand_ind) & (li1==self.rand_ind1) | (not self.same_rand):
                                if (self.weight_field is not None) & (self.weight_field1 is not None):
                                    rrresults = DDtheta_mocks(0, self.nthreads, self.binfilename,
                                                              rands['azim_ang'],
                                                              rands['polar_ang'],
                                                              weights1=rands['weight'],
                                                              RA2=rands1['azim_ang'],
                                                              DEC2=rands1['polar_ang'],
                                                              weights2=rands1['weight'],
                                                              weight_type='pair_product')
                                elif self.weight_field:
                                    rrresults = DDtheta_mocks(0, self.nthreads, self.binfilename,
                                                              rands['azim_ang'],
                                                              rands['polar_ang'],
                                                              weights1=rands['weight'],
                                                              RA2=rands1['azim_ang'],
                                                              DEC2=rands1['polar_ang'],
                                                              weights2=np.ones(len(rands1)),
                                                              weight_type='pair_product')
                                elif self.weight_field1:
                                    rrresults = DDtheta_mocks(0, self.nthreads, self.binfilename,
                                                              rands['azim_ang'],
                                                              rands['polar_ang'],
                                                              weights1=np.ones(len(rands)),
                                                              RA2=rands1['azim_ang'],
                                                              DEC2=rands1['polar_ang'],
                                                              weights2=rands1['weight'],
                                                              weight_type='pair_product')

                                else:
                                    rrresults = DDtheta_mocks(0, self.nthreads, self.binfilename,
                                                              rands['azim_ang'],
                                                              rands['polar_ang'],
                                                              RA2=rands1['azim_ang'],
                                                              DEC2=rands1['polar_ang'])
                                    rrresults['weightavg'] = 1

                            print('weightavg: {}'.format(rrresults['weightavg']))
                            self.rr[self.jcount,:,k,j,j1,i*self.nzbins1 + i1] = rrresults['npairs'] * rrresults['weightavg']


    def reduce(self, rank=None, comm=None):

        if rank is not None:
            gnd1 = comm.gather(self.nd1, root=0)
            gnr1 = comm.gather(self.nr1, root=0)
            gnd2 = comm.gather(self.nd2, root=0)
            gnr2 = comm.gather(self.nr2, root=0)

            gdd = comm.gather(self.dd, root=0)
            gdr = comm.gather(self.dr, root=0)
            grd = comm.gather(self.rd, root=0)
            grr = comm.gather(self.rr, root=0)


            if rank==0:
                nd1shape = [self.nd1.shape[i] for i in range(len(self.nd1.shape))]
                nr1shape = [self.nr1.shape[i] for i in range(len(self.nr1.shape))]
                nd2shape = [self.nd2.shape[i] for i in range(len(self.nd2.shape))]
                nr2shape = [self.nr2.shape[i] for i in range(len(self.nr2.shape))]

                ddshape = [self.dd.shape[i] for i in range(len(self.dd.shape))]
                drshape = [self.dr.shape[i] for i in range(len(self.dr.shape))]
                rdshape = [self.rd.shape[i] for i in range(len(self.rd.shape))]
                rrshape = [self.rr.shape[i] for i in range(len(self.rr.shape))]

                nd1shape.insert(1,1)
                nr1shape.insert(1,1)
                nd2shape.insert(1,1)
                nr2shape.insert(1,1)

                nd1shape[0] = self.njacktot
                nr1shape[0] = self.njacktot
                nd2shape[0] = self.njacktot
                nr2shape[0] = self.njacktot

                ddshape[0] = self.njacktot
                drshape[0] = self.njacktot
                rdshape[0] = self.njacktot
                rrshape[0] = self.njacktot

                self.nd1 = np.zeros(nd1shape)
                self.nr1 = np.zeros(nr1shape)
                self.nd2 = np.zeros(nd2shape)
                self.nr2 = np.zeros(nr2shape)

                self.dd = np.zeros(ddshape)
                self.dr = np.zeros(drshape)
                self.rd = np.zeros(rdshape)
                self.rr = np.zeros(rrshape)

                jc = 0
                for i, g in enumerate(gnd1):
                    if g is None: continue
                    nj = g.shape[0]
                    self.nd1[jc:jc+nj,0,:,:,:,:] = g
                    self.nr1[jc:jc+nj,0,:,:,:,:] = gnr1[i]
                    self.nd2[jc:jc+nj,0,:,:,:,:] = gnd2[i]
                    self.nr2[jc:jc+nj,0,:,:,:,:] = gnr2[i]

                    self.dd[jc:jc+nj,:,:,:,:,:] = gdd[i]
                    self.dr[jc:jc+nj,:,:,:,:,:] = gdr[i]
                    self.rd[jc:jc+nj,:,:,:,:,:] = grd[i]
                    self.rr[jc:jc+nj,:,:,:,:,:] = grr[i]

                    jc += nj

                self.jwtheta = np.zeros(self.dd.shape)

                self.jnd1 = self.jackknife(self.nd1, reduce_jk=False)
                self.jnr1 = self.jackknife(self.nr1, reduce_jk=False)
                self.jnd2 = self.jackknife(self.nd2, reduce_jk=False)
                self.jnr2 = self.jackknife(self.nr2, reduce_jk=False)

                self.jdd = self.jackknife(self.dd, reduce_jk=False)
                self.jdr = self.jackknife(self.dr, reduce_jk=False)
                self.jrd = self.jackknife(self.rd,reduce_jk=False)
                self.jrr = self.jackknife(self.rr, reduce_jk=False)

                self.jwxytheta = np.zeros_like(self.jdd)

                for i in range(self.njacktot):
                    for j in range(self.ncbins):
                        for k in range(self.nmbins):
                            for l in range(self.nmbins1):
                                for m in range(self.nzbins*self.nzbins1):
                                    self.jwxytheta[i,:,j,k,l,m] = convert_3d_counts_to_cf(self.jnd1[i,0,j,k,l,m],
                                                                                  self.jnd2[i,0,j,k,l,m],
                                                                                self.jnr1[i,0,j,k,l,m],
                                                                                self.jnr2[i,0,j,k,l,m],
                                                                                self.jdd[i,:,j,k,l,m],
                                                                                self.jdr[i,:,j,k,l,m],
                                                                                self.jrd[i,:,j,k,l,m],
                                                                                self.jrr[i,:,j,k,l,m])

                self.wxytheta = np.sum(self.jwxytheta, axis=0) / self.njacktot

                self.varwxytheta = np.sum((self.jwxytheta - self.wxytheta)**2, axis=0) * (self.njacktot - 1) / self.njacktot

                print('sizes of jnd1, jdd, wxytheta: {}, {}, {}'.format(self.jnd1.shape, self.jdd.shape, self.wxytheta.shape))
                if self.compute_sigma_g:

                    self.zmid = (self.zbins1[1:] + self.zbins1[:-1]) / 2
                    W = self.jrr*(self.jnd1/self.jnr1*self.jnd2/self.jnr2)

                    if self.jtype is not None:
                        area = self.ministry.galaxycatalog.getArea(jackknife=True).reshape(-1,1)
                    else:
                        area = self.ministry.galaxycatalog.getArea(jackknife=False).reshape(-1,1)

                    area_mpch = area*((np.pi/180.)**2
                                      *(self.ministry.cosmo.comoving_distance(self.zmid).value * self.ministry.h)**2).reshape(1,-1)

                    try:
                        self.jave_gdens  = self.jnd1 / area_mpch.reshape(self.njacktot, 1, 1, 1, 1, self.nzbins1)
                        jdens            = np.sum(self.jave_gdens * self.jnd2 / np.sum(self.jnd2, axis=5), axis=5)
                        self.jsigma_g    = np.sum(self.jwxytheta * W, axis=-1) / np.sum(W, axis=-1) * jdens

                        self.sigma_g     = np.sum(self.jsigma_g, axis=0) / self.njacktot
                        self.varsigma_g = np.sum((self.jsigma_g - self.sigma_g)**2, axis=0) * (self.njacktot - 1) / self.njacktot
                    except:
                        pass

        else:
            self.jwtheta = np.zeros(self.dd.shape)

            self.jnd1 = self.jackknife(self.nd1, reduce_jk=False).reshape(-1,1,self.ncbins,self.nmbins,self.nmbins1, self.nzbins*self.nzbins1)
            self.jnr1 = self.jackknife(self.nr1, reduce_jk=False).reshape(-1,1,self.ncbins,self.nmbins,self.nmbins1, self.nzbins*self.nzbins1)
            self.jnd2 = self.jackknife(self.nd2, reduce_jk=False).reshape(-1,1,self.ncbins,self.nmbins,self.nmbins1, self.nzbins*self.nzbins1)
            self.jnr2 = self.jackknife(self.nr2, reduce_jk=False).reshape(-1,1,self.ncbins,self.nmbins,self.nmbins1, self.nzbins*self.nzbins1)

            self.jdd = self.jackknife(self.dd, reduce_jk=False)
            self.jdr = self.jackknife(self.dr, reduce_jk=False)
            self.jrd = self.jackknife(self.rd,reduce_jk=False)
            self.jrr = self.jackknife(self.rr, reduce_jk=False)

            self.jwxytheta = np.zeros_like(self.jdd)

            for i in range(self.njacktot):
                for j in range(self.ncbins):
                    for k in range(self.nmbins):
                        for l in range(self.nmbins1):
                            for m in range(self.nzbins*self.nzbins1):
                                self.jwxytheta[i,:,j,k,l,m] = convert_3d_counts_to_cf(self.jnd1[i,0,j,k,l,m],
                                                                                      self.jnd2[i,0,j,k,l,m],
                                                                                      self.jnr1[i,0,j,k,l,m],
                                                                                      self.jnr2[i,0,j,k,l,m],
                                                                                      self.jdd[i,:,j,k,l,m],
                                                                                      self.jdr[i,:,j,k,l,m],
                                                                                      self.jrd[i,:,j,k,l,m],
                                                                                      self.jrr[i,:,j,k,l,m])

            self.wxytheta = np.sum(self.jwxytheta, axis=0) / self.njacktot

            self.varwxytheta = np.sum((self.jwxytheta - self.wxytheta)**2, axis=0) * (self.njacktot - 1) / self.njacktot

            if self.compute_sigma_g:

                self.zmid = (self.zbins1[1:] + self.zbins1[:-1]) / 2
                W = self.jrr*(self.jnd1/self.jnr1*self.jnd2/self.jnr2)

                if self.jtype is not None:
                    area = self.ministry.galaxycatalog.getArea(jackknife=True).reshape(-1,1)
                else:
                    area = self.ministry.galaxycatalog.getArea(jackknife=False).reshape(-1,1)

                area_mpch = area*((np.pi/180.)**2
                                  *(self.ministry.cosmo.comoving_distance(self.zmid).value * self.ministry.h)**2).reshape(1,-1)

                self.jave_gdens  = self.jnd1 / area_mpch.reshape(self.njacktot, 1, 1, 1, 1, self.nzbins1)
                jdens            = np.sum(self.jave_gdens * self.jnd2 / np.sum(self.jnd2, axis=5), axis=5)
                self.jsigma_g    = np.sum(self.jwxytheta * W, axis=-1) / np.sum(W, axis=-1) * jdens

                self.sigma_g     = np.sum(self.jsigma_g, axis=0) / self.njacktot
                self.varsigma_g = np.sum((self.jsigma_g - self.sigma_g)**2, axis=0) * (self.njacktot - 1) / self.njacktot



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
                    ye = np.sqrt(self.varwxytheta[:,c,l,z])
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


class WTheta(CorrelationFunction):

    def __init__(self, ministry, zbins=None, mbins=None,
                  abins=None, mintheta=None, maxtheta=None,
                  logbins=True, nabins=None, subjack=False,
                  catalog_type=None, tag=None, mcutind=None,
                  same_rand=False, inv_m=True, cosmology_flag=None,
                  bimodal_ccut=False, percentile_ccut=None,
                  precompute_color=False, upper_limit=False,
                  centrals_only=False, rsd=False,
                  homogenize_type=True,
                  deevolve_mstar=False, Q=None, faber=False,
                  **kwargs):
        """
        Angular correlation function, w(theta), for use with non-periodic
        data.
        """
        CorrelationFunction.__init__(self, ministry, zbins=zbins,
                                      lightcone=True, mbins=mbins,
                                      nrbins=nabins, subjack=subjack,
                                      mcutind=mcutind, same_rand=same_rand,
                                      inv_m=inv_m,catalog_type=catalog_type,
                                      tag=tag, upper_limit=upper_limit,
                                      **kwargs)

        self.bimodal_ccut = bimodal_ccut
        self.percentile_ccut = percentile_ccut
        self.splitcolor = None
        self.homogenize_type = homogenize_type
        self.deevolve_mstar = deevolve_mstar
        self.faber = faber

        if self.deevolve_mstar & (Q is None):
            raise(ValueError("Must provide Q is deevolve_mstar == True"))
        else:
            self.Q = Q

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

        if cosmology_flag is None:
            self.cosmology_flag = 2
        else:
            self.cosmology_flag = cosmology_flag

        self.jcount = 0

        self.mapkeys = [self.mkey, 'redshift', 'polar_ang', 'azim_ang']

        if self.catalog_type == ['galaxycatalog']:
            self.unitmap = {self.mkey:'mag', 'polar_ang':'dec', 'azim_ang':'ra', 'redshift':'z'}
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

        if self.weights:
            self.mapkeys.append('weight')

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
            self.writeCorrfuncBinFile(self.abins, binfilename='angular_bins')
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

        mu = {}
        mu['azim_ang'] = np.zeros(len(mapunit['azim_ang']), dtype=np.float64)
        mu['polar_ang'] = np.zeros(len(mapunit['polar_ang']), dtype=np.float64)
        mu['redshift'] = np.zeros(len(mapunit['redshift']), dtype=np.float64)

        mu['azim_ang'][:] = mapunit['azim_ang'][:].astype(np.float64)
        mu['polar_ang'][:] = mapunit['polar_ang'][:].astype(np.float64)
        mu['redshift'][:] = mapunit['redshift'][:].astype(np.float64)


        for f in self.mapkeys:
            if (f=='polar_ang') | (f=='azim_ang') | (f=='redshift') : continue
            mu[f] = mapunit[f]

        if self.deevolve_mstar:
            lum = self.deevolve_gal(mu, self.Q, faber=self.faber)
        else:
            lum = mu[self.mkey]

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
                print(z)
                print("Min and max z: {0}, {1}".format(np.min(z), np.max(z)))
                continue

            if (self.splitcolor is None) & (self.bimodal_ccut):
                ccounts, cbins = np.histogram(clr[zlidx:zhidx], self.hcbins)
                self.splitcolor = self.splitBimodal(cbins[:-1], ccounts)

            for li, j in enumerate(self.minds):
                print('Finding luminosity indices')

                if self.mcutind is not None:
                    if self.upper_limit:
                        lidx = lum[zlidx:zhidx,self.mcutind] < self.mbins[j]
                    else:
                        lidx = (self.mbins[j] <= lum[zlidx:zhidx,self.mcutind]) & (lum[zlidx:zhidx,self.mcutind] < self.mbins[j+1])
                else:
                    if self.upper_limit:
                        lidx = lum[zlidx:zhidx] < self.mbins[j]
                    else:
                        lidx = (self.mbins[j] <= lum[zlidx:zhidx]) & (lum[zlidx:zhidx] < self.mbins[j+1])

                if self.centrals_only:
                    lidx = lidx & (mu['central'][zlidx:zhidx]==1)

                if (li==self.rand_ind) | (not self.same_rand):
                    print('Generating Randoms')
                    if len(z[zlidx:zhidx][lidx])==0:
                        self.rand_ind+=1
                        continue

                    rands = self.getRandoms(mu['azim_ang'][zlidx:zhidx][lidx],
                                            mu['polar_ang'][zlidx:zhidx][lidx],z[zlidx:zhidx][lidx],
                                            zmin=self.zbins[i], zmax=self.zbins[i+1])

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

                    if self.weights:
                        ddresults = DDtheta_mocks(1, self.nthreads, self.binfilename,
                                                  mu['azim_ang'][zlidx:zhidx][cidx],
                                                  mu['polar_ang'][zlidx:zhidx][cidx],
                                                  weights1=mu['weight'][zlidx:zhidx][cidx])
                        self.dd[self.jcount,:,k,j,i] = ddresults['npairs'] * ddresults['weightavg']
                    else:
                        ddresults = DDtheta_mocks(1, self.nthreads, self.binfilename,
                                                  mu['azim_ang'][zlidx:zhidx][cidx],
                                                  mu['polar_ang'][zlidx:zhidx][cidx])
                        self.dd[self.jcount,:,k,j,i] = ddresults['npairs']


                    #data randoms
                    print('calculating data random pairs')
                    sys.stdout.flush()
                    if (self.weight_field) & (self.weights):
                        drresults = DDtheta_mocks(0, self.nthreads, self.binfilename,
                                                  mu['azim_ang'][zlidx:zhidx][cidx],
                                                  mu['polar_ang'][zlidx:zhidx][cidx],
                                                  weights1=mu['weight'][zlidx:zhidx][cidx],
                                                  RA2=rands['azim_ang'],
                                                  DEC2=rands['polar_ang'],
                                                  weights2=rands['weight'])

                        self.dr[self.jcount,:,k,j,i] = drresults['npairs'] * drresults['weightavg']
                    elif self.weights:
                        drresults = DDtheta_mocks(0, self.nthreads, self.binfilename,
                                                  mu['azim_ang'][zlidx:zhidx][cidx],
                                                  mu['polar_ang'][zlidx:zhidx][cidx],
                                                  weights1=mu['weight'][zlidx:zhidx][cidx],
                                                  RA2=rands['azim_ang'],
                                                  DEC2=rands['polar_ang'])

                        self.dr[self.jcount,:,k,j,i] = drresults['npairs'] * drresults['weightavg']
                    elif self.weight_field:
                        drresults = DDtheta_mocks(0, self.nthreads, self.binfilename,
                                                  mu['azim_ang'][zlidx:zhidx][cidx],
                                                  mu['polar_ang'][zlidx:zhidx][cidx],
                                                  RA2=rands['azim_ang'],
                                                  DEC2=rands['polar_ang'],
                                                  weights2=rands['weight'])

                        self.dr[self.jcount,:,k,j,i] = drresults['npairs'] * drresults['weightavg']

                    else:
                        drresults = DDtheta_mocks(0, self.nthreads, self.binfilename,
                                                  mu['azim_ang'][zlidx:zhidx][cidx],
                                                  mu['polar_ang'][zlidx:zhidx][cidx],
                                                  RA2=rands['azim_ang'],
                                                  DEC2=rands['polar_ang'])


                        self.dr[self.jcount,:,k,j,i] = drresults['npairs']

                    #randoms randoms
                    print('calculating random random pairs')
                    sys.stdout.flush()
                    if (li==0) | (not self.same_rand):
                        if self.weight_field:
                            rrresults = DDtheta_mocks(1, self.nthreads, self.binfilename,
                                                      rands['azim_ang'],
                                                      rands['polar_ang'],
                                                      weights1=rands['weight'])
                        else:
                            rrresults = DDtheta_mocks(1, self.nthreads, self.binfilename,
                                                      rands['azim_ang'],
                                                      rands['polar_ang'])

                    if self.weight_field:
                        self.rr[self.jcount,:,k,j,i] = rrresults['npairs'] * rrresults['weightavg']
                    else:
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

                fnorm = (self.jnr / self.jnd).reshape(-1,1,self.ncbins,self.nmbins,self.nzbins)

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

            fnorm = (self.jnr / self.jnd).reshape(-1,1,self.ncbins,self.nmbins,self.nzbins)

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
                  deevolve_mstar=False,
                  faber=False, Q=None, CMASS=False, splitcolor=None,
                  cinds=None,cbins=None,**kwargs):
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

        self.deevolve_mstar = deevolve_mstar
        self.faber = faber

        if self.deevolve_mstar & (Q is None):
            raise(ValueError("Must provide Q is deevolve_mstar == True"))
        else:
            self.Q = Q

        self.bimodal_ccut = bimodal_ccut
        self.percentile_ccut = percentile_ccut
        self.splitcolor = splitcolor

        if self.bimodal_ccut:
            self.hcbins = 100
            self.ncbins = 2
        elif self.percentile_ccut is not None:
            self.ncbins = 2
        elif self.splitcolor is not None:
            self.ncbins = 2
            if cbins is None:
                self.cbins = np.linspace(-0.2,1.2,60)
            else:
                self.cbins = cbins
        else:
            self.ncbins = 1

        if self.ncbins > 1:
            if cinds is None:
                self.cinds = [0,1]
            else:
                self.cinds = cinds

        self.pccolor = precompute_color
        self.centrals_only = centrals_only
        self.CMASS = CMASS

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

        self.jcount = 0

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
            self.unitmap['central'] = 'binary'

        if self.pccolor:
            self.mapkeys.append('color')

        if self.CMASS:
            self.mapkeys.append('appmag')
            self.unitmap['appmag'] = 'mag'

        self.rand_ind = 0

        self.nd = None
        self.nr = None
        self.dd = None
        self.dr = None
        self.rr = None


    @jackknifeMap
    def map(self, mapunit):

        if not hascorrfunc:
            raise(ImportError("CorrFunc is required to calculate wp(rp)"))

        if (self.ncbins > 1) & (~self.pccolor):
            clr = mapunit['luminosity'][:,self.cinds[0]] - mapunit['luminosity'][:,self.cinds[1]]
        elif self.pccolor:
            clr = mapunit['color']

        if self.dd is None:
            self.writeCorrfuncBinFile(self.rbins)
            self.dd = np.zeros((self.njack, self.nrbins, int(self.pimax), self.ncbins, self.nmbins, self.nzbins))
            self.dr = np.zeros((self.njack, self.nrbins, int(self.pimax), self.ncbins, self.nmbins, self.nzbins))
            self.rr = np.zeros((self.njack, self.nrbins, int(self.pimax), self.ncbins, self.nmbins, self.nzbins))
            self.nd = np.zeros((self.njack, self.ncbins, self.nmbins, self.nzbins))
            self.nr = np.zeros((self.njack, self.ncbins, self.nmbins, self.nzbins))


        mu = {}
        mu['azim_ang'] = np.zeros(len(mapunit['azim_ang']), dtype=np.float64)
        mu['polar_ang'] = np.zeros(len(mapunit['polar_ang']), dtype=np.float64)
        mu['redshift'] = np.zeros(len(mapunit['redshift']), dtype=np.float64)

        mu['azim_ang'][:] = mapunit['azim_ang'][:].astype(np.float64)
        mu['polar_ang'][:] = mapunit['polar_ang'][:].astype(np.float64)
        mu['redshift'][:] = mapunit['redshift'][:].astype(np.float64)

        for f in self.mapkeys:
            if (f=='polar_ang') | (f=='azim_ang') | (f=='redshift') : continue
            mu[f] = mapunit[f]

        if self.rsd:
            mu['velocity'] = np.zeros((len(mapunit['velocity']),3), dtype=np.float64)
            mu['velocity'][:] = mapunit['velocity'][:]

        if (self.CMASS) & (self.mkey is not 'appmag'):
            mu['appmag'] = mapunit['appmag']

        if self.deevolve_mstar:
            lum = self.deevolve_gal(mu, self.Q, faber=self.faber)
        else:
            lum = mu[self.mkey]

        if self.rsd:
            cz = self.addRSD(mu)
        else:
            cz = (mu['redshift'] * self.c).astype(np.float64)


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

            if self.CMASS:
                lidx = self.selectCMASS(mu['appmag'][zlidx:zhidx])

            for li, j in enumerate(self.minds):
                print('Finding luminosity indices')

                if self.CMASS:
                    pass
                elif self.mcutind is not None:
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

                    rands = self.getRandoms(mu['azim_ang'][zlidx:zhidx][lidx],
                                            mu['polar_ang'][zlidx:zhidx][lidx],
                                            z=cz[zlidx:zhidx][lidx],
                                            zmin=self.zbins[i]*self.c,
                                            zmax=self.zbins[i+1]*self.c)

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

                    print(mu['azim_ang'][zlidx:zhidx][cidx])
                    print(cz[zlidx:zhidx][cidx])

                    ddresults = DDrppi_mocks(1,
                                            self.cosmology_flag, self.nthreads,
                                            self.pimax,
                                            self.binfilename,
                                            mu['azim_ang'][zlidx:zhidx][cidx],
                                            mu['polar_ang'][zlidx:zhidx][cidx],
                                            cz[zlidx:zhidx][cidx])


                    self.dd[self.jcount,:,:,k,j,i] = ddresults['npairs'].reshape(-1,int(self.pimax))

                    #data randoms
                    print('calculating data random pairs')
                    sys.stdout.flush()
                    drresults = DDrppi_mocks(0, self.cosmology_flag, self.nthreads,
                                            self.pimax,
                                            self.binfilename,
                                            mu['azim_ang'][zlidx:zhidx][cidx],
                                            mu['polar_ang'][zlidx:zhidx][cidx],
                                            cz[zlidx:zhidx][cidx],
                                            RA2=rands['azim_ang'],
                                            DEC2=rands['polar_ang'],
                                            CZ2=rands['redshift'])

                    self.dr[self.jcount,:,:,k,j,i] = drresults['npairs'].reshape(-1,int(self.pimax))

                    #randoms randoms
                    print('calculating random random pairs')
                    sys.stdout.flush()
                    if (li==0) | (not self.same_rand):
                        rrresults = DDrppi_mocks(1, self.cosmology_flag, self.nthreads,
                                            self.pimax,
                                            self.binfilename,
                                            rands['azim_ang'],
                                            rands['polar_ang'],
                                            rands['redshift'])

                        try:
                            rrresults = rrresults['npairs'].reshape(-1,int(self.pimax))
                        except:
                            raise

                    self.rr[self.jcount,:,:,k,j,i] = rrresults

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
            sax.set_ylabel(r'$w_{p}(r_{p})$', labelpad=30, fontsize=20)
            sax.set_xlabel(r'$r_{p} \, [ Mpc h^{-1}]$',labelpad=30, fontsize=20)

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
                  rsd=False, upper_limit=False, splitcolor=None,
                  cinds=None, cbins=None, centrals_only=False,**kwargs):

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

        self.splitcolor = splitcolor

        if self.splitcolor is not None:
            self.ncbins = 2
            if cinds is None:
                self.cinds = [0,1]
            else:
                self.cinds = cinds

            if cbins is None:
                self.cbins = np.linspace(-0.2,1.2,60)
            else:
                self.cbins = cbins
        else:
            self.ncbins = 1

        if pimax is None:
            self.pimax = 80.0
        else:
            self.pimax = pimax

        self.rsd = rsd
        self.centrals_only = centrals_only

        self.mapkeys = ['px', 'py', 'pz', self.mkey]
        self.unitmap = {'px':'mpch', 'py':'mpch', 'pz':'mpch'}
        if self.mkey == 'luminosity':
            self.unitmap[self.mkey] = 'mag'
        else:
            self.unitmap[self.mkey] = 'msunh'

        if self.rsd:
            self.mapkeys.append('velocity')
            self.unitmap['velocity'] = 'kms'

        if self.centrals_only:
            self.mapkeys.append('central')
            self.unitmap['central'] = 'binary'

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
            self.writeCorrfuncBinFile(self.rbins)
            self.dd = np.zeros((self.njack, self.nrbins, int(self.pimax), self.ncbins, self.nmbins))
            self.dr = np.zeros((self.njack, self.nrbins, int(self.pimax), self.ncbins, self.nmbins))
            self.rr = np.zeros((self.njack, self.nrbins, int(self.pimax), self.ncbins, self.nmbins))
            self.nd = np.zeros((self.njack, self.ncbins, self.nmbins))
            self.nr = np.zeros((self.njack, self.ncbins, self.nmbins))

        if (mapunit['px'].dtype == '>f4') | (mapunit['px'].dtype == '>f8') | (mapunit['px'].dtype == np.float64):
            mu = {}
            mu['px'] = np.zeros(len(mapunit['px']), dtype=np.float32)
            mu['py'] = np.zeros(len(mapunit['py']), dtype=np.float32)
            mu['pz'] = np.zeros(len(mapunit['pz']), dtype=np.float32)

            mu['px'][:] = mapunit['px'][:]
            mu['py'][:] = mapunit['py'][:]
            mu['pz'][:] = mapunit['pz'][:]

            for f in self.mapkeys:
                if (f=='px') | (f=='py') | (f=='pz') : continue
                mu[f] = mapunit[f]

            if self.rsd:
                mu['velocity'] = np.zeros((len(mapunit['velocity']),3), dtype=np.float32)
                mu['velocity'][:] = mapunit['velocity'][:]
        else:
            mu = mapunit

        if self.rsd:
            cz = self.addRSD(mu)
        else:
            cz = mu['pz']

        if self.splitcolor is not None:
            clr = mu[self.mkey][:,self.cinds[0]] - mu[self.mkey][:,self.cinds[1]]

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

            if self.centrals_only:
                lidx &= (mapunit['central']==1)

            if (li==self.rand_ind) | (not self.same_rand):
                print('Generating Randoms')
                if len(cz[lidx])==0:
                    self.rand_ind+=1
                    continue

                rands = self.getCartesianRandoms(mu['px'][lidx], mu['py'][lidx], cz[lidx])
                if rands is None:
                    continue

            for j in range(self.ncbins):

                if self.splitcolor is not None:
                    if j==0:
                        cidx = lidx & (clr>self.splitcolor)
                    else:
                        cidx = lidx & (clr<=self.splitcolor)
                else:
                    cidx = lidx

                self.nd[self.jcount,j,i] = len(cz[cidx])
                self.nr[self.jcount,j,i] = len(rands)

                print("Number of galaxies in this z/lum/color bin: {0}".format(self.nd[self.jcount,j,i]))
                print("Number of randoms in this z/lum/color bin: {0}".format(self.nr[self.jcount,j,i]))

                if self.nd[self.jcount,j,i]<2: continue
               #data data
                print('calculating data data pairs')

                sys.stdout.flush()
                ddout = DDrppi(1,self.nthreads,self.pimax,
                               self.binfilename,
                               mu['px'][cidx],
                               mu['py'][cidx],
                               mu['pz'][cidx],
                               periodic=False)

                ddout = np.array(ddout)
                ddout = ddout.reshape(-1,int(self.pimax))
                self.dd[self.jcount,:,:,j,i] = ddout['npairs']

                print('calculating data random pairs')
                sys.stdout.flush()

                drout = DDrppi(0,self.nthreads,self.pimax,
                               self.binfilename,
                               mu['px'][cidx],
                               mu['py'][cidx],
                               mu['pz'][cidx],
                               periodic=False,
                               X2=rands['px'],
                               Y2=rands['py'],
                               Z2=rands['pz'])

                drout = np.array(drout)
                drout = drout.reshape(-1,int(self.pimax))
                self.dr[self.jcount,:,:,j,i] = drout['npairs']

                print('calculating random random pairs')
                sys.stdout.flush()

                if (li==0) | (not self.same_rand):
                    rrout = DDrppi(1,self.nthreads,self.pimax,
                                   self.binfilename,
                                   rands['px'],
                                   rands['py'],
                                   rands['pz'],
                                   periodic=False)

                    rrout = np.array(rrout)
                    rrout = rrout.reshape(-1,int(self.pimax))

                self.rr[self.jcount,:,:,j,i] = rrout['npairs']


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

                #make shapes of counts match pairs so don't need to reshape later
                ndshape.insert(1,1)
                ndshape.insert(1,1)
                ndshape.insert(1,1)
                nrshape.insert(1,1)
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
                    self.nd[jc:jc+nj,0,0,0,:] = g
                    self.nr[jc:jc+nj,0,0,0,:] = gnr[i]
                    self.dd[jc:jc+nj,:,:,:,:] = gdd[i]
                    self.dr[jc:jc+nj,:,:,:,:] = gdr[i]
                    self.rr[jc:jc+nj,:,:,:,:] = grr[i]

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
            self.nd = self.nd.reshape(self.njacktot, 1, 1, 1, self.nmbins)
            self.nr = self.nr.reshape(self.njacktot, 1, 1, 1, self.nmbins)

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


        self.wprp = self.wprp.reshape(-1, self.ncbins, self.nmbins, 1)
        self.varwprp = self.varwprp.reshape(-1, self.ncbins, self.nmbins, 1)


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
            for j in range(self.ncbins):
                ye = np.sqrt(self.varwprp[:,j,l,0])
                l1 = ax[0][i].plot(rmean, self.wprp[:,j,l,0], **kwargs)
                ax[0][i].fill_between(rmean, self.wprp[:,j,l,0]-ye, self.wprp[:,j,l,0]+ye, alpha=0.5, **kwargs)

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
            sax.set_ylabel(r'$w_{p}(r_{p})$', labelpad=40, fontsize=20)
            sax.set_xlabel(r'$r_{p} \, [ Mpc h^{-1}]$', labelpad=40, fontsize=20)

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax, l1[0]

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

class WPrpSnapshotAnalyticRandoms(CorrelationFunction):

    def __init__(self, ministry, mbins=None, rbins=None,
                  minr=None, maxr=None, logbins=True, nrbins=None,
                  pimax=None, catalog_type=None, tag=None,
                  bimodal_ccut=False, percentile_ccut=None,
                  precompute_color=False,
                  mcutind=None, same_rand=False, inv_m=True,
                  rsd=False, upper_limit=False, splitcolor=None,
                  cinds=None, cbins=None, centrals_only=False,
                  **kwargs):
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
            self.maxr = rbins[-1]
            self.nrbins = len(rbins)-1

        if pimax is None:
            self.pimax = 80.0
        else:
            self.pimax = pimax

        self.splitcolor = splitcolor
        self.bimodal_ccut = bimodal_ccut
        self.percentile_ccut = percentile_ccut

        if self.bimodal_ccut:
            self.hcbins = 100
            self.ncbins = 2
        elif self.percentile_ccut is not None:
            self.ncbins = 2
        elif self.splitcolor is not None:
            self.ncbins = 2
            if cinds is None:
                self.cinds = [0,1]
            else:
                self.cinds = cinds

            if cbins is None:
                self.cbins = np.linspace(-0.2,1.2,60)
            else:
                self.cbins = cbins
        else:
            self.ncbins = 1

        self.rsd = rsd
        self.centrals_only = centrals_only

        if self.mkey is not None:
            self.mapkeys = ['px', 'py', 'pz', self.mkey]
        else:
            self.mapkeys = ['px', 'py', 'pz']

        self.unitmap = {'px':'mpch', 'py':'mpch', 'pz':'mpch'}

        if self.mkey == 'luminosity':
            self.unitmap[self.mkey] = 'mag'

        elif self.mkey == 'halomass':
            self.unitmap[self.mkey] = 'msunh'

        if self.rsd:
            self.mapkeys.append('velocity')
            self.unitmap['velocity'] = 'kms'

        if self.centrals_only:
            self.mapkeys.append('central')
            self.unitmap['central'] = 'binary'

        self.wprp = None
        self.npairs = None

    def addRSD(self, mapunit):

        vr = mapunit['velocity'][:,2]

        return vr + mapunit['pz']


    @jackknifeMap
    def map(self, mapunit):

        if not hascorrfunc:
            raise(ImportError("CorrFunc is required to calculate wp(rp)"))

        if self.wprp is None:
            self.writeCorrfuncBinFile(self.rbins)
            self.wprp   = np.zeros((self.nrbins, self.ncbins, self.nmbins, 1))
            self.npairs = np.zeros((self.nrbins, self.ncbins, self.nmbins, 1))

        if (mapunit['px'].dtype == '>f4') | (mapunit['px'].dtype == '>f8') | (mapunit['px'].dtype == np.float64):
            mu = {}
            mu['px'] = np.zeros(len(mapunit['px']), dtype=np.float32)
            mu['py'] = np.zeros(len(mapunit['py']), dtype=np.float32)
            mu['pz'] = np.zeros(len(mapunit['pz']), dtype=np.float32)

            mu['px'][:] = mapunit['px'][:]
            mu['py'][:] = mapunit['py'][:]
            mu['pz'][:] = mapunit['pz'][:]

            for f in self.mapkeys:
                if (f=='px') | (f=='py') | (f=='pz') : continue
                mu[f] = mapunit[f]

            if self.rsd:
                mu['velocity'] = np.zeros((len(mapunit['velocity']),3), dtype=np.float32)
                mu['velocity'][:] = mapunit['velocity'][:]
        else:
            mu = mapunit

        if self.rsd:
            cz = self.addRSD(mu)
        else:
            cz = mu['pz']

        if self.splitcolor is not None:
            clr = mu[self.mkey][:,self.cinds[0]] - mu[self.mkey][:,self.cinds[1]]

        for li, i in enumerate(self.minds):
            print('Finding luminosity indices')
            if self.mcutind is not None:
                if self.upper_limit:
                    lidx = mu[self.mkey][:,self.mcutind] < self.mbins[i]
                else:
                    lidx = (self.mbins[i] <= mu[self.mkey][:,self.mcutind]) & (mu[self.mkey][:,self.mcutind] < self.mbins[i+1])
            elif (self.mcutind is None) and (self.mkey is not None):
                if self.upper_limit:
                    lidx = mu[self.mkey] < self.mbins[i]
                else:
                    lidx = (self.mbins[i] <= mu[self.mkey]) & (mu[self.mkey] < self.mbins[i+1])
            else:
                lidx = np.ones(len(mu['px']), dtype=np.bool)

            if self.centrals_only:
                lidx &= (mapunit['central']==1)

            for j in range(self.ncbins):

                if self.splitcolor is not None:
                    if j==0:
                        cidx = lidx & (clr>self.splitcolor)
                    else:
                        cidx = lidx & (clr<=self.splitcolor)
                else:
                    cidx = lidx
                    print("Number of galaxies in this z/lum bin: {0}".format(len(cz[cidx])))

                if len(cz[cidx])<2: continue
                #data data
                print('calculating wp pairs')
                sys.stdout.flush()

                results = wp(self.ministry.boxsize,
                             self.pimax,
                             self.nthreads,
                             self.binfilename,
                             mu['px'][cidx],
                             mu['py'][cidx],
                             mu['pz'][cidx])

                self.wprp[:,j,i,0] = results['wp']
                self.npairs[:,j,i,0] = results['npairs']

    def reduce(self, rank=None, comm=None):
        """
        Everything done in map.
        """
        self.varwprp = np.zeros_like(self.wprp)
        pass

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
            for j in range(self.ncbins):
                ye = np.sqrt(self.varwprp[:,j,l,0])
                l1 = ax[0][i].plot(rmean, self.wprp[:,j,l,0], **kwargs)
                ax[0][i].fill_between(rmean, self.wprp[:,j,l,0]-ye, self.wprp[:,j,l,0]+ye, alpha=0.5, **kwargs)

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
            sax.set_ylabel(r'$w_{p}(r_{p})$', labelpad=40, fontsize=20)
            sax.set_xlabel(r'$r_{p} \, [ Mpc h^{-1}]$', labelpad=40, fontsize=20)

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
                  centrals_only=False, losbins='redshift',
                  cbins=None, splitcolor=None, bimodal_ccut=False,
                  percentile_ccut=None, pccolor=False, cinds=None,
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

        if self.mkey is not None:
            self.mapkeys = ['px', 'py', 'pz', self.mkey]
        else:
            self.mapkeys = ['px','py','pz']

        self.unitmap = {'px':'mpch', 'py':'mpch', 'pz':'mpch'}

        self.centrals_only = centrals_only
        self.losbins       = losbins

        self.cbins = cbins
        self.splitcolor = splitcolor
        self.bimodal_ccut = bimodal_ccut

        if self.bimodal_ccut:
            self.hcbins = kwargs.pop('hcbins', np.linspace(-1,1,100))

        self.percentile_ccut = percentile_ccut

        if self.cbins is None:
            self.ncbins = 1
        else:
            self.ncbins = len(cbins) - 1

            if cinds is None:
                self.cinds = [0,1]
            else:
                self.cinds = cinds

        self.pccolor = pccolor

        if self.pccolor:
            self.mapkeys.append('color')

        if self.lightcone:
            if self.losbins=='redshift':
                self.mapkeys.append('redshift')
                self.unitmap['redshift'] = 'z'


        if (self.ra_lims is not None) | (self.dec_lims is not None):
            self.mapkeys.append('polar_ang')
            self.mapkeys.append('azim_ang')
            self.unitmap['polar_ang'] = 'dec'
            self.unitmap['azim_ang'] = 'ra'

        if self.mkey == 'luminosity':
            self.unitmap[self.mkey] = 'mag'
        elif self.mkey == 'halomass':
            self.unitmap[self.mkey] = 'msunh'

        if self.centrals_only:
            self.mapkeys.append('central')
            self.unitmap['central'] = 'binary'

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
            self.writeCorrfuncBinFile(self.rbins)
            self.dd = np.zeros((self.njack, self.nrbins, self.nmbins, self.ncbins, self.nzbins))
            self.dr = np.zeros((self.njack, self.nrbins, self.nmbins, self.ncbins, self.nzbins))
            self.rr = np.zeros((self.njack, self.nrbins, self.nmbins, self.ncbins, self.nzbins))
            self.nd = np.zeros((self.njack, self.nmbins, self.ncbins, self.nzbins))
            self.nr = np.zeros((self.njack, self.nmbins, self.ncbins, self.nzbins))
            if self.calc_weighted_x:
                self.rm = np.zeros((self.njack, self.nrbins, self.nmbins, self.ncbins, self.nzbins))

        if (self.ncbins > 1) & (not self.pccolor):
            clr = mapunit[self.mkey][:,self.cinds[0]] - mapunit[self.mkey][:,self.cinds[1]]

        elif self.pccolor:
            clr = mapunit['color']

        print('mapunit shape: {}'.format(mapunit['px'].shape))
        if (mapunit['px'].dtype == '>f4') | (mapunit['px'].dtype == '>f8') | (mapunit['px'].dtype == np.float64):
            newmu = True
            mu = {}
            mu['px'] = np.zeros(len(mapunit['px']), dtype=np.float32)
            mu['py'] = np.zeros(len(mapunit['py']), dtype=np.float32)
            mu['pz'] = np.zeros(len(mapunit['pz']), dtype=np.float32)

            print('mu shape: {}'.format(mu['px'].shape))

            mu['px'][:] = mapunit['px'][:]
            mu['py'][:] = mapunit['py'][:]
            mu['pz'][:] = mapunit['pz'][:]

            for f in self.mapkeys:
                if (f=='px') | (f=='py') | (f=='pz') : continue
                mu[f] = mapunit[f]
        else:
            newmu=False
            mu = mapunit

        if self.ra_lims is not None:
            aidx = (self.ra_lims[0] < mu['azim_ang']) & (mu['azim_ang'] < self.ra_lims[1])
            aidx &= (self.dec_lims[0] < mu['polar_ang']) & (mu['polar_ang'] < self.dec_lims[1])

            if not newmu:
                mu = {}
                for f in self.mapkeys:
                    mu[f] = mapunit[f][aidx]
            else:
                mu[f] = mu[f][aidx]

            del aidx


        if self.losbins=='radius':
            mu['redshift'] = np.sqrt(mu['px']**2 + mu['py']**2 + mu['pz']**2)
            zidx           = mu['redshift'].argsort()
            mu['redshift'] = mu['redshift'][zidx]

            for f in self.mapkeys:
                mu[f] = mu[f][zidx]




        if self.lightcone:
            for i in range(self.nzbins):
                print('Finding redshift indices')

                zlidx = mu['redshift'].searchsorted(self.zbins[i])
                zhidx = mu['redshift'].searchsorted(self.zbins[i+1])

                if self.ncbins>1:
                    if (self.splitcolor is None) & (self.bimodal_ccut):
                        if self.colorcat==0:
                            ccounts, cbins = np.histogram(clr[zlidx:zhidx], self.hcbins)
                        else:
                            ccounts, cbins = np.histogram(clr[zlidx1:zhidx1], self.hcbins)
                        self.splitcolor = self.splitBimodal(cbins[:-1], ccounts)

                    elif (self.splitcolor is None) & (self.percentile_ccut is not None):
                        if self.colorcat==0:
                            self.splitcolor = self.splitPercentile(clr[zlidx:zhidx], self.percentile_ccut)
                        else:
                            self.splitcolor = self.splitPercentile(clr[zlidx1:zhidx1], self.percentile_ccut)

                if zlidx==zhidx:
                    print("No galaxies in redshift bin {0} to {1}".format(self.zbins[i], self.zbins[i+1]))
                    print("Min and max z: {0}, {1}".format(np.min(mu['redshift']), np.max(mu['redshift'])))
                    print(mu['redshift'])
                    continue

                for lj, j in enumerate(self.minds):
                    if self.mkey:
                        print('Finding luminosity indices')
                        if not self.upper_limit:
                            if self.mcutind is not None:
                                lidx = (self.mbins[j] <= mu[self.mkey][zlidx:zhidx,self.mcutind]) & (mu[self.mkey][zlidx:zhidx,self.mcutind] < self.mbins[j+1])
                            else:
                                lidx = (self.mbins[j] <= mu[self.mkey][zlidx:zhidx]) & (mu[self.mkey][zlidx:zhidx] < self.mbins[j+1])
                        else:
                            if self.mcutind is not None:
                                lidx = (self.mbins[j] <= mu[self.mkey][zlidx:zhidx,self.mcutind])
                            else:
                                lidx = (self.mbins[j] <= mu[self.mkey][zlidx:zhidx])
                    else:
                        lidx = np.ones(zhidx-zlidx, dtype=np.bool)

                    if self.centrals_only:
                        lidx &= (mapunit['central']==1)


                    for k in range(self.ncbins):
                        if self.ncbins == 1:
                            cidx = lidx
                        else:
                            if self.splitcolor:
                                if k==0:
                                    cidx = lidx & (self.splitcolor < clr[zlidx:zhidx])
                                else:
                                    cidx = lidx & (self.splitcolor >= clr[zlidx:zhidx])
                            else:
                                cidx = lidx & (self.cbins[k] < clr[zlidx:zhidx]) & (clr[zlidx:zhidx] <= self.cbins[k+1])

                        if (k==self.rand_ind) | (not self.same_rand):
                            print('Generating Randoms')
                            if len(mu['pz'][zlidx:zhidx][lidx])==0:
                                self.rand_ind+=1
                                continue

                            rands = self.getCartesianRandoms(mu['px'][zlidx:zhidx][cidx],
                                                             mu['py'][zlidx:zhidx][cidx],
                                                             mu['pz'][zlidx:zhidx][cidx])
                        if rands is None:
                            continue


                        self.nd[self.jcount,j,k,i] = len(mu['pz'][zlidx:zhidx][cidx])
                        self.nr[self.jcount,j,k,i] = len(rands)

                        print("Number of galaxies in this z/lum bin: {0}".format(self.nd[self.jcount,j,k,i]))
                        print("Number of randoms in this z/lum bin: {0}".format(self.nr[self.jcount,j,k,i]))

                        if self.nd[self.jcount,j,k,i]<2: continue
                        #data data
                        print('calculating data data pairs')
                        sys.stdout.flush()

                        ddout = DD(1,self.nthreads,
                                   self.rbins,
                                   mu['px'][zlidx:zhidx][cidx],
                                   mu['py'][zlidx:zhidx][cidx],
                                   mu['pz'][zlidx:zhidx][cidx],
                                   periodic=False,
                                   output_ravg=self.calc_weighted_x)

                        ddout = np.array(ddout)

                        self.dd[self.jcount,:,j,k,i] = ddout['npairs']

                        if self.calc_weighted_x:
                            self.rm[self.jcount,:,j,k,i] = ddout['ravg']

                        print('calculating data random pairs')
                        sys.stdout.flush()

                        if 'weight' in rands.dtype.names:
                            drout = DD(0,self.nthreads,
                                       self.rbins,
                                       mu['px'][zlidx:zhidx][cidx],
                                       mu['py'][zlidx:zhidx][cidx],
                                       mu['pz'][zlidx:zhidx][cidx],
                                       weights1=np.ones(len(mu['pz'][zlidx:zhidx][cidx]),dtype=np.float32),
                                       periodic=False,
                                       X2=rands['px'],
                                       Y2=rands['py'],
                                       Z2=rands['pz'],
                                       weights2=rands['weight'])
#                                       output_ravg=self.calc_weighted_x)
                            drout = np.array(drout)
                            self.dr[self.jcount,:,j,k,i] = drout['npairs'] * drout['weightavg']

                        else:
                            drout = DD(0,self.nthreads,
                                       self.rbins,
                                       mu['px'][zlidx:zhidx][cidx],
                                       mu['py'][zlidx:zhidx][cidx],
                                       mu['pz'][zlidx:zhidx][cidx],
                                       periodic=False,
                                       X2=rands['px'],
                                       Y2=rands['py'],
                                       Z2=rands['pz'])
#                                       output_ravg=self.calc_weighted_x)

                            drout = np.array(drout)
                            self.dr[self.jcount,:,j,k,i] = drout['npairs']

                        print('calculating random random pairs')
                        sys.stdout.flush()

                        if self.rr_rand_factor<1:
                            idx = np.random.choice(np.arange(len(rands)),
                                                   size=self.rr_rand_factor*len(rands),
                                                   replace=False)
                            r = rands[idx]
                        else:
                            r = rands

                        if (lj==0) | (not self.same_rand):
                            if 'weight' in rands.dtype.names:
                                rrout = DD(1,self.nthreads,
                                           self.rbins,
                                           X1=r['px'],
                                           Y1=r['py'],
                                           Z1=r['pz'],
                                           weights1=r['weight'],
                                           periodic=False,
                                           weight_type='pair_product')
#                                           output_ravg=self.calc_weighted_x)

                                rrout = np.array(rrout)
                                rrout['npairs'] = rrout['npairs'] * rrout['weightavg']
                                print('rrout[weightavg]: {}'.format(rrout['weightavg']))
                                print('np.min(weight), np.max(weight): {}, {}'.format(np.min(r['weight']), np.max(r['weight'])))

                            else:
                                rrout = DD(1,self.nthreads,
                                           self.rbins,
                                           r['px'],
                                           r['py'],
                                           r['pz'],
                                           periodic=False)
#                                           output_ravg=self.calc_weighted_x)

                                rrout = np.array(rrout)

                        self.rr[self.jcount,:,j,k,i] = rrout['npairs']

#                        if self.calc_weighted_x:
#                            self.rm[self.jcount,:,j,k,i,2] = rrout['ravg']


        else:
            i=0
            for lj, j in enumerate(self.minds):
                print('Finding luminosity indices')
                if self.mkey:
                    if not self.upper_limit:
                        if self.mcutind is not None:
                            lidx = (self.mbins[j] <= mu[self.mkey][:,self.mcutind]) & (mu[self.mkey][:,self.mcutind] < self.mbins[j+1])
                        else:
                            lidx = (self.mbins[j] <= mu[self.mkey]) & (mu[self.mkey] < self.mbins[j+1])
                    else:
                        if self.mcutind is not None:
                            lidx = (self.mbins[j] <= mu[self.mkey][:,self.mcutind])
                        else:
                            lidx = (self.mbins[j] <= mu[self.mkey])
                else:
                    lidx = np.ones(len(mu['px']), dtype=np.bool)

                if (lj==self.rand_ind) | (not self.same_rand):
                    print('Generating Randoms')
                    if len(mu['pz'][lidx])==0:
                        self.rand_ind+=1
                        continue

                    rands = self.getCartesianRandoms(mu['px'][lidx],
                                                     mu['py'][lidx],
                                                     mu['pz'][lidx])
                    if rands is None:
                        continue

                    for k in range(self.ncbins):
                        if self.ncbins == 1:
                            cidx = lidx
                        else:
                            if self.splitcolor:
                                if k==0:
                                    cidx = lidx & (self.splitcolor < clr[:])
                                else:
                                    cidx = lidx & (self.splitcolor >= clr[:])
                            else:
                                cidx = lidx & (self.cbins[k] < clr[:]) & (clr[:] <= self.cbins[k+1])


                        self.nd[self.jcount,j,k,i] = len(mu['pz'][cidx])
                        self.nr[self.jcount,j,k,i] = len(rands)

                        print("Number of galaxies in this z/lum bin: {0}".format(self.nd[self.jcount,j,k,i]))
                        print("Number of randoms in this z/lum bin: {0}".format(self.nr[self.jcount,j,k,i]))

                        if self.nd[self.jcount,j,k,i]<2: continue
                        #data data
                        print('calculating data data pairs')
                        sys.stdout.flush()

                        ddout = DD(1,self.nthreads,
                                   self.rbins,
                                   mu['px'][cidx],
                                   mu['py'][cidx],
                                   mu['pz'][cidx],
                                   periodic=False,
                                   output_ravg=self.calc_weighted_x)

                        ddout = np.array(ddout)

                        self.dd[self.jcount,:,j,k,i] = ddout['npairs']
                        if self.calc_weighted_x:
                            self.rm[self.jcount,:,j,k,i] = ddout['ravg']

                        print('calculating data random pairs')
                        sys.stdout.flush()

                        if 'weight' in rands.dtype.names:
                            drout = DD(0,self.nthreads,
                                       self.rbins,
                                       mu['px'][cidx],
                                       mu['py'][cidx],
                                       mu['pz'][cidx],
                                       weights1=np.ones(len(mu['pz'][cidx]),dtype=np.float32),
                                       periodic=False,
                                       X2=rands['px'],
                                       Y2=rands['py'],
                                       Z2=rands['pz'],
                                       weights2=rands['weight'])
#                                       output_ravg=self.calc_weighted_x)

                            drout = np.array(drout)
                            self.dr[self.jcount,:,j,k,i] = drout['npairs'] * drout['weightavg']

                        else:
                            drout = DD(0,self.nthreads,
                                       self.rbins,
                                       mu['px'][cidx],
                                       mu['py'][cidx],
                                       mu['pz'][cidx],
                                       periodic=False,
                                       X2=rands['px'],
                                       Y2=rands['py'],
                                       Z2=rands['pz'])
 #                                      output_ravg=self.calc_weighted_x)

                            drout = np.array(drout)
                            self.dr[self.jcount,:,j,k,i] = drout['npairs']

#                        if self.calc_weighted_x:
#                            self.rm[self.jcount,:,j,k,i,1] = drout['ravg']

                        print('calculating random random pairs')
                        sys.stdout.flush()

                        if self.rr_rand_factor<1:
                            idx = np.random.choice(np.arange(len(rands)),
                                                   size=self.rr_rand_factor*len(rands),
                                                   replace=False)
                            r = rands[idx]
                        else:
                            r = rands


                        if (lj==0) | (not self.same_rand):
                            if 'weight' in rands.dtype.names:
                                rrout = DD(1,self.nthreads,
                                           self.rbins,
                                           r['px'],
                                           r['py'],
                                           r['pz'],
                                           weights1=np.ones(len(r['pz']),dtype=np.float32),
                                           periodic=False,
                                           X2=r['px'],
                                           Y2=r['py'],
                                           Z2=r['pz'],
                                           weights2=r['weight'])
#                                           output_ravg=self.calc_weighted_x)

                                rrout = np.array(rrout)
                                rrout['npairs'] *= rrout['weightavg']
                            else:
                                rrout = DD(1,self.nthreads,
                                           self.rbins,
                                           r['px'],
                                           r['py'],
                                           r['pz'],
                                           periodic=False,
                                           X2=r['px'],
                                           Y2=r['py'],
                                           Z2=r['pz'])
#                                           output_ravg=self.calc_weighted_x)

                                rrout = np.array(rrout)

                        self.rr[self.jcount,:,j,k,i] = rrout['npairs']
#                        if self.calc_weighted_x:
#                            self.rm[self.jcount,:,j,k,i,2] = rrout['ravg']



    def reduce(self, rank=None, comm=None):

        if rank is not None:
            gnd = comm.gather(self.nd, root=0)
            gnr = comm.gather(self.nr, root=0)
            gdd = comm.gather(self.dd, root=0)
            gdr = comm.gather(self.dr, root=0)
            grr = comm.gather(self.rr, root=0)
            if self.calc_weighted_x:
                grm = comm.gather(self.rm, root=0)


            if rank==0:
                ndshape = [self.nd.shape[i] for i in range(len(self.nd.shape))]
                nrshape = [self.nr.shape[i] for i in range(len(self.nr.shape))]
                ddshape = [self.dd.shape[i] for i in range(len(self.dd.shape))]
                drshape = [self.dr.shape[i] for i in range(len(self.dr.shape))]
                rrshape = [self.rr.shape[i] for i in range(len(self.rr.shape))]
                if self.calc_weighted_x:
                    rmshape = [self.rr.shape[i] for i in range(len(self.rm.shape))]

                ndshape.insert(1,1)
                nrshape.insert(1,1)

                ndshape[0] = self.njacktot
                nrshape[0] = self.njacktot
                ddshape[0] = self.njacktot
                drshape[0] = self.njacktot
                rrshape[0] = self.njacktot
                if self.calc_weighted_x:
                    rmshape[0] = self.njacktot

                self.nd = np.zeros(ndshape)
                self.nr = np.zeros(nrshape)
                self.dd = np.zeros(ddshape)
                self.dr = np.zeros(drshape)
                self.rr = np.zeros(rrshape)
                if self.calc_weighted_x:
                    self.rm = np.zeros(rmshape)

                jc = 0
                for i, g in enumerate(gnd):
                    if g is None: continue
                    nj = g.shape[0]
                    self.nd[i::len(gnd),0,:,:,:] = g
                    self.nr[i::len(gnd),0,:,:,:] = gnr[i]
                    self.dd[i::len(gnd),:,:,:,:] = gdd[i]
                    self.dr[i::len(gnd),:,:,:,:] = gdr[i]
                    self.rr[i::len(gnd),:,:,:,:] = grr[i]
                    if self.calc_weighted_x:
                        self.rm[i::len(gnd),:,:,:,:] = grm[i]

                    jc += nj

                self.jxi = np.zeros(self.dd.shape)

                self.jnd = self.jackknife(self.nd, reduce_jk=False)
                self.jnr = self.jackknife(self.nr, reduce_jk=False)
                self.jdd = self.jackknife(self.dd, reduce_jk=False)
                self.jdr = self.jackknife(self.dr, reduce_jk=False)
                self.jrr = self.jackknife(self.rr, reduce_jk=False)

                if self.calc_weighted_x:
                    self.jrm = self.jackknife(self.rm, reduce_jk=False)


                for i in range(self.njacktot):
                    for l in range(self.ncbins):
                        for j in range(self.nmbins):
                            for k in range(self.nzbins):
                                if self.jnd[i,0,j,l,k]<1: continue
                                self.jxi[i,:,j,l,k] = convert_3d_counts_to_cf(self.jnd[i,0,j,l,k],
                                                                            self.jnd[i,0,j,l,k],
                                                                            self.jnr[i,0,j,l,k],
                                                                            self.jnr[i,0,j,l,k],
                                                                            self.jdd[i,:,j,l,k],
                                                                            self.jdr[i,:,j,l,k],
                                                                            self.jdr[i,:,j,l,k],
                                                                            self.jrr[i,:,j,l,k])

                self.xi    = np.sum(self.jxi, axis=0) / self.njacktot
                self.varxi = np.sum((self.jxi - self.xi)**2, axis=0) * (self.njacktot - 1) / self.njacktot
                if self.calc_weighted_x:
                    self.rmean_weighted = np.sum(self.jrm, axis=0) / self.njacktot

        else:
            self.jxi = np.zeros(self.dd.shape)
            self.nd  = self.nd.reshape(-1,1,self.nmbins,self.ncbins,self.nzbins)
            self.nr  = self.nr.reshape(-1,1,self.nmbins,self.ncbins,self.nzbins)

            self.jnd = self.jackknife(self.nd, reduce_jk=False)
            self.jnr = self.jackknife(self.nr, reduce_jk=False)
            self.jdd = self.jackknife(self.dd, reduce_jk=False)
            self.jdr = self.jackknife(self.dr, reduce_jk=False)
            self.jrr = self.jackknife(self.rr, reduce_jk=False)
            if self.calc_weighted_x:
                self.jrm = self.jackknife(self.rm, reduce_jk=False)

            for i in range(self.njacktot):
                for l in range(self.ncbins):
                    for j in range(self.nmbins):
                        for k in range(self.nzbins):
                            if self.jnd[i,0,j,l,k]<1: continue
                            self.jxi[i,:,j,l,k] = convert_3d_counts_to_cf(self.jnd[i,0,j,l,k],
                                                                          self.jnd[i,0,j,l,k],
                                                                          self.jnr[i,0,j,l,k],
                                                                          self.jnr[i,0,j,l,k],
                                                                          self.jdd[i,:,j,l,k],
                                                                          self.jdr[i,:,j,l,k],
                                                                          self.jdr[i,:,j,l,k],
                                                                          self.jrr[i,:,j,l,k])

            self.xi    = np.sum(self.jxi, axis=0) / self.njacktot
            self.varxi = np.sum((self.jxi - self.xi)**2, axis=0) * (self.njacktot - 1) / self.njacktot

            if self.calc_weighted_x:
                self.rmean_weighted = np.sum(self.jrm, axis=0) / self.njacktot

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
                ye = np.sqrt(self.varxi[:,l,0,z])
                l1 = ax[j][i].plot(rmean, self.xi[:,l,0,z], **kwargs)
                ax[j][i].fill_between(rmean, self.xi[:,l,0,z]-ye, self.xi[:,0,l,z]+ye, alpha=0.5, **kwargs)

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


class XiofRAnalyticRandoms(CorrelationFunction):

    def __init__(self, ministry, mbins=None, zbins=None, rbins=None,
                  minr=None, maxr=None, logbins=True, nrbins=None,
                  lightcone=False, catalog_type=None, tag=None,
                  mcutind=None, same_rand=False, inv_m=True,
                  centrals_only=False, **kwargs):

        """
        Real space 3-d correlation function, xi(r), for use with periodic
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

        if self.mkey is not None:
            self.mapkeys = ['px', 'py', 'pz', self.mkey]
        else:
            self.mapkeys = ['px','py','pz']

        self.centrals_only = centrals_only

        self.unitmap = {'px':'mpch', 'py':'mpch', 'pz':'mpch'}

        if self.centrals_only:
            self.mapkeys.append('central')
            self.unitmap['central'] = 'binary'

        if self.mkey == 'luminosity':
            self.unitmap[self.mkey] = 'mag'
        elif self.mkey == 'halomass':
            self.unitmap[self.mkey] = 'msunh'

        self.xi = None

    @jackknifeMap
    def map(self, mapunit):

        if not hascorrfunc:
            raise(ImportError("CorrFunc is required to calculate xi(r)"))

        if self.xi is None:
            self.writeCorrfuncBinFile(self.rbins)
            self.xi = np.zeros((self.nrbins, self.nmbins, 1))

        mu = {}
        mu['px'] = np.zeros(len(mapunit['px']), dtype=np.float64)
        mu['py'] = np.zeros(len(mapunit['py']), dtype=np.float64)
        mu['pz'] = np.zeros(len(mapunit['pz']), dtype=np.float64)

        mu['px'][:] = mapunit['px'][:]
        mu['py'][:] = mapunit['py'][:]
        mu['pz'][:] = mapunit['pz'][:]

        for f in self.mapkeys:
            if (f=='px') | (f=='py') | (f=='pz'): continue
            mu[f] = mapunit[f]

        for lj, j in enumerate(self.minds):
            print('Finding luminosity indices')

            if not self.upper_limit:
                if self.mcutind is not None:
                    lidx = (self.mbins[j] <= mu[self.mkey][:,self.mcutind]) & (mu[self.mkey][:,self.mcutind] < self.mbins[j+1])
                elif (self.mcutind is None) and (self.mkey is not None):
                    lidx = (self.mbins[j] <= mu[self.mkey]) & (mu[self.mkey] < self.mbins[j+1])
                else:
                    lidx = np.ones(len(mu['px']), dtype=np.bool)
            else:
                if self.mcutind is not None:
                    lidx = (mu[self.mkey][:,self.mcutind] < self.mbins[j])
                elif (self.mcutind is None) and (self.mkey is not None):
                    lidx = (mu[self.mkey] < self.mbins[j])
                else:
                    lidx = np.ones(len(mu['px']), dtype=np.bool)

            if self.centrals_only:
                lidx &= (mu['central']==1)

            print("Number of galaxies in this z/lum bin: {0}".format(len(mu['pz'][lidx])))
            print('calculating xi(r)')

            sys.stdout.flush()
            ddout = xi(self.ministry.boxsize, self.nthreads,
                       self.rbins,
                       mu['px'][lidx],
                       mu['py'][lidx],
                       mu['pz'][lidx])

            ddout = np.array(ddout)

            self.xi[:,j,0] = ddout['xi']


    def reduce(self, rank=None, comm=None):
        self.varxi = np.zeros_like(self.xi)


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
                ye = np.sqrt(self.varxi[:,l,z])
                l1 = ax[j][i].plot(rmean, self.xi[:,l,z], **kwargs)
                ax[j][i].fill_between(rmean, self.xi[:,l,z]-ye, self.xi[:,l,z]+ye, alpha=0.5, **kwargs)

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

class XixyAnalyticRandoms(CrossCorrelationFunction):

    def __init__(self, ministry, mbins=None, zbins=None, rbins=None,
                  minr=None, maxr=None, logbins=True, nrbins=None,
                  catalog_type=None, tag=None,
                  mcutind=None, same_rand=False, inv_m=True,
                  centrals_only=False, centrals_only1=False,
                  rsd=False, rsd1=False, pccolor=False,
                  cbins=None, **kwargs):

        """
        Real space 3-d cross correlation function, xi_xy(r), for use with periodic
        data.
        """
        CrossCorrelationFunction.__init__(self, ministry,
                                      mbins=mbins, nrbins=nrbins,
                                      mcutind=mcutind, zbins=zbins,
                                      same_rand=same_rand, inv_m=inv_m,
                                      catalog_type=catalog_type, tag=tag,
                                      lightcone=False, **kwargs)

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

        self.mapkeys = ['px', 'py', 'pz',
                        'px1', 'py1', 'pz1']

        if self.mkey is not None:
            self.mapkeys.append(self.mkey)

        if self.mkey1 is not None:
            self.mapkeys.append(self.mkey1)

        self.unitmap = {'px':'mpch', 'py':'mpch', 'pz':'mpch',
                        'px1':'mpch', 'py1':'mpch', 'pz1':'mpch'}

        if (self.mkey == 'appmag') | (self.mkey=='luminosity'):
            self.unitmap[self.mkey] = 'mag'
        if (self.mkey == 'halomass'):
            self.unitmap[self.mkey] = 'msunh'

        if (self.mkey1 == 'appmag') | (self.mkey1=='luminosity'):
            self.unitmap[self.mkey1] = 'mag'
        if (self.mkey1 == 'halomass'):
            self.unitmap[self.mkey1] = 'msunh'

        self.rsd  = rsd
        self.rsd1 = rsd1

        self.centrals_only  = centrals_only
        self.centrals_only1 = centrals_only1

        self.cbins = cbins

        if self.cbins is None:
            self.ncbins = 1
        else:
            self.ncbins = len(cbins) - 1
            if self.colorcat is None:
                if ('galaxy' in self.catalog_type[0]) & ('galaxy' not in self.catalog_type[1]):
                    self.colorcat = 0
                elif ('galaxy' not in self.catalog_type[0]) & ('galaxy' in self.catalog_type[1]):
                    self.colorcat = 1
                else:
                    #if both galaxy catalogs, then use first
                    self.colorcat = 0

        self.pccolor = pccolor

        if self.rsd:
            self.mapkeys.append('velocity')
            self.unitmap['velocity'] = 'kms'

        if self.rsd1:
            self.mapkeys.append('velocity1')
            self.unitmap['velocity1'] = 'kms'

        if self.centrals_only:
            self.mapkeys.append('central')
            self.unitmap['central'] = 'binary'

        if self.centrals_only1:
            self.mapkeys.append('central1')
            self.unitmap['central1'] = 'binary'

        if self.pccolor:
            self.mapkeys.append('color')

        self.dd = None
        self.nd1 = None
        self.nd2 = None

    @jackknifeMap
    def map(self, mapunit):

        if (self.ncbins > 1) & (~self.pccolor):
            if self.colorcat==0:
                clr = mapunit[self.mkey][:,self.cinds[0]] - mapunit[self.mkey][:,self.cinds[1]]
            else:
                clr = mapunit[self.mkey1][:,self.cinds[0]] - mapunit[self.mkey1][:,self.cinds[1]]
        elif self.pccolor:
            clr = mapunit['color']

        if not hascorrfunc:
            raise(ImportError("CorrFunc is required to calculate xi(r)"))

        if self.dd is None:
            self.writeCorrfuncBinFile(self.rbins)
            self.dd = np.zeros((self.nrbins, self.ncbins, self.nmbins, self.nmbins1))
            self.RR = np.zeros((self.nrbins, self.ncbins, self.nmbins, self.nmbins1))
            self.nd1 = np.zeros((self.ncbins, self.nmbins, self.nmbins1))
            self.nd2 = np.zeros((self.ncbins, self.nmbins, self.nmbins1))

        mu = {}
        mu['px'] = np.zeros(len(mapunit['px']), dtype=np.float64)
        mu['py'] = np.zeros(len(mapunit['py']), dtype=np.float64)
        mu['pz'] = np.zeros(len(mapunit['pz']), dtype=np.float64)

        mu['px1'] = np.zeros(len(mapunit['px1']), dtype=np.float64)
        mu['py1'] = np.zeros(len(mapunit['py1']), dtype=np.float64)
        mu['pz1'] = np.zeros(len(mapunit['pz1']), dtype=np.float64)

        mu['px'][:] = mapunit['px'][:].astype(np.float64)
        mu['py'][:] = mapunit['py'][:].astype(np.float64)
        mu['pz'][:] = mapunit['pz'][:].astype(np.float64)

        mu['px1'][:] = mapunit['px1'][:].astype(np.float64)
        mu['py1'][:] = mapunit['py1'][:].astype(np.float64)
        mu['pz1'][:] = mapunit['pz1'][:].astype(np.float64)

        for f in self.mapkeys:
            if ('px' in f ) | ('py' in f) | ('pz' in f): continue

            mu[f] = mapunit[f]

        if self.ncbins>1:
            if (self.splitcolor is None) & (self.bimodal_ccut):
                if self.colorcat==0:
                    ccounts, cbins = np.histogram(clr[:], self.hcbins)
                else:
                    ccounts, cbins = np.histogram(clr[:], self.hcbins)
                self.splitcolor = self.splitBimodal(cbins[:-1], ccounts)

            elif (self.splitcolor is None) & (self.percentile_ccut is not None):
                if self.colorcat==0:
                    self.splitcolor = self.splitPercentile(clr[:], self.percentile_ccut)
                else:
                    self.splitcolor = self.splitPercentile(clr[:], self.percentile_ccut)

        for li, j in enumerate(self.minds):
            if (self.mkey is not None) & (self.mbins is not None):
                if self.mcutind is not None:
                    if self.upper_limit:
                        lidx = mu[self.mkey][:,self.mcutind] < self.mbins[j]
                    else:
                        lidx = (self.mbins[j] <= mu[self.mkey][:,self.mcutind]) & (mu[self.mkey][:,self.mcutind] < self.mbins[j+1])
                else:
                    if self.upper_limit:
                        lidx = mu[self.mkey][:] < self.mbins[j]
                    else:
                        lidx = (self.mbins[j] <= mu[self.mkey][:]) & (mu[self.mkey][:] < self.mbins[j+1])
            else:
                lidx = np.ones(len(mu['px']), dtype=np.bool)

            if self.centrals_only:
                lidx = lidx & (mu['central'][:]==1)

            for li1, j1 in enumerate(self.minds1):
                if (self.mkey1 is not None) & (self.mbins1 is not None) :
                    if self.mcutind1 is not None:
                        if self.upper_limit1:
                            lidx1 = mu[self.mkey1][:,self.mcutind1] < self.mbins1[j1]
                        else:
                            lidx1 = (self.mbins1[j1] <= mu[self.mkey1][:,self.mcutind1]) & (mu[self.mkey1][:,self.mcutind1] < self.mbins1[j1+1])
                    else:
                        if self.upper_limit1:
                            lidx1 = mu[self.mkey1][:] < self.mbins1[j1]
                        else:
                            lidx1 = (self.mbins1[j1] <= mu[self.mkey1][:]) & (mu[self.mkey1][:] < self.mbins1[j1+1])
                else:
                    lidx1 = np.ones(len(mu['px1']), dtype=np.bool)

                if self.centrals_only1:
                    lidx1 = lidx1 & (mu['central1'][:]==1)

                for k in range(self.ncbins):
                    if self.ncbins == 1:
                        cidx = lidx
                    else:
                        if self.splitcolor:
                            if k==0:
                                cidx = lidx & (self.splitcolor < clr[:])
                            else:
                                cidx = lidx & (self.splitcolor >= clr[:])
                        else:
                            cidx = lidx & (self.cbins[k] < clr[:]) & (clr[:] <= self.cbins[k+1])

                    self.nd1[k,j,j1] = np.sum(cidx)
                    self.nd2[k,j,j1] = np.sum(lidx1)

                    print("Number of cat1 in this lum bin: {0}".format(self.nd1[k,j,j1]))
                    print("Number of cat2 in this lum bin: {0}".format(self.nd2[k,j,j1]))

                    #data data
                    print('calculating data data pairs')
                    sys.stdout.flush()
                    if (self.nd1[k,j,j1]<2) | (self.nd2[k,j,j1]<2):
                        continue

                    ddout =     DD(0,self.nthreads,
                                   self.rbins,
                                   mu['px'][cidx],
                                   mu['py'][cidx],
                                   mu['pz'][cidx],
                                   X2=mu['px1'][lidx1],
                                   Y2=mu['py1'][lidx1],
                                   Z2=mu['pz1'][lidx1],
                                   periodic=True,
                                   boxsize=self.ministry.boxsize)

                    ddout = np.array(ddout)

                    self.dd[:,k,j,j1] = ddout['npairs']
                    self.RR[:,k,j,j1] = 4 * np.pi / 3 * self.nd1[k,j,j1] * self.nd2[k,j,j1] * np.array([ddout['rmax'][i]**3 - ddout['rmin'][i]**3 for i in range(len(ddout))]) / self.ministry.boxsize ** 3

    def reduce(self, rank=None, comm=None):
        self.xi    = ((self.dd - self.RR) / self.RR).reshape(self.nrbins,self.ncbins,self.nmbins,self.nmbins1,1)

        self.varxi = np.zeros_like(self.xi)


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
                ye = np.sqrt(self.varxi[:,l,z])
                l1 = ax[j][i].plot(rmean, self.xi[:,l,z], **kwargs)
                ax[j][i].fill_between(rmean, self.xi[:,l,z]-ye, self.xi[:,l,z]+ye, alpha=0.5, **kwargs)

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


class XiXY(CrossCorrelationFunction):

    def __init__(self, ministry, mbins=None, zbins=None, rbins=None,
                  minr=None, maxr=None, logbins=True, nrbins=None,
                  catalog_type=None, tag=None,
                  mcutind=None, same_rand=False,
                  same_rand1=False, inv_m=True,
                  centrals_only=False, centrals_only1=False,
                  rsd=False, rsd1=False, pccolor=False,
                  cbins=None, losbins='redshift', cosmology_flag=None,
                  colorcat=None, splitcolor=None, bimodal_ccut=False,
                  percentile_ccut=None, **kwargs):

        """
        Real space 3-d cross correlation function, xi_xy(r), for use with periodic
        data.
        """
        CrossCorrelationFunction.__init__(self, ministry,
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

        self.mapkeys = ['px', 'py', 'pz',
                        'px1', 'py1', 'pz1']


        if self.mkey is not None:
            self.mapkeys.append(self.mkey)

        if self.mkey1 is not None:
            self.mapkeys.append(self.mkey1)

        self.unitmap = {'px':'mpch', 'py':'mpch', 'pz':'mpch',
                        'px1':'mpch', 'py1':'mpch', 'pz1':'mpch'}

        self.losbins = losbins

        if losbins=='redshift':
            self.mapkeys.extend(['redshift', 'redshift1'])
            self.unitmap['redshift'] = 'z'
            self.unitmap['redshift1'] = 'z'

        if (self.mkey == 'appmag') | (self.mkey=='luminosity'):
            self.unitmap[self.mkey] = 'mag'
        if (self.mkey == 'halomass'):
            self.unitmap[self.mkey] = 'msunh'

        if (self.mkey1 == 'appmag') | (self.mkey1=='luminosity'):
            self.unitmap[self.mkey1] = 'mag'
        if (self.mkey1 == 'halomass'):
            self.unitmap[self.mkey1] = 'msunh'

        self.rsd  = rsd
        self.rsd1 = rsd1

        self.centrals_only  = centrals_only
        self.centrals_only1 = centrals_only1

        self.same_rand1 = same_rand1

        self.cbins = cbins
        self.colorcat = colorcat
        self.splitcolor = splitcolor
        self.bimodal_ccut = bimodal_ccut
        self.percentile_ccut = percentile_ccut

        if self.cbins is None:
            self.ncbins = 1
        else:
            self.ncbins = len(cbins) - 1
            if self.colorcat is None:
                if ('galaxy' in self.catalog_type[0]) & ('galaxy' not in self.catalog_type[1]):
                    self.colorcat = 0
                elif ('galaxy' not in self.catalog_type[0]) & ('galaxy' in self.catalog_type[1]):
                    self.colorcat = 1
                else:
                    #if both galaxy catalogs, then use first
                    self.colorcat = 0

        self.pccolor = pccolor

        if self.rsd:
            self.mapkeys.append('velocity')
            self.unitmap['velocity'] = 'kms'

        if self.rsd1:
            self.mapkeys.append('velocity1')
            self.unitmap['velocity1'] = 'kms'

        if self.centrals_only:
            self.mapkeys.append('central')
            self.unitmap['central'] = 'binary'

        if self.centrals_only1:
            self.mapkeys.append('central1')
            self.unitmap['central1'] = 'binary'

        if self.pccolor:
            self.mapkeys.append('color')

        if cosmology_flag is None:
            self.cosmology_flag = 2
        else:
            self.cosmology_flag = cosmology_flag

        self.jcount = 0
        self.rand_ind = 0
        self.rand_ind1 = 0

        self.nd1 = None
        self.nr1 = None
        self.nd2 = None
        self.nr2 = None

        self.dd = None
        self.dr = None
        self.rd = None
        self.rr = None

    def getWeights(self, z0, z1):

        zmin = np.min(z1)
        zmax = np.max(z1)
        zbins = np.linspace(zmin, zmax, 100)

        nz0, _ = np.histogram(z0, zbins, normed=True)
        nz1, _ = np.histogram(z1, zbins, normed=True)

        bin_weights = nz1 / nz0
        bin_weights[~np.isfinite(bin_weights)] = 1.0

        idx = np.digitize(z1, zbins)

        weights = np.ones_like(z1)
        weights[(idx>0)&(idx<len(zbins))] = bin_weights[idx[(idx>0)&(idx<len(zbins))]-1]

        return weights



    @jackknifeMap
    def map(self, mapunit):
        if (self.ncbins > 1) & (~self.pccolor):
            if self.colorcat==0:
                clr = mapunit[self.mkey][:,self.cinds[0]] - mapunit[self.mkey][:,self.cinds[1]]
            else:
                clr = mapunit[self.mkey1][:,self.cinds[0]] - mapunit[self.mkey1][:,self.cinds[1]]
        elif self.pccolor:
            clr = mapunit['color']

        if self.nd1 is None:
            self.writeCorrfuncBinFile(self.rbins, binfilename='angular_bins')
            self.nd1 = np.zeros((self.njack, self.ncbins, self.nmbins,
                                  self.nmbins1, self.nzbins * self.nzbins1))
            self.nr1 = np.zeros((self.njack, self.ncbins, self.nmbins,
                                  self.nmbins1, self.nzbins * self.nzbins1))
            self.nd2 = np.zeros((self.njack, self.ncbins, self.nmbins,
                                  self.nmbins1, self.nzbins * self.nzbins1))
            self.nr2 = np.zeros((self.njack, self.ncbins, self.nmbins,
                                  self.nmbins1, self.nzbins * self.nzbins1))

            self.dd = np.zeros((self.njack, self.nrbins, self.ncbins,
                                  self.nmbins, self.nmbins1, self.nzbins * self.nzbins1))
            self.dr = np.zeros((self.njack, self.nrbins, self.ncbins,
                                  self.nmbins, self.nmbins1, self.nzbins * self.nzbins1))
            self.rd = np.zeros((self.njack, self.nrbins, self.ncbins,
                                  self.nmbins, self.nmbins1, self.nzbins * self.nzbins1))
            self.rr = np.zeros((self.njack, self.nrbins, self.ncbins,
                                  self.nmbins, self.nmbins1, self.nzbins * self.nzbins1))

        mu = {}
        mu['px'] = np.zeros(len(mapunit['px']), dtype=np.float32)
        mu['py'] = np.zeros(len(mapunit['py']), dtype=np.float32)
        mu['pz'] = np.zeros(len(mapunit['pz']), dtype=np.float32)

        mu['px1'] = np.zeros(len(mapunit['px1']), dtype=np.float32)
        mu['py1'] = np.zeros(len(mapunit['py1']), dtype=np.float32)
        mu['pz1'] = np.zeros(len(mapunit['pz1']), dtype=np.float32)

        mu['px'][:] = mapunit['px'][:].astype(np.float32)
        mu['py'][:] = mapunit['py'][:].astype(np.float32)
        mu['pz'][:] = mapunit['pz'][:].astype(np.float32)

        mu['px1'][:] = mapunit['px1'][:].astype(np.float32)
        mu['py1'][:] = mapunit['py1'][:].astype(np.float32)
        mu['pz1'][:] = mapunit['pz1'][:].astype(np.float32)


        for f in self.mapkeys:
            if ('px' in f) | ('py' in f) | ('pz' in f) : continue
            mu[f] = mapunit[f]

        if self.losbins=='radius':
            mu['redshift'] = np.sqrt(mu['px']**2 + mu['py']**2 + mu['pz']**2)
            mu['redshift1'] = np.sqrt(mu['px1']**2 + mu['py1']**2 + mu['pz1']**2)

            zidx           = mu['redshift'].argsort()
            mu['redshift'] = mu['redshift'][zidx]

            for f in self.mapkeys:
                if len(mu[f])==len(mu['redshift']):
                    mu[f] = mu[f][zidx]

            zidx           = mu['redshift1'].argsort()
            mu['redshift1'] = mu['redshift1'][zidx]

            for f in self.mapkeys:
                if len(mu[f])==len(mu['redshift1']):
                    mu[f] = mu[f][zidx]

        if self.rsd:
            z = self.addRSD(mu) / self.c
        else:
            z = mu['redshift']

        if self.rsd1:
            z1 = self.addRSD(mu, cat1=True) / self.c
        else:
            z1 = mu['redshift1']

        for i in range(self.nzbins):
            print('Finding redshift indices')

            zlidx = z.searchsorted(self.zbins[i])
            zhidx = z.searchsorted(self.zbins[i+1])

            for i1 in range(self.nzbins1):
                if self.same_zbins:
                    zlidx1 = z1.searchsorted(self.zbins[i])
                    zhidx1 = z1.searchsorted(self.zbins[i+1])
                else:
                    zlidx1 = z1.searchsorted(self.zbins1[i1])
                    zhidx1 = z1.searchsorted(self.zbins1[i1+1])

                if (zlidx==zhidx):
                    print("No galaxies in redshift bin {0} to {1}".format(self.zbins[i], self.zbins[i+1]))
                    print('z: {}'.format(z))

                    print("Min and max z: {0}, {1}".format(np.min(z), np.max(z)))
                    continue
                elif (zlidx1==zhidx1):
                    print("No galaxies in redshift bin {0} to {1}".format(self.zbins1[i], self.zbins1[i+1]))
                    print('z1: {}'.format(z1))

                    print("Min and max z1: {0}, {1}".format(np.min(z1), np.max(z1)))
                    continue

                if self.ncbins>1:
                    if (self.splitcolor is None) & (self.bimodal_ccut):
                        if self.colorcat==0:
                            ccounts, cbins = np.histogram(clr[zlidx:zhidx], self.hcbins)
                        else:
                            ccounts, cbins = np.histogram(clr[zlidx1:zhidx1], self.hcbins)
                        self.splitcolor = self.splitBimodal(cbins[:-1], ccounts)

                    elif (self.splitcolor is None) & (self.percentile_ccut is not None):
                        if self.colorcat==0:
                            self.splitcolor = self.splitPercentile(clr[zlidx:zhidx], self.percentile_ccut)
                        else:
                            self.splitcolor = self.splitPercentile(clr[zlidx1:zhidx1], self.percentile_ccut)

                for li, j in enumerate(self.minds):
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

                        rands = self.getCartesianRandoms(mu['px'][zlidx:zhidx][lidx],
                                                         mu['py'][zlidx:zhidx][lidx],
                                                         mu['pz'][zlidx:zhidx][lidx])

                    for li1, j1 in enumerate(self.minds1):
                        if self.mbins1 is None:
                            lidx1 = np.ones(zhidx1-zlidx1, dtype=np.bool)
                        else:
                            if self.mcutind1 is not None:
                                if self.upper_limit1:
                                    lidx1 = mu[self.mkey1][zlidx1:zhidx1,self.mcutind1] < self.mbins1[j1]
                                else:
                                    lidx1 = (self.mbins1[j1] <= mu[self.mkey1][zlidx1:zhidx1,self.mcutind1]) & (mu[self.mkey1][zlidx1:zhidx1,self.mcutind1] < self.mbins1[j1+1])
                            else:
                                if self.upper_limit1:
                                    lidx1 = mu[self.mkey1][zlidx1:zhidx1] < self.mbins1[j1]
                                else:
                                    lidx1 = (self.mbins1[j1] <= mu[self.mkey1][zlidx1:zhidx1]) & (mu[self.mkey1][zlidx1:zhidx1] < self.mbins1[j1+1])

                        if self.centrals_only1:
                            lidx1 = lidx1 & (mu['central1'][zlidx1:zhidx1]==1)

                        if (li1==self.rand_ind1) | (not self.same_rand1):
                            print('Generating Randoms')
                            if len(z1[zlidx1:zhidx1][lidx1])==0:
                                self.rand_ind1+=1
                                continue

                            rands1 = self.getCartesianRandoms(mu['px1'][zlidx1:zhidx1][lidx1],
                                                              mu['py1'][zlidx1:zhidx1][lidx1],
                                                              mu['pz1'][zlidx1:zhidx1][lidx1])

                        for k in range(self.ncbins):
                            if self.ncbins == 1:
                                cidx = lidx
                            else:
                                if self.splitcolor:
                                    if k==0:
                                        cidx = lidx & (self.splitcolor < clr[zlidx:zhidx])
                                    else:
                                        cidx = lidx & (self.splitcolor >= clr[zlidx:zhidx])
                                else:
                                    cidx = lidx & (self.cbins[k] < clr[zlidx:zhidx]) & (clr[zlidx:zhidx] <= self.cbins[k+1])

                            print('i*self.nzbins1 + i1: {}'.format(i*self.nzbins1 + i1))

                            self.nd1[self.jcount,k,j,j1,i*self.nzbins1 + i1] = len(mu['px'][zlidx:zhidx][cidx])
                            self.nd2[self.jcount,k,j,j1,i*self.nzbins1 + i1] = len(mu['px1'][zlidx1:zhidx1][lidx1])

                            self.nr1[self.jcount,k,j,j1,i*self.nzbins1 + i1] = len(rands)
                            self.nr2[self.jcount,k,j,j1,i*self.nzbins1 + i1] = len(rands1)

                            print("Number of cat1 in this z/lum bin: {0}".format(self.nd1[self.jcount,k,j,j1,i*self.nzbins1 + i1]))
                            print("Number of cat2 in this z/lum bin: {0}".format(self.nd2[self.jcount,k,j,j1,i*self.nzbins1 + i1]))

                            print("Number of rands1 in this z/lum bin: {0}".format(self.nr1[self.jcount,k,j,j1,i*self.nzbins1 + i1]))
                            print("Number of rands2 in this z/lum bin: {0}".format(self.nr2[self.jcount,k,j,j1,i*self.nzbins1 + i1]))

                            #data data
                            print('calculating data data pairs')
                            sys.stdout.flush()
                            if (self.nd1[self.jcount,k,j,j1,i*self.nzbins1 +i1]<2) | (self.nd2[self.jcount,k,j,j1,i*self.nzbins1 + i1]<2):
                                continue

                            weights = self.getWeights(mu['redshift'][zlidx:zhidx][lidx], mu['redshift1'][zlidx1:zhidx1][lidx1])
                            rweights = self.getWeights(mu['redshift'][zlidx:zhidx][lidx], rands1['redshift'])

                            ddout = DD(0,self.nthreads,
                                       self.rbins,
                                       mu['px'][zlidx:zhidx][lidx],
                                       mu['py'][zlidx:zhidx][lidx],
                                       mu['pz'][zlidx:zhidx][lidx],
                                       weights1=np.ones(len(mu['pz'][zlidx:zhidx][lidx]),dtype=np.float32),
                                       periodic=False,
                                       X2=mu['px1'][zlidx1:zhidx1][lidx1],
                                       Y2=mu['py1'][zlidx1:zhidx1][lidx1],
                                       Z2=mu['pz1'][zlidx1:zhidx1][lidx1],
                                       weights2=weights.astype(np.float32),
                                       weight_type='pair_product')
                            ddout = np.array(ddout)
                            self.dd[self.jcount,:,k,j,j1,i*self.nzbins1 + i1] = ddout['npairs'] * ddout['weightavg']

                            #data randoms
                            print('calculating data random pairs')
                            sys.stdout.flush()
                            drout = DD(0,self.nthreads,
                                       self.rbins,
                                       mu['px'][zlidx:zhidx][lidx],
                                       mu['py'][zlidx:zhidx][lidx],
                                       mu['pz'][zlidx:zhidx][lidx],
                                       weights1=np.ones(len(mu['pz'][zlidx:zhidx][lidx]),dtype=np.float32),
                                       periodic=False,
                                       X2=rands1['px'],
                                       Y2=rands1['py'],
                                       Z2=rands1['pz'],
                                       weights2=rweights.astype(np.float32),
                                       weight_type='pair_product')
                            drout = np.array(drout)
                            self.dr[self.jcount,:,k,j,j1,i*self.nzbins1 + i1] = drout['npairs'] * drout['weightavg']


                            print('calculating data random pairs')
                            sys.stdout.flush()
                            drout = DD(0,self.nthreads,
                                       self.rbins,
                                       mu['px1'][zlidx1:zhidx1][lidx1],
                                       mu['py1'][zlidx1:zhidx1][lidx1],
                                       mu['pz1'][zlidx1:zhidx1][lidx1],
                                       weights1=weights.astype(np.float32),
                                       periodic=False,
                                       X2=rands['px'],
                                       Y2=rands['py'],
                                       Z2=rands['pz'],
                                       weights2=np.ones(len(rands['px']),dtype=np.float32),
                                       weight_type='pair_product')
                            drout = np.array(drout)
                            self.rd[self.jcount,:,k,j,j1,i*self.nzbins1 + i1] = drout['npairs'] * drout['weightavg']

                            #randoms randoms
                            print('calculating random random pairs')
                            sys.stdout.flush()
                            if (li==self.rand_ind) & (li1==self.rand_ind1) | (not self.same_rand):
                                rrout = DD(0,self.nthreads,
                                           self.rbins,
                                           rands['px'],
                                           rands['py'],
                                           rands['pz'],
                                           weights1=np.ones(len(rands['px']),dtype=np.float32),
                                           periodic=False,
                                           X2=rands1['px'],
                                           Y2=rands1['py'],
                                           Z2=rands1['pz'],
                                           weights2=rweights.astype(np.float32),
                                           weight_type='pair_product')
                                rrout = np.array(rrout)

                            self.rr[self.jcount,:,k,j,j1,i*self.nzbins1 + i1] = rrout['npairs'] * rrout['weightavg']


    def reduce(self, rank=None, comm=None):

        if rank is not None:
            gnd1 = comm.gather(self.nd1, root=0)
            gnr1 = comm.gather(self.nr1, root=0)
            gnd2 = comm.gather(self.nd2, root=0)
            gnr2 = comm.gather(self.nr2, root=0)

            gdd = comm.gather(self.dd, root=0)
            gdr = comm.gather(self.dr, root=0)
            grd = comm.gather(self.rd, root=0)
            grr = comm.gather(self.rr, root=0)


            if rank==0:
                nd1shape = [self.nd1.shape[i] for i in range(len(self.nd1.shape))]
                nr1shape = [self.nr1.shape[i] for i in range(len(self.nr1.shape))]
                nd2shape = [self.nd2.shape[i] for i in range(len(self.nd2.shape))]
                nr2shape = [self.nr2.shape[i] for i in range(len(self.nr2.shape))]

                ddshape = [self.dd.shape[i] for i in range(len(self.dd.shape))]
                drshape = [self.dr.shape[i] for i in range(len(self.dr.shape))]
                rdshape = [self.rd.shape[i] for i in range(len(self.rd.shape))]
                rrshape = [self.rr.shape[i] for i in range(len(self.rr.shape))]

                nd1shape.insert(1,1)
                nr1shape.insert(1,1)
                nd2shape.insert(1,1)
                nr2shape.insert(1,1)

                nd1shape[0] = self.njacktot
                nr1shape[0] = self.njacktot
                nd2shape[0] = self.njacktot
                nr2shape[0] = self.njacktot

                ddshape[0] = self.njacktot
                drshape[0] = self.njacktot
                rdshape[0] = self.njacktot
                rrshape[0] = self.njacktot

                self.nd1 = np.zeros(nd1shape)
                self.nr1 = np.zeros(nr1shape)
                self.nd2 = np.zeros(nd2shape)
                self.nr2 = np.zeros(nr2shape)

                self.dd = np.zeros(ddshape)
                self.dr = np.zeros(drshape)
                self.rd = np.zeros(rdshape)
                self.rr = np.zeros(rrshape)

                jc = 0
                for i, g in enumerate(gnd1):
                    if g is None: continue
                    nj = g.shape[0]
                    self.nd1[jc:jc+nj,0,:,:,:,:] = g
                    self.nr1[jc:jc+nj,0,:,:,:,:] = gnr1[i]
                    self.nd2[jc:jc+nj,0,:,:,:,:] = gnd2[i]
                    self.nr2[jc:jc+nj,0,:,:,:,:] = gnr2[i]

                    self.dd[jc:jc+nj,:,:,:,:,:] = gdd[i]
                    self.dr[jc:jc+nj,:,:,:,:,:] = gdr[i]
                    self.rd[jc:jc+nj,:,:,:,:,:] = grd[i]
                    self.rr[jc:jc+nj,:,:,:,:,:] = grr[i]

                    jc += nj

                self.jnd1 = self.jackknife(self.nd1, reduce_jk=False)
                self.jnr1 = self.jackknife(self.nr1, reduce_jk=False)
                self.jnd2 = self.jackknife(self.nd2, reduce_jk=False)
                self.jnr2 = self.jackknife(self.nr2, reduce_jk=False)

                self.jdd = self.jackknife(self.dd, reduce_jk=False)
                self.jdr = self.jackknife(self.dr, reduce_jk=False)
                self.jrd = self.jackknife(self.rd,reduce_jk=False)
                self.jrr = self.jackknife(self.rr, reduce_jk=False)

                self.jxixy = np.zeros_like(self.jdd)

                for i in range(self.njacktot):
                    for j in range(self.ncbins):
                        for k in range(self.nmbins):
                            for l in range(self.nmbins1):
                                for m in range(self.nzbins*self.nzbins1):
                                    if (self.jnd1[i,0,j,k,l,m]<1) | (self.jnd2[i,0,j,k,l,m]<1): continue
                                    self.jxixy[i,:,j,k,l,m] = convert_3d_counts_to_cf(self.jnd1[i,0,j,k,l,m],
                                                                                      self.jnd2[i,0,j,k,l,m],
                                                                                      self.jnr1[i,0,j,k,l,m],
                                                                                      self.jnr2[i,0,j,k,l,m],
                                                                                      self.jdd[i,:,j,k,l,m],
                                                                                      self.jdr[i,:,j,k,l,m],
                                                                                      self.jrd[i,:,j,k,l,m],
                                                                                      self.jrr[i,:,j,k,l,m])

                self.xixy = np.sum(self.jxixy, axis=0) / self.njacktot

                self.varxixy = np.sum((self.jxixy - self.xixy)**2, axis=0) * (self.njacktot - 1) / self.njacktot

                print('sizes of jnd1, jdd, wxytheta: {}, {}, {}'.format(self.jnd1.shape, self.jdd.shape, self.xixy.shape))

        else:
            self.jnd1 = self.jackknife(self.nd1, reduce_jk=False)
            self.jnr1 = self.jackknife(self.nr1, reduce_jk=False)
            self.jnd2 = self.jackknife(self.nd2, reduce_jk=False)
            self.jnr2 = self.jackknife(self.nr2, reduce_jk=False)

            self.jdd = self.jackknife(self.dd, reduce_jk=False)
            self.jdr = self.jackknife(self.dr, reduce_jk=False)
            self.jrd = self.jackknife(self.rd,reduce_jk=False)
            self.jrr = self.jackknife(self.rr, reduce_jk=False)

            self.jxixy = np.zeros_like(self.jdd)

            for i in range(self.njacktot):
                for j in range(self.ncbins):
                    for k in range(self.nmbins):
                        for l in range(self.nmbins1):
                            for m in range(self.nzbins*self.nzbins1):
                                self.jxixy[i,:,j,k,l,m] = convert_3d_counts_to_cf(self.jnd1[i,0,j,k,l,m],
                                                                                  self.jnd2[i,0,j,k,l,m],
                                                                                  self.jnr1[i,0,j,k,l,m],
                                                                                  self.jnr2[i,0,j,k,l,m],
                                                                                  self.jdd[i,:,j,k,l,m],
                                                                                  self.jdr[i,:,j,k,l,m],
                                                                                  self.jrd[i,:,j,k,l,m],
                                                                                  self.jrr[i,:,j,k,l,m])

            self.xixy = np.sum(self.jxixy, axis=0) / self.njacktot

            self.varxixy = np.sum((self.jxixy - self.xixy)**2, axis=0) * (self.njacktot - 1) / self.njacktot
            print('sizes of jnd1, jdd, wxytheta: {}, {}, {}'.format(self.jnd1.shape, self.jdd.shape, self.xixy.shape))


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
                    ye = np.sqrt(self.varxixy[:,c,l,0,z])
                    l1 = ax[j][i].plot(rmean, self.xixy[:,c,l,0,z], **kwargs)
                    ax[j][i].fill_between(rmean, self.xixy[:,c,l,0,z]-ye, self.xixy[:,c,l,0,z]+ye, alpha=0.5, **kwargs)

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
            sax.set_ylabel(r'$\xi(r)$', labelpad=20, fontsize=16)
            sax.set_xlabel(r'$r \, [ Mpc\/h^{-1} ]$',labelpad=20, fontsize=16)

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


class DDCounts(CorrelationFunction):

    def __init__(self, ministry, mbins=None, zbins=None, rbins=None,
                  minr=None, maxr=None, logbins=True, nrbins=None,
                  lightcone=False, catalog_type=None, tag=None,
                  mcutind=None, inv_m=True,
                  centrals_only=False, **kwargs):

        """
        Real space 3-d normalized pairs, DD(r).
        """
        CorrelationFunction.__init__(self, ministry, lightcone=lightcone,
                                      mbins=mbins, nrbins=nrbins,
                                      mcutind=mcutind, zbins=zbins,
                                      inv_m=inv_m,catalog_type=catalog_type,
                                      tag=tag, **kwargs)

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

        if self.mkey is not None:
            self.mapkeys = ['px', 'py', 'pz', self.mkey]
        else:
            self.mapkeys = ['px','py','pz']

        self.centrals_only = centrals_only

        self.unitmap = {'px':'mpch', 'py':'mpch', 'pz':'mpch'}

        if self.centrals_only:
            self.mapkeys.append('central')
            self.unitmap['central'] = 'binary'

        if self.mkey == 'luminosity':
            self.unitmap[self.mkey] = 'mag'
        elif self.mkey == 'halomass':
            self.unitmap[self.mkey] = 'msunh'

        self.dd = None
        self.nd = None

    @jackknifeMap
    def map(self, mapunit):

        if not hascorrfunc:
            raise(ImportError("CorrFunc is required to calculate xi(r)"))

        if self.dd is None:
            self.writeCorrfuncBinFile(self.rbins)
            self.dd = np.zeros((self.njack, self.nrbins, self.nmbins, 1))
            self.nd = np.zeros((self.njack, self.nmbins, 1))

        if (mapunit['px'].dtype == '>f4') | (mapunit['px'].dtype == '>f8') | (mapunit['px'].dtype == np.float64):
            mu = {}
            mu['px'] = np.zeros(len(mapunit['px']), dtype=np.float32)
            mu['py'] = np.zeros(len(mapunit['py']), dtype=np.float32)
            mu['pz'] = np.zeros(len(mapunit['pz']), dtype=np.float32)

            mu['px'][:] = mapunit['px'][:]
            mu['py'][:] = mapunit['py'][:]
            mu['pz'][:] = mapunit['pz'][:]

            for f in self.mapkeys:
                if (f=='px') | (f=='py') | (f=='pz'): continue
                mu[f] = mapunit[f]

        else:
            mu = mapunit

        for lj, j in enumerate(self.minds):
            print('Finding luminosity indices')

            if not self.upper_limit:
                if self.mcutind is not None:
                    lidx = (self.mbins[j] <= mu[self.mkey][:,self.mcutind]) & (mu[self.mkey][:,self.mcutind] < self.mbins[j+1])
                elif (self.mcutind is None) and (self.mkey is not None):
                    lidx = (self.mbins[j] <= mu[self.mkey]) & (mu[self.mkey] < self.mbins[j+1])
                else:
                    lidx = np.ones(len(mu['px']), dtype=np.bool)
            else:
                if self.mcutind is not None:
                    lidx = (mu[self.mkey][:,self.mcutind] < self.mbins[j])
                elif (self.mcutind is None) and (self.mkey is not None):
                    lidx = (mu[self.mkey] < self.mbins[j])
                else:
                    lidx = np.ones(len(mu['px']), dtype=np.bool)

            if self.centrals_only:
                lidx &= (mu['central']==1)

            self.nd[self.jcount,j,0] = len(mu['pz'][lidx])

            print("Number of galaxies in this z/lum bin: {0}".format(len(mu['pz'][lidx])))
            print('calculating xi(r)')

            sys.stdout.flush()
            ddout = DD(1,self.nthreads,
                         self.rbins,
                         mu['px'][lidx],
                         mu['py'][lidx],
                         mu['pz'][lidx],
                         periodic=False)

            ddout = np.array(ddout)
            self.dd[self.jcount,:,j,0] = ddout['npairs']


    def reduce(self, rank=None, comm=None):
        if rank is not None:
            gnd = comm.gather(self.nd, root=0)
            gdd = comm.gather(self.dd, root=0)

            if rank==0:
                ndshape = [self.nd.shape[i] for i in range(len(self.nd.shape))]
                ddshape = [self.dd.shape[i] for i in range(len(self.dd.shape))]

                ndshape.insert(1,1)

                ndshape[0] = self.njacktot
                ddshape[0] = self.njacktot

                self.nd = np.zeros(ndshape)
                self.dd = np.zeros(ddshape)

                jc = 0
                for i, g in enumerate(gnd):
                    if g is None: continue
                    nj = g.shape[0]
                    self.nd[jc:jc+nj,0,:,:] = g
                    self.dd[jc:jc+nj,:,:,:] = gdd[i]

                    jc += nj

                self.jnd = self.jackknife(self.nd, reduce_jk=False)
                self.jdd = self.jackknife(self.dd, reduce_jk=False)

                self.jDD = self.jdd / self.jnd.reshape(-1,1,self.nmbins,1)

                self.DD    = np.sum(self.jDD, axis=0) / self.njacktot
                self.varDD = np.sum((self.jDD - self.DD)**2, axis=0) * (self.njacktot - 1) / self.njacktot

        else:
            self.jnd = self.jackknife(self.nd, reduce_jk=False)
            self.jdd = self.jackknife(self.dd, reduce_jk=False)

            self.jDD = self.jdd / self.jnd.reshape(-1,1,self.nmbins,1)

            self.DD    = np.sum(self.jDD, axis=0) / self.njacktot
            self.varDD = np.sum((self.jDD - self.DD)**2, axis=0) * (self.njacktot - 1) / self.njacktot


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
            ye = np.sqrt(self.varDD[:,l,0])
            l1 = ax[0][i].plot(rmean, self.DD[:,l,0], **kwargs)
            ax[0][i].fill_between(rmean, self.DD[:,l,0]-ye, self.DD[:,l,0]+ye, alpha=0.5, **kwargs)

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
            sax.set_ylabel(r'$DD(r)$', fontsize=16, labelpad=20)
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
                 massbins=None, subjack=False, mcutind=None,
                 catalog_type=['galaxycatalog'], splitcolor=None,
                 cinds=None, cbins=None, tag=None, lightcone=True,
                 **kwargs):
        """
        Radial profile of galaxies around their nearest halos.
        """

        Metric.__init__(self, ministry, tag=tag, **kwargs)

        self.catalog_type = catalog_type

        self.lightcone = lightcone

        if self.lightcone:
            if zbins is None:
                self.zbins = [0.0, 0.2]
            else:
                self.zbins = zbins
                self.zbins = np.array(self.zbins)

            self.nzbins = len(self.zbins)-1
        else:
            self.zbins = None
            self.nzbins = 1

        self.splitcolor = splitcolor

        if self.splitcolor is not None:
            self.ncbins = 2
            if cinds is None:
                self.cinds = [0,1]
            else:
                self.cinds = cinds
        else:
            self.ncbins = 1



        if lumbins is None:
            self.lumbins = np.array([-22, -21, -20, -19])
        else:
            self.lumbins = lumbins

        self.nlumbins = len(self.lumbins)-1

        self.mcutind = mcutind

        if massbins is None:
            self.massbins = np.logspace(np.log10(5e12),15,5)
        else:
            self.massbins = massbins

        self.nmassbins = len(self.massbins) - 1

        if rbins is None:
            self.rbins = np.logspace(-2, 1, 21)
        else:
            self.rbins = rbins

        self.nrbins = len(self.rbins)-1

        self.aschema = 'galaxyonly'

        self.mapkeys = ['luminosity', 'rhalo', 'halomass', 'haloid']
        self.unitmap = {'luminosity':'mag', 'polar_ang':'dec', 'azim_ang':'ra',
                        'halomass':'msunh'}

        if self.lightcone:
            self.mapkeys.append('redshift')
            self.unitmap['redshift'] = 'z'

        self.rcounts = None
        self.hcounts = None

    @jackknifeMap
    def map(self, mapunit):

        if self.rcounts is None:
            self.rcounts = np.zeros((self.njack, self.nrbins,
                                     self.ncbins, self.nlumbins,
                                     self.nmassbins,
                                     self.nzbins))
            self.hcounts = np.zeros((self.njack, self.nmassbins,
                                     self.nzbins))

        if self.ncbins>1:
            color = mapunit['luminosity'][:,self.cinds[0]] - mapunit['luminosity'][:,self.cinds[1]]

        if self.lightcone:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

                for j, m in enumerate(self.massbins[:-1]):
                    midx = ((self.massbins[j] < mapunit['halomass'][zlidx:zhidx])
                            & (mapunit['halomass'][zlidx:zhidx] <= self.massbins[j+1])).reshape(zhidx-zlidx)
                    self.hcounts[self.jcount,j,i] += len(np.unique(mapunit['haloid'][zlidx:zhidx]))

                    for k, l in enumerate(self.lumbins[:-1]):
                        if self.mcutind is not None:
                            lidx = ((self.lumbins[k]<mapunit['luminosity'][zlidx:zhidx,self.mcutind])
                                    & (mapunit['luminosity'][zlidx:zhidx,self.mcutind]<self.lumbins[k+1])).reshape(zhidx-zlidx)
                        else:
                            lidx = ((self.lumbins[k]<mapunit['luminosity'][zlidx:zhidx])
                                    & (mapunit['luminosity'][zlidx:zhidx]<self.lumbins[k+1])).reshape(zhidx-zlidx)

                        for n in range(self.ncbins):
                            if self.ncbins>1:
                                if n==0:
                                    cidx = lidx & (color[zlidx:zhidx]<self.splitcolor)
                                else:
                                    cidx = lidx & (color[zlidx:zhidx]>=self.splitcolor)
                            else:
                                cidx = lidx


                            c, e = np.histogram(mapunit['rhalo'][zlidx:zhidx][midx&cidx], bins=self.rbins)
                            self.rcounts[self.jcount,:,n,k,j,i] += c
        else:
            for j, m in enumerate(self.massbins[:-1]):
                midx = ((self.massbins[j] < mapunit['halomass'][:])
                        & (mapunit['halomass'][:] <= self.massbins[j+1])).reshape(len(mapunit['halomass']))
                self.hcounts[self.jcount,j,0] += len(np.unique(mapunit['haloid'][:]))

                for k, l in enumerate(self.lumbins[:-1]):
                    if self.mcutind is not None:
                        lidx = ((self.lumbins[k]<mapunit['luminosity'][:,self.mcutind])
                                & (mapunit['luminosity'][:,self.mcutind]<self.lumbins[k+1])).reshape(len(mapunit['halomass']))
                    else:
                        lidx = ((self.lumbins[k]<mapunit['luminosity'][:])
                                & (mapunit['luminosity'][:]<self.lumbins[k+1])).reshape(len(mapunit['halomass']))

                    for n in range(self.ncbins):
                        if self.ncbins>1:
                            if n==0:
                                cidx = lidx & (color[:]<self.splitcolor)
                            else:
                                cidx = lidx & (color[:]>=self.splitcolor)
                        else:
                            cidx = lidx

                        c, e = np.histogram(mapunit['rhalo'][:][midx&cidx], bins=self.rbins)
                        self.rcounts[self.jcount,:,n,k,j,0] += c


    def reduce(self, rank=None, comm=None):

        if rank is not None:
            grcounts = comm.gather(self.rcounts, root=0)
            ghcounts = comm.gather(self.hcounts, root=0)
            if rank==0:
                dshape = self.rcounts.shape
                hshape = self.hcounts.shape
                dshape = [dshape[i] for i in range(len(dshape))]
                hshape = [hshape[i] for i in range(len(hshape))]
                dshape[0] = self.njacktot
                hshape[0] = self.njacktot
                self.rcounts = np.zeros(dshape)
                self.hcounts = np.zeros(hshape)

                jc = 0
                #iterate over gathered arrays, filling in arrays of rank==0
                #process
                for i,g in enumerate(grcounts):
                    if g is None: continue
                    nj = g.shape[0]
                    self.rcounts[jc:jc+nj,:,:,:,:,:] = g
                    self.hcounts[jc:jc+nj,:,:] = ghcounts[i]

                    jc += nj

                self.rmean = (self.rbins[1:] + self.rbins[:-1]) / 2
                vol = 4 * np.pi * (self.rmean**3) / 3

                self.jurprof = self.rcounts / vol.reshape((1,self.nrbins,1,1,1,1))

                self.jurprof  = self.jackknife(self.jurprof, reduce_jk=False)
                self.jhcounts = self.jackknife(self.hcounts, reduce_jk=False)

                self.jrprof   = self.jurprof / self.jhcounts.reshape(-1,1,1,1,self.nmassbins,self.nzbins)

                self.rprof    = np.sum(self.jrprof, axis=0) / self.njacktot
                self.varrprof = (np.sum((self.jrprof - self.rprof)**2, axis=0) * (self.njacktot - 1)
                                   / self.njacktot)

        else:
            self.rmean = (self.rbins[1:] + self.rbins[:-1]) / 2
            vol = 4 * np.pi * (self.rmean**3) / 3

            self.jurprof = self.rcounts / vol.reshape((1,self.nrbins,1,1,1,1))

            self.jurprof  = self.jackknife(self.jurprof, reduce_jk=False)
            self.jhcounts = self.jackknife(self.hcounts, reduce_jk=False)

            self.jrprof   = self.jurprof / self.jhcounts.reshape(-1,1,1,1,self.nmassbins,self.nzbins)
            self.rprof    = np.sum(self.jrprof, axis=0) / self.njacktot
            self.varrprof = (np.sum((self.jrprof - self.rprof)**2, axis=0) * (self.njacktot - 1)
                               / self.njacktot)


    def visualize(self, plotname=None, f=None, ax=None,
                  compare=False, usecols=None, usez=None,
                  uselum=None, usecolor=None, xlabel=None,
                  ylabel=None, logx=True, logy=True, **kwargs):

        if usez is None:
            usez = range(self.nzbins)
        if usecols is None:
            usecols = range(self.nmassbins)
        if uselum is None:
            uselum = range(self.nlumbins)
        if usecolor is None:
            usecolor = range(self.ncbins)

        if f is None:
            f, ax = plt.subplots(len(usez), len(usecols), sharex=True,
                                    sharey=True, figsize=(8,8))
            ax = np.array(ax)
            ax = ax.reshape(len(usez), len(usecols))
            newaxes = True
        else:
            newaxes = False

        for i, z in enumerate(usez):
            for j, c in enumerate(usecols):
                for k, lu in enumerate(uselum):
                    for n, cl in enumerate(usecolor):
                        ye = np.sqrt(self.varrprof[:,cl,lu,j,i])
                        l = ax[i][j].plot(self.rmean, self.rprof[:,cl,lu,c,z],
                                          **kwargs)
                        ax[i][j].fill_between(self.rmean, self.rprof[:,cl,lu,c,z]-ye,
                                              self.rprof[:,cl,lu,c,z]+ye,alpha=0.5,**kwargs)

        if logx:
            ax[0][0].set_xscale('log')
        if logy:
            ax[0][0].set_yscale('log')

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
            sax.set_xlabel(r'$r\, [Mpc \, h^{-1}]$', labelpad=40, fontsize=16)
            sax.set_ylabel(r'$\rho_{g} \, [Mpc^{3} \, h^{-3}]$', labelpad=40, fontsize=16)

        if (plotname is not None) & (not compare):
            plt.savefig(plotname)

        return f, ax, l

    def compare(self, othermetrics, plotname=None, usecols=None,
                 usez=None, uselum=None, labels=None, **kwargs):

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

        if uselum is not None:
            if not hasattr(uselum[0], '__iter__'):
                uselum = [uselum]*len(tocompare)
            else:
                assert(len(uselum)==len(tocompare))
        else:
            uselum = [None]*len(tocompare)


        if labels is None:
            labels = [None]*len(tocompare)

        lines = []

        for i, m in enumerate(tocompare):
            if usecols[i] is not None:
                assert(len(usecols[0])==len(usecols[i]))
            if i==0:
                f, ax, l1 = m.visualize(usecols=usecols[i], usez=usez[i],
                                          compare=True, color=Metric._color_list[i],
                                          uselum=uselum[i],**kwargs)
            else:
                f, ax, l1 = m.visualize(usecols=usecols[i], usez=usez[i],
                                          compare=True, color=Metric._color_list[i],
                                          f=f, ax=ax, uselum=uselum[i],**kwargs)
            lines.append(l1)

        if labels[0]!=None:
            f.legend(lines, labels)

        if plotname is not None:
            plt.savefig(plotname)

        return f, ax
