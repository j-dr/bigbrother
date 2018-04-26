from __future__ import print_function, division
from .metric import GMetric, jackknifeMap
import matplotlib.pyplot as plt
import numpy as np

class VelocityDistribution(GMetric):

    def __init__(self, ministry, zbins=None, mbins=None,
                   vbins=None, subjack=False, lightcone=True,
                   catalog_type=None, mcutind=None,
                   tag=None, upper_limit=False, centrals_only=False,
                   **kwargs):
        """
        Distribution of velocities for tracers.
        """
        GMetric.__init__(self, ministry, zbins=zbins, xbins=vbins,
                         catalog_type=catalog_type, tag=tag, **kwargs)

        if catalog_type is None:
            self.catalog_type = ['galaxycatalog']
        else:
            self.catalog_type = catalog_type

        self.upper_limit = upper_limit

        self.lightcone = lightcone

        if (mbins is None) & (self.catalog_type == ['galaxycatalog']):
            self.mbins = np.array([-30, 0])
        elif (mbins is None) & (self.catalog_type == ['halocatalog']):
            self.mbins = np.array([10**7, 10**17])
        elif self.catalog_type == ['particlecatalog']:
            self.mbins = [0,1]
        else:
            self.mbins = mbins

        if self.upper_limit:
            self.nmbins = len(self.mbins)
        else:
            self.nmbins = len(self.mbins)-1

        self.subjack = subjack

        if 'galaxycatalog' in self.catalog_type:
            self.aschema = 'galaxygalaxy'
            self.mkey = 'luminosity'
        elif 'halocatalog' in self.catalog_type:
            self.mkey = 'halomass'
            self.aschema = 'halohalo'
        else:
            self.mkey = None
            self.aschema = 'particleparticle'

        self.mcutind = mcutind
        self.centrals_only = centrals_only 

        if self.subjack:
            raise NotImplementedError

        self.jsamples = 0

        self.mapkeys = ['vx', 'vy', 'vz']
        self.unitmap = {'vx':'kms', 'vy':'kms', 'vz':'kms'}

        if self.mkey is not None:
            self.mapkeys.append(self.mkey)
            if self.mkey == 'halomass':
                self.unitmap[self.mkey] = 'msunh'
            elif self.mkey == 'luminosity':
                self.unitmap[self.mkey] = 'magh'

        if self.centrals_only:
            self.mapkeys.append('central')
            self.unitmap['central'] = 'binary'
            
        self.velocity_counts = None

    @jackknifeMap
    def map(self, mapunit):

        if self.velocity_counts is None:
            self.velocity_counts = np.zeros((self.njack, self.nxbins, self.nmbins, self.nzbins))

        v = np.sqrt(mapunit['vx']**2 + mapunit['vy']**2 + mapunit['vz']**2)
            
        if self.lightcone:
            for i, z in enumerate(self.zbins[:-1]):
                zlidx = mapunit['redshift'].searchsorted(self.zbins[i])
                zhidx = mapunit['redshift'].searchsorted(self.zbins[i+1])

                for j in range(self.nmbins):
                    if self.mkey is not None:
                        if not self.upper_limit:
                            if self.mcutind is not None:
                                midx = ((self.mbins[j] <= mapunit[self.mkey][zlidx:zhidx,self.mcutind]) &
                                        (mapunit[self.mkey][zlidx:zhidx,self.mcutind]< self.mbins[j+1]))
                            else:
                                midx = ((self.mbins[j] <= mapunit[self.mkey][zlidx:zhidx]) &
                                        (mapunit[self.mkey][zlidx:zhidx]< self.mbins[j+1]))
                        else:
                            if self.mcutind is not None:
                                midx = (self.mbins[j] <= mapunit[self.mkey][zlidx:zhidx,self.mcutind])
                            else:
                                midx = (self.mbins[j] <= mapunit[self.mkey][zlidx:zhidx])
                    else:
                        midx = np.ones(zhidx-zlidx, dtype=np.bool)
                            
                    c, e = np.histogram(v[zlidx:zhidx][midx],
                                            bins=self.xbins)
                    self.velocity_counts[self.jcount,:,j,i] += c

        else:
            for j in range(self.nmbins):
                if self.mkey is not None:
                    if not self.upper_limit:
                        if self.mcutind is not None:
                            midx = ((self.mbins[j] <= mapunit[self.mkey][:,self.mcutind]) &
                                    (mapunit[self.mkey][:,self.mcutind]< self.mbins[j+1]))
                        else:
                            midx = ((self.mbins[j] <= mapunit[self.mkey][:]) &
                                    (mapunit[self.mkey][:]< self.mbins[j+1]))
                    else:
                        if self.mcutind is not None:
                            midx = (self.mbins[j] <= mapunit[self.mkey][:,self.mcutind])
                        else:
                            midx = (self.mbins[j] <= mapunit[self.mkey][:])

                else:
                    midx = np.ones(len(mapunit['vx']), dtype=np.bool)
                    
                c, e = np.histogram(v[midx],bins=self.xbins)
                self.velocity_counts[self.jcount,:,j,0] += c

    def reduce(self, rank=None, comm=None):

        if rank is not None:
            gdata = comm.gather(self.velocity_counts, root=0)

            if rank==0:
                gshape = [self.velocity_counts.shape[i] for i in range(len(self.velocity_counts.shape))]
                gshape[0] = self.njacktot

                self.velocity_counts = np.zeros(gshape)
                jc = 0
                #iterate over gathered arrays, filling in arrays of rank==0
                #process
                for g in gdata:
                    if g is None: continue
                    nj = g.shape[0]
                    self.velocity_counts[jc:jc+nj,:,:,:] = g

                    jc += nj

                cat  = getattr(self.ministry, self.catalog_type[0])
                area = cat.getArea(jackknife=True)
                vol = np.zeros((self.njacktot, self.nzbins))

                dv  = (self.xbins[1:] - self.xbins[:-1]).reshape(1,-1,1,1)
                if self.lightcone:
                    for i in range(self.nzbins):
                        vol[:,i] = self.ministry.calculate_volume(area, self.zbins[i], self.zbins[i+1])
                else:
                    vol[:,0] = self.ministry.boxsize**3 / self.njacktot

                self.jvelocity_counts  = self.jackknife(self.velocity_counts, reduce_jk=False)
                self.jvelocity_function = self.jvelocity_counts / vol.reshape(self.njacktot, 1, 1, -1) / dv

                self.velocity_function = np.sum(self.jvelocity_function, axis=0) / self.njacktot
                self.varvelocity_function = np.sum((self.jvelocity_function - self.velocity_function) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot
                self.y = self.velocity_function
                self.ye = np.sqrt(self.varvelocity_function)
        else:
            cat  = getattr(self.ministry, self.catalog_type[0])


            if self.jtype is not None:
                area = cat.getArea(jackknife=True)
            else:
                area = cat.getArea(jackknife=False)

            vol = np.zeros((self.njacktot, self.nzbins))
            dv  = (self.xbins[1:] - self.xbins[:-1]).reshape(1,-1,1,1)

            if self.lightcone:
                for i in range(self.nzbins):
                    vol[:,i] = self.ministry.calculate_volume(area, self.zbins[i], self.zbins[i+1])
            else:
                vol[:,0] = self.ministry.boxsize**3 / self.njacktot

            self.jvelocity_counts  = self.jackknife(self.velocity_counts, reduce_jk=False)
            self.jvelocity_function = self.jvelocity_counts / vol.reshape(self.njacktot, 1, 1, -1) / dv

            self.velocity_function = np.sum(self.jvelocity_function, axis=0) / self.njacktot
            self.varvelocity_function = np.sum((self.jvelocity_function - self.velocity_function) ** 2, axis=0) * (self.njacktot - 1) / self.njacktot
            self.y = self.velocity_function
            self.ye = np.sqrt(self.varvelocity_function)
        

    def visualize(self, plotname=None, usecols=None, usez=None,fracdev=False,
                  ref_y=None, ref_x=[None], xlim=None, ylim=None, fylim=None,
                  f=None, ax=None, xlabel=None,ylabel=None,compare=False,**kwargs):

        if xlabel is None:
            xlabel = r"$v$"

        if ylabel is None:
            ylabel = r'$\phi \, [Mpc^{-3}\, h^{3}]$'

        self.nbands = self.nmbins

        return GMetric.visualize(self, plotname=plotname, usecols=usecols, usez=usez,
                                 fracdev=fracdev, ref_y=ref_y, ref_x=ref_x, xlim=xlim,
                                 ylim=ylim, fylim=fylim, f=f, ax=ax, xlabel=xlabel,
                                 ylabel=ylabel, compare=compare, **kwargs)
