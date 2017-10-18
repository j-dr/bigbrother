from __future__ import print_function, division
from collections import OrderedDict
from .corrmetric import WPrpSnapshotAnalyticRandoms
from .healpix_utils import PixMetric
from .basecatalog     import BaseCatalog
from astropy.cosmology import z_at_value
from collections import namedtuple

from helpers import SimulationAnalysis
import astropy.units as u
import numpy as np
import healpy as hp
import struct
import sys


class ParticleCatalog(BaseCatalog):
    """
    Base class for particle catalogs
    """

    __GadgetHeader_fmt = '6I6dddii6Iiiddddii6Ii'
    __finenside = 8192

    GadgetHeader = namedtuple('GadgetHeader', \
        'npart mass time redshift flag_sfr flag_feedback npartTotal flag_cooling num_files BoxSize Omega0 OmegaLambda HubbleParam flag_age flag_metals NallHW flag_entr_ics')


    def __init__(self, ministry, filestruct, downsample_factor=None, **kwargs):

        self.ctype = 'particlecatalog'
        BaseCatalog.__init__(self, ministry, filestruct, **kwargs)

        self.downsample_factor = downsample_factor

    def parseFileStruct(self, filestruct):
        """
        Given a filestruct object, namely a list of truth
        and/or obs files, map fields in these files
        to generalized observables which our map functions
        know how to deal with
        """
        self.filestruct = filestruct
        self.filetypes = self.filestruct.keys()

    def unitConversion(self, mapunit):

        midx = mapunit['halomass']!=0.0

        for mapkey in mapunit.keys():
            mapunit[mapkey] = mapunit[mapkey][midx]
            if mapkey=='halomass':
                mapunit[mapkey] = np.log10(mapunit[mapkey][midx])

        return mapunit

    def readMappable(self, mappable, fieldmap):
        """
        Takes a mappable object and a fieldmap as inputs
        and returns a mapunit containing the data required
        by the fieldmap.
        """
        if self.reader=='fits':
            mapunit = self.readFITSMappable(mappable, fieldmap)
        if self.reader=='lgadget':
            mapunit = self.readLGadgetMappable(mappable, fieldmap)

        return mapunit

    def readLGadgetMappable(self, mappable, fieldmap):
        """
        Takes in a mappable object, and a fieldmap and spits out a mappable
        """

	mapunit = {}
	ft      = mappable.dtype
	fname   = mappable.name

	for f in fieldmap.keys():
	    fields = []
	    for val in fieldmap[ft].values():
	        if hasattr(val, '__iter__'):
	            fields.extend(val)
	        else:
	            fields.extend([val])

        fields = list(np.unique(fields))

	data = self.readGadgetBlock(fname, fields, lgadget=True, downsample=self.downsample_factor)

	for mapkey in fieldmap[ft].keys():
            if 'px' in mapkey:
                mapunit[mapkey] = data[fieldmap[ft][mapkey]][:,0]
            elif 'py' in mapkey:
                mapunit[mapkey] = data[fieldmap[ft][mapkey]][:,1]
            elif 'pz' in mapkey:
                mapunit[mapkey] = data[fieldmap[ft][mapkey]][:,2]
            elif 'vx' in mapkey:
                mapunit[mapkey] = data[fieldmap[ft][mapkey]][:,0]
            elif 'vy' in mapkey:
                mapunit[mapkey] = data[fieldmap[ft][mapkey]][:,1]
            elif 'vz' in mapkey:
                mapunit[mapkey] = data[fieldmap[ft][mapkey]][:,2]
            else:
                mapunit[mapkey] = data[fieldmap[ft][mapkey]]
                if hasattr(fieldmap[ft][mapkey], '__iter__'):
                    dt = mapunit[mapkey].dtype[0]
                    ne = len(mapunit[mapkey])
                    nf = len(fieldmap[ft][mapkey])
                    mapunit[mapkey] = mapunit[mapkey].view(dt).reshape((ne,nf))
            
	return mapunit

    def readGadgetBlock(self, filename, fields, print_header=False, single_type=-1, 
                        lgadget=False, downsample=None):
        """
        This function reads the Gadget-2 snapshot file.
        
        Parameters
        ----------
        filename : str
        path to the input file
        read_pos : bool, optional
        Whether to read the positions or not. Default is false.
        read_vel : bool, optional
        Whether to read the velocities or not. Default is false.
        read_id : bool, optional
        Whether to read the particle IDs or not. Default is false.
        read_mass : bool, optional
        Whether to read the masses or not. Default is false.
        print_header : bool, optional
        Whether to print out the header or not. Default is false.
        single_type : int, optional
        Set to -1 (default) to read in all particle types. 
        Set to 0--5 to read in only the corresponding particle type.
        lgadget : bool, optional
        Set to True if the particle file comes from l-gadget. 
        Default is false.
        
        Returns
        -------
        ret : tuple
        A tuple of the requested data. 
        The first item in the returned tuple is always the header.
        The header is in the GadgetHeader namedtuple format.
        """
        if 'position' in fields:
            read_pos = True
        else:
            read_pos = False
        if 'velocity' in fields:
            read_vel = True
        else:
            read_vel = False
        if 'id' in fields:
            read_id = True
        else:
            read_id = False
        if 'mass' in fields:
            read_mass = True
        else:
            read_mass = False

        blocks_to_read = (read_pos, read_vel, read_id, read_mass)
        ret = []
        fields = []
        with open(filename, 'rb') as f:
            f.seek(4, 1)
            h = list(struct.unpack( ParticleCatalog.__GadgetHeader_fmt, f.read(struct.calcsize(ParticleCatalog.__GadgetHeader_fmt))))
            if lgadget:
                h[30] = 0
                h[31] = h[18]
                h[18] = 0
                single_type = 1
            h = tuple(h)
            header = ParticleCatalog.GadgetHeader._make((h[0:6],) + (h[6:12],) + h[12:16] \
                                            + (h[16:22],) + h[22:30] + (h[30:36],) + h[36:])
            if print_header:
                print( header )
            if not any(blocks_to_read):
                return header
            ret.append(header)
            fields.append('header')
            f.seek(256 - struct.calcsize(ParticleCatalog.__GadgetHeader_fmt), 1)
            f.seek(4, 1)

            mass_npart = [0 if m else n for m, n in zip(header.mass, header.npart)]
            if single_type not in range(6):
                single_type = -1

            for i, b in enumerate(blocks_to_read):
                if i < 2:
                    fmt = np.dtype(np.float32)
                    item_per_part = 3
                    npart = header.npart
                elif i==2:
                    fmt = np.dtype(np.uint64) if lgadget or any(header.NallHW) \
                        else np.dtype(np.uint32)
                    item_per_part = 1
                    npart = header.npart
                elif i==3:
                    fmt = np.dtype(np.float32)
                    if sum(mass_npart) == 0:
                        ret.append(np.array([], fmt))
                        break
                    item_per_part = 1
                    npart = mass_npart
                size_per_part = item_per_part*fmt.itemsize

                f.seek(4, 1)
                if not b:
                    f.seek(sum(npart)*size_per_part, 1)
                else:
                    if i==0:
                        fields.append('position')
                    elif i==1:
                        fields.append('velocity')
                    elif i==2:
                        fields.append('id')
                    elif i==3:
                        fields.append('mass')

                    if single_type > -1:
                        f.seek(sum(npart[:single_type])*size_per_part, 1)
                        npart_this = npart[single_type]
                    else:
                        npart_this = sum(npart)
                    data = np.fromstring(f.read(npart_this*size_per_part), fmt)
                    if item_per_part > 1:
                        data.shape = (npart_this, item_per_part)

                    if downsample:
                        try:
                            idx = np.random.choice(np.arange(npart_this), 
                                                   size=npart_this/downsample, 
                                                   replace=False)
                        except RuntimeError as e:
                            sys.setrecursionlimit(10000)
                            print(e)
                            print('Failed on file {}'.format(filename))
                            print('Number of particles in this file, and downsampled: {}, {}'.format(npart_this, npart_this/downsample))

                            idx = np.random.choice(np.arange(npart_this), 
                                                   size=npart_this/downsample, 
                                                   replace=False)

                        data = data[idx,:]

                    ret.append(data)
                    if not any(blocks_to_read[i+1:]):
                        break
                    if single_type > -1:
                        f.seek(sum(npart[single_type+1:])*size_per_part, 1)
                f.seek(4, 1)

        return dict(zip(fields, ret))

class LGadgetSnapshot(ParticleCatalog):
    """
    Class to handle L-Gadget snapshot outputs
    """

    def __init__(self, ministry, filestruct, **kwargs):

        if 'unitmap' not in kwargs.keys():
            kwargs['unitmap'] = {'px':'mpch', 'py':'mpch', 'pz':'mpch',
                                 'vx':'kms', 'vy':'kms', 'vz':'kms'}

        ParticleCatalog.__init__(self, ministry, filestruct, reader='lgadget', **kwargs)

        if self.fieldmap is None:
            self.fieldmap = {'px':OrderedDict([('position',['ptruth'])]),
                             'py':OrderedDict([('position',['ptruth'])]),
                             'pz':OrderedDict([('position',['ptruth'])]),
                             'vx':OrderedDict([('velocity',['ptruth'])]),
                             'vy':OrderedDict([('velocity',['ptruth'])]),
                             'vz':OrderedDict([('velocity',['ptruth'])]),
                             'id':OrderedDict([('id', ['ptruth'])])}

            self.hasz = False
            self.sortbyz = False

        self.lgadget=True
        
        
    def parseFileStruct(self, filestruct):
        """
        Given a filestruct object, map fields in these files
        to generalized observables which our map functions
        know how to deal with
        """
        self.filestruct = filestruct
        filetypes = self.filestruct.keys()
        self.filetypes = filetypes

        if len(self.filetypes)>1:
            raise(ValueError("""Only one type of file per particle catalog
                              is currently supported"""))

