from __future__ import print_function
from .metric import Metric, GMetric
import numpy as np
import healpy as hp


def sortHpixFileStruct(filestruct):

    if len(filestruct.keys())>1:
        opix =  np.array([int(t.split('/')[-1].split('.')[-2]) for t
                          in filestruct[filetypes[0]]])
        oidx = opix.argsort()
        
        for ft in filetypes:
            assert(len(filestruct[ft])==len(filestruct[filetypes[0]]))
            pix = np.array([int(t.split('/')[-1].split('.')[-2]) for t
                            in filestruct[ft]])
            idx = pix.argsort()
            assert(pix[idx]==opix[oidx])
            
            if len(idx)==1:
                filestruct[ft] = [filestruct[ft][idx]]
            else:
                filestruct[ft] = filestruct[ft][idx]
    
    return filestruct

class PixMetric(Metric):

    def __init__(self, ministry, nside)
        """
        Initialize a PixMetric object. Note, all metrics should define
        an attribute called mapkeys which specifies the types of data that they
        expect.

        Arguments
        ---------
        ministry : Ministry
            The ministry object that this metric is associated with.
        """
        Metric.__init__(self, ministry)

        self.nside

        self.mapkeys = ['polar_ang', 'azim_ang']
        self.aschema = 'singleonly'

        
    def map(self, mapunit):
        
        pix = hp.ang2pix(self.nside, mapunit['polar_ang'], mapunit['azim_ang'])
        
        return pix
