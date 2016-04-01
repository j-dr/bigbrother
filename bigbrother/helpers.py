from __future__ import print_function
import numpy as np


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
