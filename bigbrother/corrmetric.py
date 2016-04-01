from __future__ import print_function, division
if __name__=='__main__':
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pylab as plt
import treecorr as tc
import numpy as np
import .ministry



class AngularCorrelationFunction(Metric):

    def __init__(self, ministry, zbins=None, lumbins=None):
        Metric.__init__(self, ministry)

        if zbins is None:
            self.zbins = [0.0, 0.2]
        else:
            self.zbins = zbins
            self.zbins = np.array(self.zbins)

        self.nzbins = len(self.zbins)-1

        if lumbins is None:
            self.lumbins = np.linspace(-25, -11, 30)
        else:
            self.lumbins = lumbins

        self.mapkeys = ['luminosity', 'redshift', 'polar_ang', 'azim_ang']
        
