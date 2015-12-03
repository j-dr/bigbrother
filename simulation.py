from __future__ import print_function, division

class Simulation:
    """
    A class which owns all the other catalog data 
    """
    
    def __init__(self, omega_m, omega_l, h):
        
        self.omega_m = omega_m
        self.omega_l = omega_l
        self.h = h

    def setGalaxyCatalog(self, catalog_type, filestruct, input_LF=None):
        """
        Fill in the galaxy catalog information for this simulation
        """

        if catalog_type == "BCC":
            self.galaxycatalog = BCCCatalog(filestruct, input_LF=input_LF)

    def setHaloCatalog(self, catalog_type, filestruct):
        """
        Fill in the halo catalog information for this simulation
        """

    def setNBody(self, simulation_type):
        """
        Fill in NBody parameters and particles
        """

    def validate(self, metrics=None):
        """
        Run all validation metrics by iterating over only the files we
        need at a given time, mapping catalogs to relevant statistics
        which are reduced at the end of the iteration into observables 
        that we care about
        """

        #For now, just focus on galaxy observables
        #catalogs that need to be in memory at a particular
        #moment get more complicated when calculating galaxy-halo 
        #relation statistics

        for f in self.galaxycatalog.mappable:
            self.galaxycatalog.map(f)

        self.galaxycatalog.reduce()
        
class GalaxyCatalog:
    """
    Base class for galaxy catalogs
    """
    
    __metaclass__ = ABCMeta
    
    def __init__(self, filestruct, input_LF=None):
        self.filestruct = filestruct

    #@abstractmethod
    #def validate(self):
    #    pass
    
    @abstractmethod
    def parseFileStruct(self, filestruct):
        """
        Given a filestruct object, create map from parameters 
        we require to filepaths for easy access
        """

    @abstractmethod
    def map(self, mappable):
        """
        Do some operations on a mappable unit of the catalog
        """
        
    @abstractmethod
    def reduce(self, mappable):
        """
        Do some operations on a mappable unit of the catalog
        """

class BCCCatalog(GalaxyCatalog):
    """
    BCC style ADDGALS catalog
    """

    def __init__(self, filestruct, input_LF=None):
        GalaxyCatalog.__init__(self, filestruct)
        self.input_LF = input_LF
        
