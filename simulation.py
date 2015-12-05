from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import fitsio

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

        mappables = self.galaxycatalog.genMappable(metrics)

        #probably want Simulation to have map method
        #which knows how to combine different 
        #types of catalogs
        for f in mappable:
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
    def genMappable(self, metrics):
        """
        Given a set of metrics, generate a list of mappables
        which can be fed into map functions
        """
        
    @abstractmethod
    def map(self, mappable):
        """
        Do some operations on a mappable unit of the catalog
        """
        
    @abstractmethod
    def reduce(self):
        """
        Reduce the information produced by the map operations
        """

    def luminosityFunctionMap(self, mapunit):
        """
        A really simple example of what a map function should look like.
        """
        #should check to see if all set up operations have been performed
        nbands = mapunit['luminosity'].shape[2]

        if not hasattr(self, 'lumcounts'):
            self.lumcounts = np.zeros((len(self.lumbins)-1, nbands))

        self.lumcounts += np.histogram(lums, bins=self.lumbins)

    def luminosityFunctionReduce(self):
        """
        Given counts in luminosity bins, generate a luminosity function
        """
        
        self.luminosity_function = self.lumcounts/self.volume
        

class BCCCatalog(GalaxyCatalog):
    """
    BCC style ADDGALS catalog
    """

    def __init__(self, filestruct, input_LF=None):
        GalaxyCatalog.__init__(self, filestruct)
        self.input_LF = input_LF
        self.filestruct = filestruct
        self.parseFileStruct(filestruct)
        self.metrics = ['lf', 'counts/mag', 'redshift']
        self.fieldmap = {'luminosity':{'AMAG':['obs','truth',]},
                         'appmag':{'MAG_G':['obs'], 'MAG_R':['obs'],
                                   'MAG_I':['obs'], 'MAG_Z':['obs'],
                                   'MAG_Y':['obs']},
                         'redshift':{'Z':['truth']}}
        
    def parseFileStruct(self, filestruct):
        """
        Given a filestruct object, namely a list of truth 
        and/or obs files, map fields in these files 
        to generalized observables which our map functions
        know how to deal with
        """
        self.filestruct = filestruct
        filetypes = self.filestruct.keys()
        self.filetypes = filetypes        

        if len(filestruct.keys())>1:
            opix =  np.array([int(t.split('/')[-1].split('.')[-2]) for t
                              in self.filestruct[filetypes[0]]])
            oidx = opix.argsort()

            for ft in filetypes:
                assert(len(filestruct[ft])==len(filestruct[filetypes[0]]))
                pix = np.array([int(t.split('/')[-1].split('.')[-2]) for t
                    in self.filestruct[ft]])
                idx = pix.argsort()
                assert(pix[idx]==opix[oidx])
                
                self.filestruct[ft] = self.filestruct[ft][idx]

    def setFieldMap(self, fieldmap):
        self.fieldmap = fieldmap

    def genMappable(self, metrics):
        """
        Given a set of metrics, generate a list of mappables
        which can be fed into map functions
        """
        mappables = []
        mapkeys = []
        fieldmap = {}
        usedfiletypes = []

        if metrics!=None:
            self.metrics = metrics

        for m in self.metrics:
            if m=='lf':
                mapkeys.append('luminosity')
            if m=='counts/mag':
                mapkeys.append('appmag')
                pass
            if m=='redshift':
                mapkeys.append('redshift')
        
        #for each type of data necessary for 
        #the metrics we want to calculate,
        #determine the file type it's located
        #in and the field 
        for mapkey in mapkeys:
            fileinfo = self.fieldmap[mapkey]
            for field in fileinfo.keys():
                valid = False
                filetypes = fileinfo[field]
                for ft in filetypes:
                    if ft in self.filetypes:
                        #if already have one field
                        #make a list of fields
                        if ft not in fieldmap.keys():
                            fieldmap[ft] = {}
                        if mapkey in fieldmap[ft].keys():
                            if hasattr(fieldmap[ft][mapkey],'__iter__'):
                                fieldmap[ft][mapkey].append(field)
                            else:
                                fieldmap[ft][mapkey] = [fieldmap[ft][mapkey],field]
                        else:
                            fieldmap[ft][mapkey] = field

                        valid = True
                        if ft not in usedfiletypes:
                            usedfiletypes.append(ft)
                        break

                if not valid:
                    raise Exception("Filetypes {0} for mapkey {1} are not available!".format(filetypes, mapkey))
                

        #Create mappables out of filestruct and fieldmaps
        for i in range(len(self.filestruct[self.filetypes[0]])):
            mappable = {}
            for ft in usedfiletypes:
                mappable[self.filestruct[ft][i]] = fieldmap[ft]
            mappables.append(mappable)

        return mappables
        
    def map(self, mappable):
        """
        Do some operations on a mappable unit of the catalog
        """

        mapunit = self.readMappable(mappable)
        for m in self.metrics:
            m(mapunit)
        
    def reduce(self):
        """
        Reduce the information produced by the map operations
        """

    def readMappable(self, mappable):
        """
        For each element in the mappable, read in the fi
        specified in the fields dictionary and return
        a map from field names that the map function recognizes
        to the relevant data
        """

        mapunit = {}

        for f in mappable.keys():

            fieldmap = mappable[f]
            print(fieldmap.values())
            fields = []
            for val in fieldmap.values():
                fields.extend([val])

            print(fields)
            data = fitsio.read(f, columns=fields)
            for mapkey in fieldmap.keys():
                mapunit[mapkey] = data[fieldmap[mapkey]]
                if hasattr(fieldmap[mapkey], '__iter__'):
                    dt = mapunit[mapkey].dtype[0].subdtype[0]
                    ne = len(mapunit[mapkey])
                    nf = len(fieldmap[mapkey])
                    mapunit[mapkey] = mapunit[mapkey].view(dt).reshape((ne,nf))

        return mapunit

