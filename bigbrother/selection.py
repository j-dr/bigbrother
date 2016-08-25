from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from itertools import product
import numpy as np

class Selector:
    """
    Handles all selections for metrics. This includes creating
    arrays to store data created from map functions, creating
    generators of selections, and handling making plots of them.
    """

    def __init__(self, selection_dict):
        """
        Instantiate selection object.

        inputs
        --------
        selection_dict - dictionary
        Dictionary whose keys are types of selections, values are diction
        """

        if selection_dict is None:
            self.selection_dict = OrderedDict([])
        else:
            self.selection_dict = selection_dict

        self.selections = self.parseSelectionDict()

    def parseSelectionDict(self):
        """
        Create a dictionary containing functions that make
        indices for each selection
        """
        selections = {}

        for label in self.selection_dict:
            selections[label] = []
            sel = self.selection_dict[label]

            if self.selection_dict[label]['selection_type'] == 'binned1d':
                selections[label].extend(self.binned1dSelection(sel))
            elif self.selection_dict[label]['selection_type'] =='cut1d':
                selections[label].extend(self.cut1dSelection(sel))
            elif self.selection_dict[label]['selection_type'] == 'cut2D':
                selections[label].extend(self.cut2DSelection(sel))

        return selections

    def binned1dSelection(self, selection):
        """
        Create a list of functions which create
        boolean indices for a 1-dimensional binned selection

        inputs
        ------
        selection -- dict
        A dictionary specifying the selection information. Needs
        to have keys: bins, mapkeys, selection_ind.
        """
        sfunctions = []

        #Should only be one mapkey for a 1d selection
        mk = selection['mapkeys'][0]

        #Iterate over bins, creating functions that make indices
        for i in range(len(selection['bins'])-1):
            if selection['selection_ind'] is None:
                sf = lambda data : (selection['bins'][i]<=data[mk]) & (selection['bins'][i+1])<data[mk]
            else:
                si = selection['selection_ind']
                sf = lambda data : (selection['bins'][i]<=data[mk][:,si]) & (selection['bins'][i+1])<data[mk][:,si]

            sfunctions.append(sf)

        return sfunctions

    def cut1dSelection(self, selection):
        """
        Create a list of functions which create
        boolean indices for a 1-dimensional cut selection

        inputs
        ------
        selection -- dict
        A dictionary specifying the selection information. Needs
        to have keys: bins, mapkeys, lower, selection_ind.

        returns
        --------
        A list of functions of length given by the number of
        bins. These functions should do the following:
        - Take in an array of data as their only argument
        - Return an array of length of the input data, containing
          True or False in each element depending on whether or not
          the corresponding element of the input array satisfied
          the selection criteria.


        """
        sfunctions = []

        #Should only be one mapkey for a 1d selection
        mk = selection['mapkeys'][0]

        #Iterate over bins, creating functions that make indices
        for i in range(len(selection['bins'])):
            if selection['lower']:
                if selection['selection_ind'] is None:
                    sf = lambda data : selection['bins'][i]<=data[mk]
                else:
                    si = selection['selection_ind']
                    sf = lambda data : selection['bins'][i]<=data[mk][:,si]
            else:
                if selection['selection_ind'] is None:
                    sf = lambda data : selection['bins'][i]>=data[mk]
                else:
                    si = selection['selection_ind']
                    sf = lambda data : selection['bins'][i]>=data[mk][:,si]

            sfunctions.append(sf)

        return sfunctions

    def cut2DSelection(self, selection):
        """
        Create a list of functions which create boolean indices
        for a 2-dimensional cut for various types of galaxy samples.

        inputs
        ------
        selection - dictionary specifying selection data.
        Needs the keys: selection_type, mapkeys, selection_ind, slopes, intercepts, lower.

        -mapkeys is a list of lists of the keywords for the data types used.
        -selection_ind gives the specific column of data from the data file.
        -slopes is a list of the slopes, where each list contains two slopes.
        -intercepts is a list of the intercepts, where each list contains two intercepts.
        -lower is a list of booleans, with each boolean telling which side of the line to select.

        returns
        -------
        One arrays of functions with size given by the length of the input data.
        Each element contains True or False depending on whether or not the
        corresponding input element fulfills the selection criteria.
        """
        #Create the sfunctions to contain the index-generating functions.
        sfunctions = []

        #Iterate over the bins and make an array of functions that create indices.
        for i in range (len(selection['slopes'])):
            sf = lambda mapunit : self.cut2DHelper(mapunit, selection, i)
            sfunctions.append(sf)

        return sfunctions

    def cut2DHelper(self, mapunit, selection, i):
        #Make the fields.
        field1 = None
        field2 = None
        if len(selection['mapkeys']) > 1:
            field1 = mapunit[selection['mapkeys'][0][0]][:,selection['selection_ind'][0][0]]-mapunit[selection['mapkeys'][1][0]][:,selection['selection_ind'][1][0]]
            field2 = mapunit[selection['mapkeys'][0][1]][:,selection['selection_ind'][0][1]]-mapunit[selection['mapkeys'][1][1]][:,selection['selection_ind'][1][1]]
        else:
            #No selection index for a 1D array.
            field1 = selection['mapkeys'][0][0]-selection['mapkeys'][1][0]
            field2 = selection['mapkeys'][0][1]-selection['mapkeys'][1][1]

        #Use the fields to make the ith cut.
        if selection['lower'][0]:
            if selection['lower'][1]:
                sf = False
            else:
                sf = ((selection['intercepts'][i][0] + (selection['slopes'][i][0])*field1) >= field2) & ((selection['intercepts'][i][1] + (selection['slopes'][i][1])*field1) >= field1)
        else:
            if selection['lower'][1]:
                sf = []
                sf_other = ((selection['intercepts'][i][0] + (selection['slopes'][i][0])*field1) >= field2) & ((selection['intercepts'][i][1] + (selection['slopes'][i][1])*field1) >= field1)
                for i in range(len(sf_other)):
                    if sf_other[i] == True:
                        sf[i] = False
                    else:
                        sf[i] = True
            else:
                sf = False

        #Return the array specifying the cuts.
        return sf

    def mapArray(self):
        """
        Given a selection dictionary return an array
        to store map outputs

        returns
        --------
        maparray - np.array
        An n-dimensional array with shape given by the
        """

        pass

    def select(self, mapunit, sfunctions):
        """
        Given selection functions and a mapunit, generate the indices
        for the selection, only updating the individual indices when
        necessary

        inputs
        -------
        mapunit -- mapunit
          A dictionary of data to generate the indices for
        sfunctions -- list
          A list of functions which generate indices for each type of selection

        returns
        -------
        An index for one selection

        """
        for i, sf in enumerate(sfunctions):
            if self.scount[i]%self.sshape[i]==0:
                self.idxarray[:,i] = sf(mapunit)
                self.scount[i] = 0
            else:
                self.scount[i] += 1

        return self.idxarray.all(axis=1)

    def generateSelections(self, mapunit):
        """
        Given a selection dictionary return a generator which
        yields indices.

        inputs
        ------
        mapunit -- mapunit
        The data to generate the selection indices for
        """
        self.sshape = np.array([len(self.selections[k]) for k in self.selections.keys()])
        self.scount = np.array([len(self.selections[k]) for k in self.selections.keys()])
        self.idxarray = np.zeros((len(mapunit[mapunit.keys()[0]]), len(self.sshape)), dtype=bool)

        iselection = self.selections.values()
        #use itertools to make loop from input selection datatype

        for sel in product(*iselection):
            yield self.select(mapunit, sel), self.scount

    def selectionAxes(self):
        """
        Make an appropriately shaped set of axes. Might need to
        rethink since shapes of axes dictated by more than just
        selections.
        """
        pass

    def selectionIndex(self):
        """
        Returns the indices of a particular selection in either the
        map array, or the axes upon which the data are plotted.
        """
        pass
