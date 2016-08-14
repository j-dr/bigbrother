from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from itertools import product
import numpy as np

class Selector:
    """
    Handles all selections for metrics. This includes creating
    arrays to store data created from map functions, creating
    generators of selections, and handling plotting them.
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

        return selections

    def binned1dSelection(self, selection):
        """
        Create a list of functions which create
        boolean indices for a 1-dimensional binned selection

        inputs
        ------
        seletion -- dict
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
        seletion -- dict
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
          the selection criterea.


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
        Needs the keys: mapkeys, int1, slope1, int2, slope2, lower1,
        lower2, bins1, bins2.

        returns
        -------
        Two arrays of functions, with size given by the size of the
        input data (colorx or colory). Each element contains True or False
        depending on whether or not the corresponding input element fulfills
        the selection criteria.
        """
        #Create the sfunctions to contain the index-generating functions.
        sfunctions1 = []
        sfunctions2 = []

        #Calculate the color arrays from mapkeys (three columns corresponding to magnitude data).
        #Two mapkeys for a 2D selection.
        mk1 = selection['mapkeys'][0] - selection['mapkeys'][1]
        mk2 = selection['mapkeys'][0] - selection['mapkeys'][2]

        #Iterate over the bins and make an array of functions that create indices.
        #Returns two arrays, one for each axis.
        for i in range (len(selection['bins'])):
            if selection['lower1']:
                if selection['lower2']:
                    if selection['int1'] is None or selection['slope1'] is None:
                        if selection ['int2'] is None or selection ['slope2'] is None: sf1 = lambda data : selection['bins1'][i] <= data[mk1], sf2 = lambda data : selection['bins2'][i] <= data[mk2]
                        else: si = int1 + (slope1*i), sf1 = lambda data : selection['bins1'][i] <= data[mk1][:,si], sf2 = lambda data : selection['bins2'][i] <= data[mk2]
                    else: si1 = int1 + (slope1*i), si2 = int2 + (slope2*i), sf1 = lambda data : selection['bins1'][i] <= data[mk1][:,si1], sf2 = lambda data : selection['bins2'][i] <= data[mk2][:,si2]
                else:
                    if selection['int1'] is None or selection['slope1'] is None:
                        if selection ['int2'] is None or selection ['slope2'] is None: sf1 = lambda data : selection['bins1'][i] <= data[mk1], sf2 = lambda data : selection['bins2'][i] >= data[mk2]
                        else: si = int1 + (slope1*i), sf1 = lambda data : selection['bins1'][i] <= data[mk1][:,si], sf2 = lambda data : selection['bins2'][i] >= data[mk2]
                    else: si1 = int1 + (slope1*i), si2 = int2 + (slope2*i), sf1 = lambda data : selection['bins1'][i] <= data[mk1][:,si1], sf2 = lambda data : selection['bins2'][i] >= data[mk2][:,si2]
            else:
                if selection['lower2']:
                    if selection['int1'] is None or selection['slope1'] is None:
                        if selection ['int2'] is None or selection ['slope2'] is None: sf1 = lambda data : selection['bins1'][i] >= data[mk1], sf2 = lambda data : selection['bins2'][i] <= data[mk2]
                        else: si = int1 + (slope1*i), sf1 = lambda data : selection['bins1'][i] >= data[mk1][:,si], sf2 = lambda data : selection['bins2'][i] <= data[mk2]
                    else: si1 = int1 + (slope1*i), si2 = int2 + (slope2*i), sf1 = lambda data : selection['bins1'][i] >= data[mk1][:,si1], sf2 = lambda data : selection['bins2'][i] <= data[mk2][:,si2]
                else:
                    if selection['int1'] is None or selection['slope1'] is None:
                        if selection ['int2'] is None or selection ['slope2'] is None: sf1 = lambda data : selection['bins1'][i] >= data[mk1], sf2 = lambda data : selection['bins2'][i] >= data[mk2]
                        else: si = int1 + (slope1*i), sf1 = lambda data : selection['bins1'][i] >= data[mk1][:,si], sf2 = lambda data : selection['bins2'][i] >= data[mk2]
                    else: si1 = int1 + (slope1*i), si2 = int2 + (slope2*i), sf1 = lambda data : selection['bins1'][i] >= data[mk1][:,si1], sf2 = lambda data : selection['bins2'][i] >= data[mk2][:,si2]

            sfunctions1.append(sf1)
            sfunction2.append(sf2)

        return sfunctions1, sfunctions2

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
