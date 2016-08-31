from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from itertools import product
from functools import partial
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

        self.selections = None

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

        if not hasattr(selection['selection_ind'], '__iter__'):
            si = [selection['selection_ind']] * len(selection['bins'])
        else:
            si = selection['selection_ind']

        #Iterate over bins, creating functions that make indices
        for i in range(len(selection['bins'])-1):
            sfunctions.append(partial(self.bin1dhelper, selection['bins'][i], selection['bins'][i+1], mk, si[i]))

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
        """
        sfunctions = []

        #Should only be one mapkey for a 1d selection
        mk = selection['mapkeys'][0]
        if not hasattr(selection['selection_ind'], '__iter__'):
            si = [selection['selection_ind']] * len(selection['bins'])
        else:
            si = selection['selection_ind']

        if not hasattr(selection['lower'], '__iter__'):
            lower = [selection['lower']] * len(selection['bins'])
        else:
            lower = selection['lower']

        #Iterate over bins, creating functions that make indices
        for i in range(len(selection['bins'])):
            sfunctions.append(partial(self.cut1dhelper, selection['bins'][i], mk, si[i], lower[i]))

        return sfunctions

    def bin1dhelper(self, lbound, hbound, key, index, data):
        if index is None:
            return (lbound <= data[key]) & (data[key] < hbound)
        else:
            return (lbound <= data[key][:,index]) & (data[key][:,index] < hbound)

    def cut1dhelper(self, cut, key, index, lower, data):
        if lower:
            if index is None:
                return data[key]<=cut
            else:
                return data[key][:,index]<=cut
        else:
            if index is None:
                return data[key]>cut
            else:
                return data[key][:,index]>cut


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
            if ((self.scount[i]+1)%self.sshape[i])==0:
                self.scount[i] = 0
            else:
                self.scount[i] += 1

            self.idxarray[:,i] = sf(mapunit)

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

        if self.selections is None:
            self.selections = self.parseSelectionDict()

        self.sshape = np.array([len(self.selections[k]) for k in self.selections.keys()])
        self.scount = np.array([len(self.selections[k])-1 for k in self.selections.keys()])
        self.idxarray = np.zeros((len(mapunit[mapunit.keys()[0]]), len(self.sshape)), dtype=bool)

        iselection = self.selections.values()
        #use itertools to make loop from input selection datatype

        for sel in product(*iselection):
            yield self.select(mapunit, sel), self.scount

        self.selections = None

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
