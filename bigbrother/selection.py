from __future__ import print_function, division
from abc import ABCMeta, abstractmethod

class Selection:

    def __init__(self, label, selection_type, **kwargs):
        pass

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
            self.selection_dict = {}
        else:
            self.selection_dict = selection_dict

        self.selections = self.parseSelectionDict()

    def parseSelectionDict(self):

        selections = {}

        for label in self.selection_dict:
            selections[label] = []
            sel = self.selection_dict[label]

            if selection_dict[label].selection_type == 'binned1d':
                selections[label].extend(self.binnedSelection(sel))
            elif selection_dict[label].selection_type =='cut1d':
                selections[label].extend(self.cutSelection(sel))

    def binned1dSelection(self, selection):

        sfunctions = []

        for i in range(len(selection.bins)-1):
            sf = lambda skey : (selection.bins[i]<=skey) & (selection.bins[i+1])<skey
            sfunctions.append(sf)

        return sfunctions

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

    def generateSelectionFunctions(self):

        for

    def generateSelections(self, data):
        """
        Given a selection dictionary return a generator which
        yields
        """

        #use itertools to make loop from input selection datatype
        indices = []

        for sel in selections:

            for c in cuts:
                yield data[t]<c
                #indices.append(data[t]<c)

        return indices


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
