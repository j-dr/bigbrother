from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

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

            if selection_dict[label].selection_type == 'binned1d':
                selections[label].extend(self.binnedSelection(sel))
            elif selection_dict[label].selection_type =='cut1d':
                selections[label].extend(self.cutSelection(sel))

        self.selections = selections

    def binned1dSelection(self, selection):
        """
        Given a type of selection, create a list
        of functions which create a boolean index

        inputs
        ------
        seletion -- dict
        A dictionary specifying the selection information. Needs
        to have
        """
        sfunctions = []

        #Should only be one mapkey for a 1d selection
        mk = selection.mapkeys[0]

        #Iterate over bins, creating functions that make indices
        for i in range(len(selection['bins'])-1):
            sf = lambda data : (selection['bins'][i]<=data[mk]) & (selection['bins'][i+1])<data[mk]

            sfunctions.append(sf)

        return sfunctions

    def cut1dselection(self, selection):
        """
        Same as binned1dselection, but for cut selections rather than bins
        """


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
        for sf in sfunctions:
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
        self.idxarray((len(data), len(self.sshape)), dtype=bool)

        iselection = self.selections.values()
        #use itertools to make loop from input selection datatype

        for sel in product(iselection):
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
