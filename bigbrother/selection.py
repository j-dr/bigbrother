from __future__ import print_function, division
from abc import ABCMeta, abstractmethod

class Selection:
    """
    Base class for selection classes.
    """
    def __init__(self, selection_dict):

        self.mapkeys = selection_dict.keys()
        self.selection_dict = selection_dict

    def select(self, data, selection):
        pass


    def applySelection(self, mapunit):

        mu = {}

        for key in mapunit.keys():
            if key in self.mapkeys:
                mu[key] = self.select(mu[key], self.selection_dict[key])
