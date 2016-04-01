from __future__ import print_function, division
from collections import OrderedDict
import yaml

def read_cfg(filename):
    
    with open(filename, 'r') as fp:
        cfg = yaml.load(fp)

    return cfg


def parse_config(cfg):
    pass
    
