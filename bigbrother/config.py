from __future__ import print_function, division
from collections import OrderedDict
from glob import glob
import numpy as np
import yaml

from .ministry import Ministry
from .galaxy   import GalaxyCatalog
from .halo     import HaloCatalog
import bigbrother.magnitudemetric as mam
import bigbrother.massmetric      as msm
import bigbrother.corrmetric      as crm

def readCfg(filename):
    
    with open(filename, 'r') as fp:
        cfg = yaml.load(fp)

    return cfg

def parseFileStruct(cfs):
    
    fs = {}
    
    for key in cfs.keys():
        files = glob(cfs[key])
        fs[key] = np.array(files)

    return fs

def parseFieldMap(cfm):

    fm = {}

    for key in cfm:
        fields = cfm[key]
        fieldlist = []
        for f in fields:
            k = f.keys()[0]
            pair = (k, f[k])
            fieldlist.append(pair)
        
        fm[key] = OrderedDict(fieldlist)
        
    return fm

def parseConfig(cfg):
    
    if 'Ministry' in cfg.keys():
        mcfg = cfg['Ministry']
        try:
            mstry = Ministry(mcfg['omegam'], mcfg['omegal'], mcfg['h'],
                             mcfg['minz'], mcfg['maxz'], mcfg['area'])
        except KeyError as e:
            print('One of the necessary keys for Ministry is missing')
            print(e)
    else:
        raise(ValueError('No Ministry information!'))

    if 'GalaxyCatalog' in cfg.keys():
        gcfg = cfg['GalaxyCatalog']
        
        fs   = parseFileStruct(gcfg['filestruct'])
        
        if 'fieldmap' in gcfg.keys():
            fm  = parseFieldMap(gcfg['fieldmap'])
        else:
            fm  = None

        if 'unitmap' in gcfg.keys():
            um = gcfg['unitmap']
        else:
            um = None

        if 'filters' in gcfg.keys():
            flt = gcfg['filters']
        else:
            flt = None

        if gcfg['catalog_type'] in Ministry._known_galaxy_catalog_types:
            mstry.setGalaxyCatalog(gcfg['catalog_type'], fs, fieldmap=fm,
                                   unitmap=um, filters=flt)   
        else:
            if fm is None:
                raise(ValueError("Must supply fieldmap for generic galaxy catalog"))

            gc = GalaxyCatalog(mstry, fs, fieldmap=fm)
            mstry.galaxycatalog = gc

    if 'HaloCatalog' in cfg.keys():
        hcfg = cfg['HaloCatalog']
        
        fs   = parseFileStruct(hcfg['filestruct'])
        
        if 'fieldmap' in hcfg.keys():
            fm  = parseFieldMap(hcfg['fieldmap'])
        else:
            fm  = None

        if 'unitmap' in hcfg.keys():
            um = hcfg['unitmap']
        else:
            um = None

        if 'filters' in hcfg.keys():
            flt = hcfg['filters']
        else:
            flt = None

        if hcfg['catalog_type'] in Ministry._known_galaxy_catalog_types:
            mstry.setHaloCatalog(hcfg['catalog_type'], fs, fieldmap=fm,
                                   unitmap=um, filters=flt)   
        else:
            if fm is None:
                raise(ValueError("Must supply fieldmap for generic galaxy catalog"))

            hc = HaloCatalog(mstry, fs, fieldmap=fm, unitmap=um, filters=flt)
            mstry.halocatalog = hc

    if 'metrics' in mcfg.keys():
        metrics = []

        for m in mcfg['metrics']:
            if hasattr(mam, m):
                mtr = getattr(mam, m)
            elif hasattr(msm, m):
                mtr = getattr(msm, m)
            elif hasattr(crm, m):
                mtr = getattr(crm, m)

            if m in mcfg.keys():
                mtr = mtr(mstry, **cfg[m])
            else:
                mtr = mtr(mstry)

            metrics.append(mtr)
        
        mstry.metrics = metrics
    else:

        if mstry.galaxycatalog is not None:
            mstry.metrics = mstry.galaxycatalog.metrics
        if mstry.halocatalog is not None:
            mstry.metrics.extend(mstry.halocatalog.metrics)


    return mstry
