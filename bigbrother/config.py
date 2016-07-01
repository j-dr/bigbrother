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
import bigbrother.healpix_utils   as hpm

_eval_keys = ['zbins']

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

def replaceNoneStr(cfg):
    for key in cfg.keys():
        if cfg[key] == 'None':
            cfg[key] = None
        elif hasattr(cfg[key], 'keys'):
            cfg[key] = replaceNoneStr(cfg[key])

    return cfg

def parseConfig(cfg):

    replaceNoneStr(cfg)

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
        gct  = gcfg.pop('catalog_type', None)
        fs   = parseFileStruct(gcfg.pop('filestruct', None))

        if 'fieldmap' in gcfg.keys():
            fm  = parseFieldMap(gcfg.pop('fieldmap', None))
        else:
            fm  = None

        for key in gcfg.keys():
            if key in _eval_keys:
                gcfg[key] = eval(gcfg[key])

        if gct in Ministry._known_galaxy_catalog_types:
            mstry.setGalaxyCatalog(gct, fs, fieldmap=fm, **gcfg)
        else:
            if fm is None:
                raise(ValueError("Must supply fieldmap for generic galaxy catalog"))

            gc = GalaxyCatalog(mstry, fs, fieldmap=fm, **gcfg)
            mstry.galaxycatalog = gc

    if 'HaloCatalog' in cfg.keys():
        hcfg = cfg['HaloCatalog']
        hct  = hcfg.pop('catalog_type', None)
        fs   = parseFileStruct(hcfg.pop('filestruct', None))

        if 'fieldmap' in hcfg.keys():
            fm  = parseFieldMap(hcfg.pop('fieldmap', None))
        else:
            fm  = None

        for key in hcfg.keys():
            if key in _eval_keys:
                hcfg[key] = eval(hcfg[key])

        if hct in Ministry._known_galaxy_catalog_types:
            mstry.setHaloCatalog(hct, fs, fieldmap=fm, **hcfg)
        else:
            if fm is None:
                raise(ValueError("Must supply fieldmap for generic galaxy catalog"))

            hc = HaloCatalog(mstry, fs, fieldmap=fm, **hcfg)
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
            elif hasattr(hpm, m):
                mtr = getattr(hpm, m)

            for k in mcfg['metrics'][m]:
                if mcfg['metrics'][m][k] == 'None':
                    mcfg['metrics'][m][k] = None

            mtr = mtr(mstry, **mcfg['metrics'][m])
            metrics.append(mtr)

        mstry.metrics = metrics
    else:

        if mstry.galaxycatalog is not None:
            mstry.metrics = mstry.galaxycatalog.metrics
        if mstry.halocatalog is not None:
            mstry.metrics.extend(mstry.halocatalog.metrics)


    return mstry
