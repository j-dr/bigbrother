from __future__ import print_function, division
from collections import OrderedDict
from glob import glob
import numpy as np
import yaml

from .ministry import Ministry
from .galaxy   import GalaxyCatalog
from .halo     import HaloCatalog
from .particle import ParticleCatalog
import bigbrother.magnitudemetric as mam
import bigbrother.massmetric      as msm
import bigbrother.corrmetric      as crm
import bigbrother.lineofsight     as lsm
import bigbrother.healpix_utils   as hpm
import bigbrother.densitymetric   as dnm
import bigbrother.velocity        as vlm
import bigbrother.shapemetric     as spm

_eval_keys = ['zbins', 'magbins', 'lumbins', 'cbins', 'mbins', 'abins', 'magcuts', 'massbins', 'magbins','wszbins', 'rbins', 'mbins1', 'zbins1']

def readCfg(filename):

    with open(filename, 'r') as fp:
        cfg = yaml.load(fp)

    return cfg

def parseFileStruct(cfs):

    if cfs is None:
        return None

    fs = {}

    for key in cfs.keys():
        files = glob(cfs[key])
        fs[key] = np.array(files)

    return fs

def parseFieldMap(cfm):

    if cfm is None:
        return None

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

        if 'boxsize' not in mcfg.keys():
            mcfg['boxsize'] = None

        try:
            cmetrics = mcfg.pop('metrics', None)
            om = mcfg.pop('omegam', None)
            ol = mcfg.pop('omegal', None)
            h  = mcfg.pop('h', None)
            minz = mcfg.pop('minz', None)
            maxz = mcfg.pop('maxz', None)

            mstry = Ministry(om, ol, h, minz, maxz, **mcfg)

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

    if 'ParticleCatalog' in cfg.keys():
        pcfg = cfg['ParticleCatalog']
        pct  = pcfg.pop('catalog_type', None)
        fs   = parseFileStruct(pcfg.pop('filestruct', None))

        if 'fieldmap' in pcfg.keys():
            fm  = parseFieldMap(pcfg.pop('fieldmap', None))
        else:
            fm  = None

        for key in pcfg.keys():
            if key in _eval_keys:
                pcfg[key] = eval(pcfg[key])

        if pct in Ministry._known_particle_catalog_types:
            mstry.setParticleCatalog(pct, fs, fieldmap=fm, **pcfg)
        else:
            if fm is None:
                raise(ValueError("Must supply fieldmap for generic particle catalog"))

            pc = ParticleCatalog(mstry, fs, fieldmap=fm, **pcfg)
            mstry.particlecatalog = pc

    if cmetrics is not None:
        metrics = []

        for m in cmetrics:
            if hasattr(mam, m):
                mtr = getattr(mam, m)
            elif hasattr(msm, m):
                mtr = getattr(msm, m)
            elif hasattr(crm, m):
                mtr = getattr(crm, m)
            elif hasattr(lsm, m):
                mtr = getattr(lsm, m)
            elif hasattr(hpm, m):
                mtr = getattr(hpm, m)
            elif hasattr(dnm, m):
                mtr = getattr(dnm, m)
            elif hasattr(vlm, m):
                mtr = getattr(vlm, m)
            elif hasattr(spm, m):
                mtr = getattr(spm, m)

            else:
                raise(AttributeError('No metric {}'.format(m)))

            for k in cmetrics[m]:
                if cmetrics[m][k] == 'None':
                    cmetrics[m][k] = None

                if k in _eval_keys:
                    try:
                        if cmetrics[m][k] is None:
                            continue
                        else:
                            cmetrics[m][k] = eval(cmetrics[m][k])
                    except TypeError as e:
                        print('Config parsing failed on key {} of {}'.format(k, m))
                        raise(e)


            mtr = mtr(mstry, **cmetrics[m])
            metrics.append(mtr)

        mstry.metrics = metrics
    else:

        if mstry.galaxycatalog is not None:
            mstry.metrics = mstry.galaxycatalog.metrics
        if mstry.halocatalog is not None:
            mstry.metrics.extend(mstry.halocatalog.metrics)
        if mstry.particlecatalog is not None:
            mstry.metrics.extend(mstry.particlecatalog.metrics)

    return mstry
