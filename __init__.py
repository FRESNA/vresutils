# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import os.path
import six
import logging

logger = logging.getLogger(__name__)

def indicator(N, indices):
    m = np.zeros(N)
    m[indices] = 1.
    return m

def iterable(obj):
    'return true if *obj* is iterable'
    try:
        iter(obj)
    except TypeError:
        return False
    return True

def make_toDataDir(modulefilename):
    """
    Returns a function which translates relative names to a path
    starting from modulefilename.

    The idea is to start a module with
    toDataDir = make_toDataDir(__file__)

    Then a call like toDataDir('file') will return the full path
    from the directory of the module instead of the working directory.
    """
    dataDir = os.path.join(os.path.realpath(os.path.dirname(modulefilename)), 'data')
    def translate_toDataDir_and_maybe_check_for_existence(fn, check_for_existence=True):
        if not (os.path.isabs(fn) or fn[0] == '.'):
            fn = os.path.join(dataDir, fn)
        if check_for_existence and not os.path.exists(fn):
            logger.warning("""
               The data file %s was not found. The README at

                 %s

               should detail were it can be obtained from,
               alternatively there are archive data bundles for
               each repository available from

                 http://fias.uni-frankfurt.de/~hoersch/data/
            """, fn, os.path.realpath(os.path.dirname(modulefilename)))
        return fn

    return translate_toDataDir_and_maybe_check_for_existence

class Singleton(object):
    def __new__(cls, *p, **k):
        if not '_the_instance' in cls.__dict__:
            cls._the_instance = object.__new__(cls)
        return cls._the_instance

def get_config(config_fn, defaults=dict(), overwrites=dict()):
    if '/' not in config_fn:
        config_fn = os.path.join('~', config_fn)
    if '~' in config_fn:
        config_fn = os.path.expanduser(config_fn)

    config = defaults
    if os.path.exists(config_fn):
        exec(compile(open(config_fn).read(), config_fn, 'exec'), dict(), config)
    config.update(overwrites)

    return config

config = get_config(
    '.vresutils.config',
    defaults=dict(
        cache_dir="/home/vres/data/" + ("cache" if six.PY2 else "cache3"),
        fallback_cache_dirs=[] if six.PY2 else ["/home/vres/data/cache"]
    )
)

# backwards compatibility for moved functions
from .decorators import _format_filename as format_filename, cachable, optional, timer, staticvars, CachedAttribute, indexer
