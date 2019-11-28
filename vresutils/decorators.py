# -*- coding: utf-8 -*-

## Copyright 2015-2017 Frankfurt Institute for Advanced Studies

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
"""

from __future__ import absolute_import
from __future__ import print_function

from hashlib import sha1
from functools import wraps
from six.moves import cPickle
import weakref
import sys, os, stat, string
from six import iteritems
import six

from . import config
from .benchmark import timer, optional

import logging
logger = logging.getLogger(__name__)

def _format_filename(s):
    """
    Take a string and return a valid filename constructed from the string.
    Uses a whitelist approach: any characters not present in valid_chars are
    removed. Also spaces are replaced with underscores.
    """
    valid_chars = frozenset("-_.() %s%s" % (string.ascii_letters, string.digits))
    return ''.join(c for c in s if c in valid_chars).replace(' ','_')


def cachable(func=None, version=None, cache_dir=config['cache_dir'],
             fallback_cache_dirs=config['fallback_cache_dirs'],
             keepweakref=False, ignore=set(), verbose=True):
    """
    Decorator to mark long running functions, which should be saved to
    disk for a pickled version of their arguments.

    Arguments:
    func        - Shouldn't be supplied, but instead contains the
                  function, if the decorator is used without arguments
    version     - if given it is saved together with the arguments and
                  must be the same as the cache to be valid.
    cache_dir   - where to save the cached files, if it does not exist
                  it is created (default defined by ~/.vresutils.config)
    keepweakref - Also store a weak reference and return it instead of
                  rereading from disk (default: False).
    ignore      - Set of kwd arguments not to take into account.
    verbose     - Output cache hits and timing information (default:
                  True).
    """
    enable = os.path.isdir(cache_dir)

    if enable:
        st = os.stat(cache_dir)
        gid = st.st_gid
        # mode is bitmask of the same rights as the directory without
        # exec rights for anybody
        mode = stat.S_IMODE(st.st_mode) & ~(stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # remove missing directories
    fallback_cache_dirs = [d for d in fallback_cache_dirs if os.path.isdir(d)]

    def deco(func):
        """
        Wrap function to check for a cached result. Everything is
        pickled to a file of the name
        cache_dir/<funcname>_param1_param2_kw1key.kw1val
        (it would be better to use np.save/np.load for numpy arrays)
        """

        cache_fn = func.__module__ + "." + func.__name__ + "_"
        if version is not None:
            cache_fn += _format_filename("ver" + str(version) + "_")

        if keepweakref:
            cache = weakref.WeakValueDictionary()

        def name(x):
            y = six.text_type(x)
            if len(y) > 40:
                y = sha1(y.encode('utf-8')).hexdigest()
            return y

        def load_from(fn, dn, try_latin=False):
            full_fn = os.path.join(dn, fn)
            if os.path.exists(full_fn):
                try:
                    with open(full_fn, 'rb') as f:
                        dn_label = os.path.basename(dn)
                        if try_latin:
                            dn_label += " (with forced encoding)"
                        with optional(
                                verbose,
                                timer("Serving call to {} from file {} of {}"
                                      .format(func.__name__, fn, dn_label))
                        ):
                            if try_latin:
                                return cPickle.load(f, encoding='latin-1')
                            else:
                                return cPickle.load(f)
                except Exception as e:
                    if not try_latin and isinstance(e, UnicodeDecodeError):
                        return load_from(fn, dn, try_latin=True)
                    print("Couldn't unpickle from %s: %s" % (fn, e.args[0]), file=sys.stderr)

        @wraps(func)
        def wrapper(*args, **kwds):
            recompute = kwds.pop('recompute', False)

            # Check if there is a cached version
            fn = cache_fn + _format_filename(
                '_'.join(name(a) for a in args) + '_' +
                '_'.join(name(k) + '.' + name(v)
                         for k,v in iteritems(kwds)
                         if k not in ignore) +
                '.pickle'
            )

            ret = None
            if not recompute:
                if keepweakref and fn in cache:
                    return cache[fn]

                ret = load_from(fn, cache_dir)

            if ret is None:
                if not recompute:
                    for fallback in fallback_cache_dirs:
                        ret = load_from(fn, fallback)
                        if ret is not None: break

                if ret is None:
                    with optional(
                            verbose,
                            timer("Caching call to {} in {}".format(func.__name__, fn))
                    ):
                        ret = func(*args, **kwds)
                try:
                    with open(os.path.join(cache_dir, fn), 'wb') as f:
                        if gid: os.fchown(f.fileno(), -1, gid)
                        if mode: os.fchmod(f.fileno(), mode)
                        cPickle.dump(ret, f, protocol=-1)
                except Exception as e:
                    print("Couldn't pickle to %s: %s" % (fn, e.args[0]), file=sys.stderr)

            if keepweakref and ret is not None:
                cache[fn] = ret

            return ret

        return wrapper

    if not enable:
        def deco(func):
            # logger.warn("Deactivating cache for function %s, since cache directory %s does not exist",
            #             func.__name__, cache_dir)
            return func

    if callable(func):
        return deco(func)
    else:
        return deco

def staticvars(**vars):
    def deco(fn):
        fn.__dict__.update(vars)
        return fn
    return deco

class CachedAttribute(object):
    '''
    Computes attribute value and caches it in the instance.
    From the Python Cookbook (Denis Otkidach)
    This decorator allows you to create a property which can be
    computed once and accessed many times. Sort of like memoization.
    '''
    def __init__(self, method, name=None, doc=None):
        # record the unbound-method and the name
        self.method = method
        self.name = name or method.__name__
        self.__doc__ = doc or method.__doc__
    def __get__(self, inst, cls):
        if inst is None:
            # instance attribute accessed on class, return self
            # You get here if you write `Foo.bar`
            return self
        # compute, cache and return the instance's attribute value
        result = self.method(inst)
        # setattr redefines the instance's attribute so this doesn't get called again
        setattr(inst, self.name, result)
        return result

def indexer(func):
    '''
    Decorator to turn a method into an object with a __getitem__ method.
    Can be used to turn a method into some extended getitem equivalent.
    '''
    class Indexer(object):
        def __init__(self, inst):
            self.inst = inst
        def __getitem__(self, key):
            return func(self.inst, key)
    return CachedAttribute(lambda self: Indexer(self), name=func.__name__, doc=func.__doc__)
