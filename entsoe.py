from __future__ import absolute_import, print_function

import os, shutil
import urllib2
import pandas as pd

from . import config
from .decorators import cachable

assert 'entsoeftp' in config, '''
    Please get host, username and password from the entsoe documentation at

      https://entsoe.zendesk.com/hc/en-us/articles/115000173266-Overview-of-data-download-options-on-Transparency-Platform

    and add it to your ~/.vresutils.config file like:

      entsoeftp = 'ftp://<username>:<password>@<host>/export/export/'
    '''

csv_options = {'encoding': 'utf-16le',
               'sep': '\t'}

def urlopen(path):
    url = os.path.join(config['entsoeftp'], path)
    req = urllib2.Request(url)
    return urllib2.urlopen(req)

def listfiles(path='.', omit_dirname=False):
    fns = pd.Series(urlopen(path).readlines()).str.strip().str.split(' +').str[8]
    if not omit_dirname:
        fns = path + '/' + fns
    return fns

def downloadfile(path, dir='.'):
    localfn = os.path.join(dir, os.path.basename(path))
    if not os.path.exists(localfn):
        if not os.path.exists(os.path.dirname(localfn)):
            os.makedirs(os.path.dirname(localfn))
        response = urlopen(path)
        with open(localfn, 'wb') as fout:
            shutil.copyfileobj(response, fout)
    return localfn

@cachable
def readfile(path):
    response = urlopen(path)
    return pd.read_csv(response, **csv_options)
