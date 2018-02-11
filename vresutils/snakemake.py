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

import os
import yaml
from six import iteritems

from . import Dict

class MockSnakemake(object):
    def __init__(self, path='..', wildcards={}, **kwargs):
        self.path = path
        self.wildcards = Dict(wildcards)
        self.config = kwargs.get('config')
        if self.config is None:
            with open(os.path.join(self.path, 'config.yaml')) as f:
                self.config = yaml.load(f)

        for k, v in iteritems(kwargs):
            setattr(self, k, self.expand(v))

    def expand(self, data):
        if isinstance(data, dict):
            return Dict((k, os.path.join(self.path, v.format(**self.wildcards)))
                        for k, v in iteritems(data))
        else:
            return [os.path.join(self.path, p.format(**self.wildcards)) for p in data]
