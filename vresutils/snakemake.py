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
    def __init__(self, wildcards={}, path='..', config=None, **kwargs):
        self.wildcards = Dict(wildcards)
        self.path = path
        self.config = config

        config_fn = os.path.join(self.path, 'config.yaml')
        if self.config is None and os.path.exists(config_fn):
            with open(config_fn) as f:
                self.config = yaml.safe_load(f)

        for k, v in iteritems(kwargs):
            setattr(self, k, self.expand(v))

    def expand(self, data):
        def expand_path(n):
            return os.path.join(self.path, n.format(**self.wildcards))

        if isinstance(data, dict):
            return Dict((k, expand_path(v)) for k, v in iteritems(data))
        else:
            return [expand_path(n) for n in data]
