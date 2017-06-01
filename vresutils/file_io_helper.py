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


import os, shutil

def ensure_mkdir(path):
    '''Conflict free mkdir.'''
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def copy_without_overwrite(src, dest, quiet=True):
    # Open the file and dont do anything if it exists
    try:
        fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError:
        if os.path.isfile(dest) and quiet:
            return
        else: raise

    # Copy the file and automatically close files at the end
    with os.fdopen(fd,'w') as f:
        with open(src) as sf:
            shutil.copyfileobj(sf, f)
    shutil.copymode(src, dest)
