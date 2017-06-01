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

import gurobipy as gb
import numpy as np
import scipy as sp
import scipy.sparse
import collections
from itertools import count, starmap, repeat
from six import iterkeys
from six.moves import map, range, zip

from . import iterable

def asLists(N, *arrs):
    return [asList(N, a) for a in arrs]

def asList(N, a):
    if isinstance(a, (list, tuple, np.ndarray)):
        assert len(a) == N
        return a
    elif iterable(a):
        a = list(a)
        assert len(a) == N
        return a
    else:
        return N * [a]

def asIterables(N, *arrs):
    return [asIterable(N, a) for a in arrs]

def asIterable(N, a):
    return a if iterable(a) else repeat(a, N)

class GbVec(object):
    def __init__(self, model, items):
        self.model = model
        self.items = items

    def remove(self):
        for v in self.items:
            self.model.remove(v)

    def __getitem__(self, key):
        try:
            item = self.items[key]
        except TypeError:
            # key is something more difficult than a normal slice so
            # we take the time to convert the self.items list to numpy
            self.items = np.asarray(self.items)
            item = self.items[key]

        if not isinstance(item, (list, np.ndarray)):
            return item
        else:
            view = self.__class__.__new__(self.__class__)
            GbVec.__init__(view, self.model, item)
            return view

    def __getattr__(self, attr):
        try:
            return self.model.getAttr(attr, self.items)
        except gb.GurobiError as e:
            raise AttributeError("Gurobi: {}".format(e))

    def __setattr__(self, attr, value):
        if attr in ('name', 'items', 'model'):
            self.__dict__[attr] = value
        else:
            self.model.setAttr(attr, self.items, asList(len(self), value))

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

class GbVecConstr(GbVec):
    def __init__(self, model, N, name, lhs, sense, rhs, update=True):
        super(GbVecConstr, self).__init__(model,
            [model.addConstr(lhs, sense, rhs, name=name + str(i))
             for i, lhs, rhs in zip(range(N), *asIterables(N, lhs, rhs))]
        )

        if update:
            model.update()

class GbVecVar(GbVec):
    def __init__(self, model, N, name, update=True, **kwargs):
        self.name = name
        assert 'vtype' not in kwargs or kwargs['vtype'] == gb.GRB.CONTINUOUS, \
            "Only vector constraints of type CONTINUOUS are supported"

        super(GbVecVar, self).__init__(model,
            [model.addVar(name=name + str(x[0]),
                          **dict(zip(iterkeys(kwargs), x[1:])))
             for x in zip(range(N), *asIterables(N, *kwargs.values()))]
        )

        if update:
            model.update()

    def copy(self, model):
        var = self.__class__.__new__(self.__class__)
        var.name = self.name
        var.items = np.array([model.getVarByName(var.name + str(i))
                              for i in range(len(self))])
        return var

    def LinExpr(self, d=1.0):
        return gb.LinExpr(asList(len(self), d), self.items
                          if isinstance(self.items, list) else self)

    def QuadExpr(self, d=1.0):
        ret = gb.QuadExpr()
        ret.addTerms(asList(len(self), d), self, self.items
                     if isinstance(self.items, list) else self)
        return ret

    def __neg__(self):
        return GbVecExpr(svals=[-1.0], svecs=[self])

    def __mul__(self, other):
        if np.isscalar(other):
            return GbVecExpr(svals=[other], svecs=[self])
        else:
            return NotImplemented
    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, GbVecVar):
            return GbVecExpr(svals=[1.0, 1.0], svecs=[self, other])
        else:
            return GbVecExpr(svals=[1.0], svecs=[self], cval=other)

    def __sub__(self, other):
        if isinstance(other, GbVecVar):
            return GbVecExpr(svals=[1.0, -1.0], svecs=[self, other])
        else:
            return GbVecExpr(svals=[1.0], svecs=[self], cval=-other)

class GbVecExpr(object):
    def __init__(self, svals=[], svecs=[], lvals=[], lvecs=[], cval=0):
        assert len(svals) == len(svecs), "svals and svecs must come in pairs"
        self.svals = svals
        self.svecs = svecs
        assert len(lvals) == len(lvecs), "lvals and lvecs must come in pairs"
        self.lvals = [sp.sparse.csr_matrix(lv) for lv in lvals]
        self.lvecs = lvecs
        self.cval = cval

        length = len(self)
        assert (all(len(i) == length for i in svecs) and
                all(i.shape[0] == length for i in lvals) and
                (not iterable(self.cval) or len(self.cval) == length)), \
            "Dimensions must match"

    def __neg__(self):
        return -1. * self

    def __mul__(self, other):
        if np.isscalar(other):
            return GbVecExpr(svals=[other*v for v in self.svals],
                             svecs=list(self.svecs),
                             lvals=[other*v for v in self.lvals],
                             lvecs=list(self.lvecs),
                             cval=other*self.cval)
        else:
            return NotImplemented
    __rmul__ = __mul__

    def __imul__(self, other):
        if np.isscalar(other):
            self.svals = [other*v for v in self.svals]
            self.lvals = [other*v for v in self.lvals]
            self.cval *= other
            return self
        else:
            # If we return NotImplemented here, then the eager logic
            # of scipy sparse matrices might kick in and do unexpected
            # stuff
            raise TypeError

    def __add__(self, other):
        cval = self.cval.copy() if iterable(self.cval) else self.cval
        expr = GbVecExpr(svals=list(self.svals), svecs=list(self.svecs),
                         lvals=list(self.lvals), lvecs=list(self.lvecs),
                         cval=cval)
        expr += other
        return expr
    __radd__ = __add__

    def __iadd__(self, other):
        if len(self) != 0 and iterable(other) and len(other) != 0:
            assert len(self) == len(other), "Dimensions must match"
        if isinstance(other, GbVecExpr):
            self.svals += other.svals
            self.svecs += other.svecs
            self.lvals += other.lvals
            self.lvecs += other.lvecs
            self.cval += other.cval
        elif isinstance(other, GbVecVar):
            self.svals.append(1.0)
            self.svecs.append(other)
        else:
            self.cval += other

        return self

    def __sub__(self, other):
        return self + (- other)

    def __rsub__(self, other):
        return (- self) + other

    def __isub__(self, other):
        return self.__iadd__(- other)

    def __len__(self):
        if len(self.svecs):
            return len(self.svecs[0])
        elif len(self.lvals):
            return self.lvals[0].shape[0]
        else:
            return 0

    def __iter__(self):
        exprs = []

        # scalar
        if len(self.svecs):
            exprs.append((gb.LinExpr(self.svals, vecs)
                          for vecs in zip(*self.svecs)))

        # matrix
        if len(self.lvecs):
            def generate_matrix_rows(val, vec):
                for i in range(val.shape[0]):
                    indptr = slice(val.indptr[i], val.indptr[i+1])
                    yield gb.LinExpr(val.data[indptr], vec[val.indices[indptr]])
            exprs += starmap(generate_matrix_rows, zip(self.lvals, self.lvecs))

        # constant
        if iterable(self.cval):
            exprs.append(self.cval)
        elif self.cval != 0:
            exprs.append(repeat(self.cval))

        return map(gb.quicksum, zip(*exprs))

def ismatrixlike(v):
    return sp.sparse.isspmatrix(v) or (isinstance(v, np.ndarray) and v.ndim==2)

def isvectorlike(v):
    return isinstance(v, list) or (isinstance(v, np.ndarray) and v.ndim==1)

def gbdot(v1, v2):
    if ismatrixlike(v1) and isinstance(v2, GbVecVar):
        assert v1.shape[1] == len(v2), "Dimensions must match v1.shape[1] == len(v2)"
        return GbVecExpr(lvals=[v1], lvecs=[v2])
    elif isinstance(v1, GbVecVar) and isinstance(v2, GbVecVar):
        assert len(v1) == len(v2)
        expr = gb.QuadExpr()
        expr.addTerms([1.0] * len(v1), v1, v2)
        return expr
    elif isvectorlike(v1) and isinstance(v2, GbVecVar):
        assert len(v1) == len(v2)
        return gb.LinExpr(v1, v2)
    elif isinstance(v1, (GbVecVar, GbVecExpr)) and isinstance(v2, (GbVecVar, GbVecExpr)):
        # a faster implementation is probably possible from within
        # GbVecExpr but this pedestrian one should be good enough
        # TODO : Needs to be tested
        assert len(v1) == len(v2)
        return gb.quicksum(n1 * n2 for n1, n2 in zip(v1, v2))
    else:
        raise NotImplementedError
