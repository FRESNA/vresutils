import gurobipy as gb
import numpy as np
import scipy as sp
import scipy.sparse
import collections
from itertools import izip, count, imap, starmap

def asLists(N, *arrs):
    return [asList(N, a) for a in arrs]

def asList(N, a):
    if isinstance(a, (list, tuple)):
        assert len(a) == N
        return a
    elif isinstance(a, collections.Iterable):
        a = list(a)
        assert len(a) == N
        return a
    else:
        return N * [a]

class GbVec(object):
    def __init__(self, model, items):
        self.model = model
        self.items = items

    def __getitem__(self, key):
        item = self.items[key]

        if not isinstance(item, np.ndarray):
            return item
        else:
            view = self.__class__.__new__(self.__class__)
            GbVec.__init__(view, self.model, item)
            return view

    def __getattr__(self, attr):
        try:
            return self.model.getAttr(attr, self.items)
        except gb.GurobiError:
            raise AttributeError

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
            np.array([model.addConstr(lhs, sense, rhs, name=name + str(i))
                      for lhs, rhs, i in izip(asList(N, lhs), asList(N, rhs), count())])
        )

        if update:
            model.update()

class GbVecVar(GbVec):
    def __init__(self, model, N, name, update=True, **kwargs):
        self.name = name
        assert 'vtype' not in kwargs or kwargs['vtype'] == gb.GRB.CONTINUOUS, \
            "Only vector constraints of type CONTINUOUS are supported"

        super(GbVecVar, self).__init__(model,
            np.asarray([model.addVar(name=name + str(x[0]),
                                     **dict(izip(kwargs.iterkeys(), x[1:])))
                        for x in izip(np.arange(N), *asLists(N, *kwargs.values()))])
        )

        if update:
            model.update()

    def copy(self, model):
        var = self.__class__.__new__(self.__class__)
        var.name = self.name
        var.items = np.array([model.getVarByName(var.name + str(i))
                              for i in np.arange(len(self))])
        return var

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
            return NotImplemented

class GbVecExpr(object):
    def __init__(self, svals=[], svecs=[], lvals=[], lvecs=[]):
        self.svals = svals
        self.svecs = svecs
        self.lvals = lvals
        self.lvecs = lvecs

    def __mul__(self, other):
        if np.isscalar(other):
            return GbVecExpr(svals=[other*v for v in self.svals],
                             svecs=list(self.svecs),
                             lvals=[other*v for v in self.lvals],
                             lvecs=list(self.lvecs))
        else:
            return NotImplemented
    __rmul__ = __mul__

    def __imul__(self, other):
        if np.isscalar(other):
            self.svals = [other*v for v in self.svals]
            self.lvals = [other*v for v in self.lvals]
            return self
        else:
            # If we return NotImplemented here, then the eager logic
            # of scipy sparse matrices might kick in and do unexpected
            # stuff
            raise TypeError

    def __add__(self, other):
        expr = GbVecExpr(**{k: list(v) for k,v in self.__dict__.iteritems()})
        expr += other
        return expr
    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, GbVecExpr):
            self.svals += other.svals
            self.svecs += other.svecs
            self.lvals += other.lvals
            self.lvecs += other.lvecs
        elif isinstance(other, GbVecVar):
            self.svals.append(1.0)
            self.svecs.append(other)

        return self

    def __iter__(self):
        scalarexprs = (gb.LinExpr(self.svals, vecs)
                       for vecs in izip(*self.svecs))

        def generate_matrix_rows(val, vec):
            if sp.sparse.isspmatrix(val):
                val = val.tocsr()
            else:
                val = np.asmatrix(val)
            for i in np.arange(val.shape[0]):
                row = val[i]
                elems = row.nonzero()
                yield gb.LinExpr(np.squeeze(np.asarray(row[elems])),
                                 vec[np.squeeze(np.asarray(elems[1]))])
        matrixexprs = starmap(generate_matrix_rows, izip(self.lvals, self.lvecs))

        return imap(gb.quicksum, izip(scalarexprs, *matrixexprs))

def gbdot(v1, v2):
    if (sp.sparse.isspmatrix(v1) or isinstance(v1, np.ndarray)) and isinstance(v2, GbVecVar):
        assert v1.shape[1] == len(v2), "Dimensions must match v1.shape[1] == len(v2)"
        return GbVecExpr(lvals=[v1], lvecs=[v2])
    elif isinstance(v1, GbVecVar) and isinstance(v2, GbVecVar):
        assert len(v1) == len(v2)
        expr = gb.QuadExpr()
        expr.addTerms([1.0] * len(v1), v1, v2)
        return expr
    elif (isinstance(v1, list) or isinstance(v1, np.ndarray)) and isinstance(v2, GbVecVar):
        assert len(v1) == len(v2)
        return gb.LinExpr(v1, v2)
    else:
        raise NotImplementedError
