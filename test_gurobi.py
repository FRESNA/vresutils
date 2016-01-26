from __future__ import absolute_import

import unittest

import numpy as np, scipy as sp
import scipy.sparse
from numpy.testing import assert_allclose

from .gurobi import GbVec, GbVecConstr, GbVecVar, GbVecExpr, gbdot
from .array import spdiag

import gurobipy as gb

class GbVecVarTest(unittest.TestCase):
    def setUp(self):
        self.model = gb.Model()
        self.v = GbVecVar(self.model, 2, name='v', ub=3.0, lb=(1.0, 2.0))
        self.model.update()

        self.v2 = self.model.getVars()

    def test_init(self):
        self.assertIsInstance(self.v, GbVecVar)

        v = self.v
        v2 = self.v2

        self.assertEqual(len(v2), 2)
        self.assertTrue(v.items[0].sameAs(v2[0]))
        self.assertTrue(v.items[1].sameAs(v2[1]))

        self.assertEqual(v2[0].lb, 1.0)
        self.assertEqual(v2[1].lb, 2.0)

        self.assertEqual(v2[0].ub, 3.0)
        self.assertEqual(v2[1].ub, 3.0)

    def test_setattr(self):
        self.v.ub = 2.5
        self.model.update()
        self.assertEqual(self.v2[0].ub, 2.5)
        self.assertEqual(self.v2[1].ub, 2.5)

        self.v.lb = (.0, .5)
        self.model.update()
        self.assertEqual(self.v2[0].lb, .0)
        self.assertEqual(self.v2[1].lb, .5)

    def test_getitem(self):
        # item index
        v0 = self.v[0]
        self.assertIsInstance(v0, gb.Var)
        self.assertTrue(v0.sameAs(self.v2[0]))

        # slices
        v1s = self.v[1:]
        self.assertIsInstance(v1s, GbVecVar)
        self.assertTrue(v1s.items[0].sameAs(self.v2[1]))

        # special expression
        vs = self.v[np.array((False, True))]
        self.assertEqual(len(vs), 1)
        self.assertTrue(vs[0].sameAs(self.v2[1]))

    def test_getattr(self):
        self.assertEqual(self.v.lb, [1.0, 2.0])

    def test_mul(self):
        expr = 7.0 * self.v
        self.assertIsInstance(expr, GbVecExpr)
        self.assertEqual(expr.svals, [7.0])
        self.assertEqual(expr.svecs, [self.v])

        K = sp.sparse.diags(((1.0, 1.0),), (0,))
        expr = gbdot(K, self.v)
        self.assertIsInstance(expr, GbVecExpr)
        self.assertEqual(len(expr.lvals), 1)
        self.assertEqual((expr.lvals[0] - K).nnz, 0)
        self.assertEqual(expr.lvecs, [self.v])

    def test_add(self):
        v2 = GbVecVar(self.model, 2, name='B')
        expr = self.v + v2
        self.assertIsInstance(expr, GbVecExpr)
        self.assertEqual(expr.svals, [1.0, 1.0])
        self.assertEqual(expr.svecs, [self.v, v2])

    def test_linexpr(self):
        self.assertEqual(str(self.v.LinExpr()), '<gurobi.LinExpr: v0 + v1>')
        self.assertEqual(str(self.v.LinExpr(np.array((1.0,2.0)))),
                         '<gurobi.LinExpr: v0 + 2.0 v1>')

    def test_quadexpr(self):
        self.assertEqual(str(self.v.QuadExpr(np.array((1.0,2.0)))),
                        '<gurobi.QuadExpr: 0.0 + [ v0 ^ 2 + 2.0 v1 ^ 2 ]>')


class GbVecExprTest(unittest.TestCase):
    def setUp(self):
        self.model = gb.Model()
        self.v1 = GbVecVar(self.model, 2, name='v1_')
        self.v2 = GbVecVar(self.model, 2, name='v2_')

        self.K = sp.sparse.diags([(1, 1)], [0]).tocsr()

        self.model.update()
        self.vs = self.model.getVars()

        self.ex = GbVecExpr(svals=[2.0], svecs=[self.v1], lvals=[self.K], lvecs=[self.v2])

    def test_iter(self):
        exprs = list(self.ex)
        self.assertEqual(len(exprs), 2)
        self.assertEqual(str(exprs[0]), '<gurobi.LinExpr: 2.0 v1_0 + v2_0>')
        self.assertEqual(str(exprs[1]), '<gurobi.LinExpr: 2.0 v1_1 + v2_1>')

    def test_mul(self):
        # scalar multiplication
        ex = 3.0 * self.ex
        self.assertIsInstance(ex, GbVecExpr)
        self.assertEqual(ex.svals, [6.0])
        self.assertEqual(ex.svecs, [self.v1])
        self.assertEqual(len(ex.lvals), 1)
        assert_allclose(ex.lvals[0].todense(), 3.0*self.K.todense())
        self.assertEqual(ex.lvecs, [self.v2])

        # scalar multiplication
        ex = self.ex * 4.0
        self.assertIsInstance(ex, GbVecExpr)
        self.assertEqual(ex.svals, [8.0])
        self.assertEqual(ex.svecs, [self.v1])
        self.assertEqual(len(ex.lvals), 1)
        assert_allclose(ex.lvals[0].todense(), 4.0*self.K.todense())
        self.assertEqual(ex.lvecs, [self.v2])

        # matrix multiplication throws TypeError
        self.assertRaises(TypeError, lambda: self.K * self.ex)

    def test_imul(self):
        self.ex *= 4.0
        self.assertIsInstance(self.ex, GbVecExpr)
        self.assertEqual(self.ex.svals, [8.0])
        self.assertEqual(self.ex.svecs, [self.v1])
        self.assertEqual(len(self.ex.lvals), 1)
        assert_allclose(self.ex.lvals[0].todense(), 4.0*self.K.todense())
        self.assertEqual(self.ex.lvecs, [self.v2])

        def do_matmul():
            self.ex *= self.K
        self.assertRaises(TypeError, do_matmul)

    def test_add(self):
        ex = self.ex + self.v2
        self.assertIsInstance(ex, GbVecExpr)
        self.assertEqual(ex.svals, [2.0, 1.0])
        self.assertEqual(ex.svecs, [self.v1, self.v2])
        self.assertEqual(len(ex.lvals), 1)
        self.assertEqual((ex.lvals[0] - self.K).nnz, 0)
        self.assertEqual(ex.lvecs, [self.v2])

        ex = self.ex - self.v2
        self.assertIsInstance(ex, GbVecExpr)
        self.assertEqual(ex.svals, [2.0, -1.0])
        self.assertEqual(ex.svecs, [self.v1, self.v2])
        self.assertEqual(len(ex.lvals), 1)
        self.assertEqual((ex.lvals[0] - self.K).nnz, 0)
        self.assertEqual(ex.lvecs, [self.v2])

        ex = self.ex + 3.0 * self.v2
        self.assertIsInstance(ex, GbVecExpr)
        self.assertEqual(ex.svals, [2.0, 3.0])
        self.assertEqual(ex.svecs, [self.v1, self.v2])
        self.assertEqual(len(ex.lvals), 1)
        self.assertEqual((ex.lvals[0] - self.K).nnz, 0)
        self.assertEqual(ex.lvecs, [self.v2])

        self.ex += 3.0 * self.v2
        self.assertIsInstance(self.ex, GbVecExpr)
        self.assertEqual(self.ex.svals, [2.0, 3.0])
        self.assertEqual(self.ex.svecs, [self.v1, self.v2])
        self.assertEqual(len(ex.lvals), 1)
        self.assertEqual((ex.lvals[0] - self.K).nnz, 0)
        self.assertEqual(self.ex.lvecs, [self.v2])

    def test_only_lvals(self):
        # bug reported by sarah on 29/05/15
        ex = GbVecExpr(lvals=[spdiag(np.ones(len(self.v1)))], lvecs=[self.v1])
        self.assertEqual(len(list(ex)), len(self.v1))

class GbDotTest(unittest.TestCase):
    def setUp(self):
        self.model = gb.Model()
        self.v1 = GbVecVar(self.model, 2, name='v1_')
        self.v2 = GbVecVar(self.model, 2, name='v2_')
        self.v3 = GbVecVar(self.model, 3, name='v3_')

        self.K = sp.sparse.diags([(1, 1)], [0])

        self.model.update()
        self.vs = self.model.getVars()

    def test_vecvec(self):
        ex = gbdot(self.v1, self.v2)
        self.assertEqual(str(ex), '<gurobi.QuadExpr: 0.0 + [ v1_0 * v2_0 + v1_1 * v2_1 ]>')

        # dimension mismatch
        self.assertRaises(AssertionError, lambda: gbdot(self.v1, self.v3))

    def test_matvec(self):
        ex = gbdot(self.K, self.v1)
        self.assertIsInstance(ex, GbVecExpr)
        self.assertEqual(len(ex.lvals), 1)
        self.assertEqual((ex.lvals[0] - self.K).nnz, 0)
        self.assertEqual(ex.lvecs, [self.v1])

        # dimension mismatch
        self.assertRaises(AssertionError, lambda: gbdot(self.K, self.v3))

    def test_listvec(self):
        ex = gbdot([1.0, 2.0], self.v1)

        self.assertIsInstance(ex, gb.LinExpr)
        self.assertEqual(str(ex), '<gurobi.LinExpr: v1_0 + 2.0 v1_1>')


class GbVecConstrTest(unittest.TestCase):
    def setUp(self):
        self.model = gb.Model()
        self.v1 = GbVecVar(self.model, 2, name='v1_')
        self.v2 = GbVecVar(self.model, 2, name='v2_')
        self.K = sp.sparse.diags([(1, 1)], [0])

        self.ex = GbVecExpr(svals=[2.0], svecs=[self.v1], lvals=[self.K], lvecs=[self.v2])

    def test_init(self):
        lhsconstr = GbVecConstr(self.model, 2, "lhsconstr", self.ex, '>', 0)
        self.assertIsInstance(lhsconstr, GbVecConstr)
        self.assertEqual(len(lhsconstr), 2)

        self.assertEqual(str(lhsconstr[0]), '<gurobi.Constr lhsconstr0>')
        self.assertEqual(lhsconstr[0].sense, '>')
        self.assertEqual(lhsconstr[1].sense, '>')

        self.assertEqual(str(lhsconstr[1]), '<gurobi.Constr lhsconstr1>')

if __name__ == '__main__':
    main()
