from __future__ import absolute_import

import numpy as np
from scipy.spatial import cKDTree as KDTree
from itertools import product
import scipy.sparse as sparse
from six.moves import map
from six.moves import range
from six.moves import zip

def Points2Points(orig, dest, surjective=False):
    """
    Computes the NxM transfer matrix from M k-dimensional points
    `orig` to N k-dimensional points `dest`.

    Multiplying an M-dim vector of data at the points in `orig` (the i-th
    entry in the vector corresponds to the data at the i-th
    coordinates in `orig`) with this transfer matrix will result in
    the N-dim vector at the points in `dest`.
    """

    transfer = sparse.lil_matrix((len(dest), len(orig)), dtype=np.float)

    _, indy = KDTree(dest).query(orig)
    transfer[indy, list(range(len(indy)))] = 1

    if surjective:
        _, indx = KDTree(orig).query(dest)
        for y, x in enumerate(indx): transfer[y,x] += 1

        # sum of input vectors must be preserved
        ssum = np.squeeze(np.asarray(transfer.sum(axis=0)))
        for i,j in zip(*transfer.nonzero()):
            transfer[i,j] /= ssum[j]

    return transfer.tocsr()

try:
    import shapely.geometry as geo
    from shapely.prepared import prep
    from .shapes import reproject

    def Shapes2Points(orig, dest, **kwargs):
        return Points2Points(Centroid(orig), dest, **kwargs)

    def Points2Shapes(orig, dest, **kwargs):
        return Points2Points(orig, Centroid(dest), **kwargs)

    def Shapes2Shapes(orig, dest, normed=True, equalarea=False, prep_first=True, **kwargs):
        if equalarea:
            dest = list(map(reproject, dest))
            orig = list(map(reproject, orig))

        if prep_first:
            orig_prepped = list(map(prep, orig))
        else:
            orig_prepped = orig

        transfer = sparse.lil_matrix((len(dest), len(orig)), dtype=np.float)
        for i,j in product(range(len(dest)), range(len(orig))):
            if orig_prepped[j].intersects(dest[i]):
                area = orig[j].intersection(dest[i]).area
                transfer[i,j] = area/dest[i].area

        # sum of input vectors must be preserved
        if normed:
            ssum = np.squeeze(np.asarray(transfer.sum(axis=0)))
            for i,j in zip(*transfer.nonzero()):
                transfer[i,j] /= ssum[j]

        return transfer

    def asShapely(shape):
        if isinstance(shape, geo.base.BaseGeometry):
            return shape
        else:
            return geo.Polygon(shape)

    def Centroid(shapes):
        pts = np.empty((len(shapes), 2))
        for i, sh in enumerate(shapes):
            pts[i] = asShapely(sh).centroid
        return pts

except ImportError:
    pass
