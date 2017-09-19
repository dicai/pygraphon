"""
Examples of graphons
    contains graphons, random graphons, and functions that generate them.
"""

import os
import sys
import numpy as np
from bisect import bisect_left

from pygraphon.utils.graph_utils import closed_triangle_to_square
from pygraphon.utils.graph_utils import triangle_to_square
from pygraphon.utils.graph_utils import num_edges_across_cut
from pygraphon.utils.graph_utils import edge_density_across_cut
from pygraphon.utils.graph_utils import subdivisions

################################################################################
# Functions that generate graphons
################################################################################
bipartite = lambda p, q: \
            (lambda x, y: p if (x < q and y >= q) or (x >= q and y < q) else 0)

er = lambda p: (lambda x, y: p)

def SBM(p1=0.35, p2=0.65, q=0.4):
    return lambda x, y: p1 if (x < q and y >= q) or (x >= q and y < q) else p2

def blockmodel(seed=1001):

    np.random.seed(seed)
    num_divisions = np.max((1, int(np.floor(np.random.gamma(9, 0.5))+2)))
    print num_divisions
    subdivision = np.sort(np.random.uniform(size=num_divisions-1))
    triangle = [[np.random.beta(0.5, 0.5) for i in xrange(j)]
        for j in xrange(num_divisions)]
    wts = triangle_to_square(triangle)
    return lambda x, y: wts[bisect_left(subdivision, x),bisect_left(subdivision, y)]

def IRM_full(alpha=1, a=1, b=1, T=2000, seed=1001, symmetric=True):
    """
    Returns the IRM function along with cluster props and weights.
    """

    from pygraphon.utils.stats import DP_stick

    np.random.seed(seed)
    dpwts = DP_stick(alpha, T)
    num_row_clust = len(dpwts)

    if symmetric:
        triangle = [[np.random.beta(a, b) for i in xrange(j+1)]
            for j in xrange(num_row_clust)]
        wts = closed_triangle_to_square(triangle)
    else:
        wts = np.array([np.array([np.random.beta(a, b) for i in
            xrange(num_row_clust)]) for j in xrange(num_row_clust)])

    row_wts = dpwts.cumsum()

    return lambda x, y: wts[bisect_left(row_wts, x),
            bisect_left(row_wts, y)], dpwts, wts

def IRM_symmetric(alpha=1, a=1, b=1, T=2000, seed=1001):
    """
    Symmetric IRM function only.

    alpha: concentration parameter
    a, b: beta parameters
    T: truncation parameter
    seed: random seed
    """

    from pygraphon.utils.stats import DP_stick

    np.random.seed(seed)
    dpwts = np.sort(DP_stick(alpha, T))[::-1]
    num_row_clust = len(dpwts)

    triangle = [[np.random.beta(a, b) for i in xrange(j+1)] for j in xrange(num_row_clust)]
    wts = closed_triangle_to_square(triangle)
    row_wts = dpwts.cumsum()

    return lambda x, y: wts[bisect_left(row_wts, x), bisect_left(row_wts, y)]

def IRM(alpha=1, a=1, b=1, T=2000, seed=1001, verbose=False):
    """
    IRM function but no longer symmetric in the partition.
    #TODO: asymmetric IRM

    alpha: concentration parameter
    a, b: beta parameters
    T: truncation parameter
    seed: random seed
    """

    from pygraphon.utils.stats import DP_stick

    np.random.seed(seed)
    dp_row_wts = DP_stick(alpha, T)
    dp_col_wts = DP_stick(alpha, T)
    num_row_clust = len(dp_row_wts)
    num_col_clust = len(dp_col_wts)
    thetas = np.random.beta(a, b, (num_row_clust, num_col_clust))
    row_wts = dp_row_wts.cumsum()
    col_wts = dp_col_wts.cumsum()

    return lambda x, y: thetas[bisect_left(row_wts, x), bisect_left(col_wts, y)]


################################################################################
# Functions that are graphons
################################################################################

# graphons from ipython notebook
gradient = lambda x, y: ((1-x) + (1-y)) / 2

# graphons listed in Chan and Airoldi, 2014
g1 = lambda u, v:  u * v
g2 = lambda u, v: np.exp(-(u**0.7 + v**0.7))
g3 = lambda u, v: 0.25 * (u**2 + v**2 + u**0.5 + v**0.5)
g4 = lambda u, v: 0.5 * (u + v)
g5 = lambda u, v: 1./ (1 + np.exp(-10 * (u**2 + v**2)))
g6 = lambda u, v: np.abs(u - v)
g7 = lambda u, v: 1./(1 + np.exp(-np.max(u,v)**2 + np.min(u,v)**4))
g8 = lambda u, v: np.exp(-np.max(u,v)**0.75)
g9 = lambda u, v: np.exp(-0.5*(np.min(u,v) + u**0.5 + v**0.5))
g10 = lambda u, v: np.log(1 + 0.5 * np.max(u,v))
