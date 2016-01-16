"""
    graphon utilities
"""

import numpy as np
from bisect import bisect_left
import pylab

import sys, os

from pygraphon.utils.graph_utils import closed_triangle_to_square
from pygraphon.utils.graph_utils import triangle_to_square
from pygraphon.utils.graph_utils import num_edges_across_cut
from pygraphon.utils.graph_utils import edge_density_across_cut
from pygraphon.utils.graph_utils import subdivisions


def cluster_graphon(G, bins):
    """
    Produces a graphon function given a partition of the vertices (list of
        lists containing vertices).

    arguments:
        G: 2D numpy array
        bins: the partition (list of lists)
    """
    edge_counts = np.array([num_edges_across_cut(G, bin, range(len(G)))
        / len(bin) for bin in bins])

    inds = edge_counts.argsort()[::-1]
    new_bins =  np.array(bins)[inds]
    subdivision = subdivisions(new_bins)

    # iterate up to j+1 to count self-edges
    triangle = [[edge_density_across_cut(G, new_bins[i], new_bins[j]) for i in xrange(j+1)]
            for j in xrange(len(new_bins))]

    wts = closed_triangle_to_square(triangle)

    return lambda x, y: wts[bisect_left(subdivision, x),bisect_left(subdivision, y)]


def step_function(graph):
    """
    Given a graph, returns the step-function graphon of it.

    arguments:
        graph: 2d numpy array for adjacency matrix of an undirected graph
    """
    N = graph.shape[0]
    pixel_size = 1./N
    return lambda x, y: graph[np.floor(x/pixel_size), np.floor(y/pixel_size)]


################################################################################
# sampling utilities
################################################################################

def sample_subgraph(graphon, N, networkx=False, seed=1001, weighted=False,
        sort_U=False):
    """
    Samples a subgraph of size N given a graphon function. Returns a 2D numpy
    array representing the adjacency matrix.

    arguments:
        graphon: a symmetric function from [0,1]^2 -> [0,1]
        N: number of points along each axis at which to sample
        seed: random seed
        weighted: if True, returns weighted n-vertex graph from W(U_i, U_j) values
        sort_U: sort the U_i values first before determining edges
    """

    from pygraphon.utils.stats import generate_symmetric_graph

    np.random.seed(seed)
    U = np.random.uniform(0,1,N)
    if sort_U == True:
        U.sort()

    wts = np.array([np.array([graphon(U[i], U[j]) for i in xrange(N)])
            for j in xrange(N)])

    if weighted:
        graph = wts
    else:
        graph = generate_symmetric_graph(wts)

    if networkx:
        return nx.Graph(graph)
    else:
        return graph


################################################################################
# graphon estimator evaluation
################################################################################

def compute_MSE_sorting(W, W_hat, inds, l2=True, seed=None):

    '''
    W: original graphon
    W_hat: output of SAS etc
    ##### [old] sample: NxN graph
    inds: array of inds to sort
    l2: whether or not to use l2
    seed: seed that sample was sampled with
    '''

    ### squared error for sorting by degree

    if seed == None:
        raise Exception("Must specify a seed")

    N = W_hat.shape[0]

    #inds = sample.sum(1).argsort()[::-1]

    # get U_is that correspond to ones used to sample the data
    np.random.seed(seed)
    U = np.random.uniform(0, 1, N)
    # construct the matrix M_ij := W(U_i, U_j)
    W_values = np.zeros((N, N))
    for i in xrange(N):
        for j in xrange(N):
            W_values[i,j] = W(U[i], U[j])

    ### sorted version of M_ij (according to degree of data)
    W_sort = np.array([np.array([W_values[inds[i], inds[j]] for i in xrange(N)])
        for j in xrange(N)])


    # return MSE
    if l2:
        return  ((W_hat - W_sort)**2).sum() / (N**2)
    # return MAE
    else:
        return  np.abs(W_hat - W_sort).sum() / (N**2)


def compute_MSE_new(W, W_hat, l2=True, seed=None):
    """
    MSE for graphon value estimation!

        W: original graphon function
        W_hat: nxn matrix (numpy array) estimator
        l2: whether to do L2 (or L1)
        seed: the seed data was generated using

    returns MSE (or MAE)
    """

    if seed == None:
        raise Exception("Must specify a seed")

    N = W_hat.shape[0]

    # get U_is that correspond to ones used to sample the data
    np.random.seed(seed)
    U = np.random.uniform(0, 1, N)
    # sort the U_i's
    U.sort()

    # construct the matrix W(U_i, U_j), sorted by increasing U_i
    W_values = np.zeros((N, N))
    for i in xrange(N):
        for j in xrange(N):
            W_values[i,j] = W(U[i], U[j])

    # return MSE
    if l2:
        return  ((W_hat - W_values)**2).sum() / (N**2)
    # return MAE
    else:
        return  np.abs(W_hat - W_values).sum() / (N**2)

def compute_MSE(sorted_G, graphon, l2=True):
    """
    graph: graph is the sampled adjacency matrix
    g_est: is the partition object
    """
    convert = lambda v1, v2, num: (1./num * v1, 1./num * v2)
    MSE = 0
    N = sorted_G.shape[0]
    for v1 in xrange(N):
        for v2 in xrange(N):
            u1, u2 = convert(v1, v2, N)
            val = graphon(u1, u2)
            if l2:
                MSE += (val - sorted_G[v1, v2]) ** 2
            else:
                MSE += np.abs(val - sorted_G[v1, v2])

    return MSE / N**2

def compute_MSE_graphon(sorted_G, disc_G):
    N = len(sorted_G)
    assert  N == len(disc_G)
    MSE = 0
    return ((disc_G.ravel() - sorted_G.ravel)**2).sum()
    for v1 in xrange(N):
        for v2 in xrange(N):
            MSE += (disc_G[v1, v2] - sorted_G[v1, v2]) ** 2
    return MSE


################################################################################
# sorting utilties
################################################################################

def sort_graph_by_unifs(graph, unifs):
    """
    Given a graph, sorts it according to increasing value of the uniform random
    variables from which it was sampled, i.e., reorders the rows and columns so
    as to look like the original graphon.

    arguments:
        graph: 2D numpy array
        unifs: uniform random variables from which graph was sampled
    """
    N = graph.shape[0]
    u_order = unifs.argsort()
    return np.array([np.array([graph[u_order[i], u_order[j]] for j in
        xrange(N)]) for i in xrange(N)])

def sort_graph_given_order(graph, order):
    """
    Given a graph, sorts it according to the specified order of vertices.

    arguments:
        graph: 2D numpy array
        order: list (or array) mapping current index to new index
    """
    N = graph.shape[0]
    return np.array([np.array([graph[order[i], order[j]] for j in
        xrange(N)]) for i in xrange(N)])

def sort_rows_columns_full(graph, sortfn=np.argsort, noise=True, iters=1,
        sparse=False, K=1000):
    """
    Sorts rows and columns, returns sorted result and a reverse map

    graph: 2D numpy array for adjacency matrix of undirected graph
    sortfn: the function by which to order graph, returns new ordering of rows
        e.g. output of np.argsort
    noise: if True, adds small uniform noise
    iters: recursion parameter
    sparse: if sparse, sorts the edges, returning a new graph with top K
    """

    reverse_map = {}

    N = graph.shape[0]

    if noise:
        rows_perturbed = graph.sum(0) + np.random.uniform(-0.3, 0.3, size=N)
    else:
        ### TODO: was this intentional?
        rows_perturbed = graph

    # this doesn't take arbitrary sorting method for now
    # TODO: factor this out to do degree, triangles, etc.
    if sparse:
        graph_nx = nx.Graph(graph)
        counts = np.zeros(N)
        for i,j in graph_nx.edges():
            counts[i] += 1
            counts[j] += 1

        inds = counts.argsort()[::-1][:K]

        for i in range(len(inds)):
            reverse_map[inds[i]] = i
        #reverse_map[new_ind] = inds[i] for i in range(len(inds))

        print 'Reordering graph...'

        reordered_graph = np.array([np.array([graph[inds[i], inds[j]]
            for i in xrange(K)]) for j in xrange(K)])

        return reordered_graph, reverse_map

    else:
        # gets new order of rows/columns
        order = sortfn(rows_perturbed)[::-1]

        for i in range(len(order)):
            reverse_map[order[i]] = i
        # create new graph with permuted rows and columns
        reordered_graph = np.array([np.array([graph[order[i], order[j]]
            for i in xrange(N)]) for j in xrange(N)])

        if iters == 1:
            return reordered_graph, reverse_map
        else:
            raise Exception


def sort_triangle_only(graph):
    """
    returns indices for the new sorted order

    graph: 2D numpy array, adj matrix # TODO cache degrees
    """
    # TODO: make more efficient
    tris = compute_triangles(graph)
    return np.argsort(tris)[::-1]

def compute_degree(graph, ax=1):
    return graph.sum(ax)

def compute_triangles(graph, sparse=False):
    """
    graph: NxN numpy array, adjacency matrix

    returns dict containing dict[node] = number of triangles node is part of
    """

    N = graph.shape[0]
    triangles = np.zeros(N, dtype='i4')
    seen = set()

    if sparse:
        # search all pairs of edges
        # would need this information pre-computed
        pass
    else:
        # search all triples of vertices
        for i in xrange(N):
            for j in xrange(N):
                for k in xrange(N):
                    if graph[i,j] and graph[i,k] and graph[j,k]:
                        if frozenset([i,j,k]) not in seen:
                            triangles[i] += 1
                            triangles[j] += 1
                            triangles[k] += 1
                            seen.add(frozenset([i,j,k]))

    return triangles

def gray_order(row1, row2, label=None):
    """
    input: two rows (numpy arrays) of adj matrix.
    lable: if there's a label in the first entry

    we assume no rows consist of all zeros

    """

    if label:
        row1 = row1[1]
        row2 = row2[1]
        i = np.nditer(np.nonzero(row1))
        j = np.nditer(np.nonzero(row2))
    else:
        i = np.nditer(np.nonzero(row1))
        j = np.nditer(np.nonzero(row2))

    p = False

    while True:

        try:
            a = i.next()
        except:
            a = np.inf
        try:
            b = j.next()
        except:
            b = np.inf

        if i.finished and j.finished:
            return 0

        if a != b:
            if p ^ (a < b):
                return 1
            else:
                return -1

        p = not p

# for a regular list of bits
order_unlabel = lambda a,b: gray_order(a, b, label=False)

def get_permutations(order):
    """
    Helper function that given a mapping (list), returns the forward (same as list
    but as a map) and reverse permutations.

    order: a list that maps index -> new value (is a permutation)
    """
    forward = {}
    reverse = {}
    for ind, val in enumerate(order):
        forward[ind] = val
        reverse[val] = ind
    return forward, reverse


def sort_given_permutation(graph, perm):
    """
    Given a graph, reorders the graph according to some permutation (a map).
    Returns the new graph.

    graph: 2D numpy array
    perm: a 1-to-1 mapping between vertices.
    """
    N = graph.shape[0]
    return np.array([np.array([graph[perm[i], perm[j]] for i in range(N)])
        for j in range(N)])


################################################################################
# graphon plotting utilties
################################################################################

def plot_graphon(W, N=200, colorbar=False, wts=None, save=None, title=None):
    """
    Plots the graphon, given a function.

    W: graphon function [0,1]^2 -> [0,1]
    N: number of points to discretize (bigger values, more pixels)
    colorbar: whether or not to display a colorbar
    wts: weights
    save: if None, doesn't save; if a string, saves a file with that name
    title: add a title to the plot with this string
    """

    if save is not None:
        pylab.figure(figsize=(2,2))

    disc = _discretize(W, N)
    plot_graph(disc, colorbar=colorbar, title=title)

    if wts is not None:
        wts_c = wts.cumsum()
        lw=1; c='black'
        [(pylab.axvline(wt*N, c=c, lw=lw), pylab.axhline(wt*N, c=c, lw=lw))
                for wt in wts_c[:-1]]

    if save is not None:
        pylab.savefig('%s.png' % save, dpi=100, bbox_inches='tight')
    else:
        pylab.show()

def plot_graph(G, wts=None, cumsum=False, clusts=None, colorbar=False,
        save=None, title=None):
    """
    Plots a graph (2D numpy array).

    G: NxN numpy array
    wts: cluster proportion weights for drawing lines showing delineation
    cumsum: whether the weights are a cumsum or not
    clusts: clusters vertices belong to
    colorbar: whether to display a colorbar
    save: if string arguments, saves a file with that name
    title: add a title with this string to the plot
    """

    if save is not None:
        pylab.figure(figsize=(2,2))

    pylab.imshow(G, cmap='gray_r', interpolation='nearest', vmin=0., vmax=1.)
    pylab.gca().tick_params(labelbottom='off', labelleft='off',
        bottom='off', top='off', left='off', right='off')

    if colorbar:
        pylab.colorbar()

    if wts is not None:
        N = G.shape[0]
        if cumsum:
            wts_c = wts
        else:
            wts_c = wts.cumsum()
        lw=2; c='b'
        [(pylab.axvline(wt*N, c=c, lw=lw), pylab.axhline(wt*N, c=c, lw=lw))
                for wt in wts_c[:-1]]

    if clusts is not None:
        ### TODO: need to allow for more colors
        col= ['red', 'purple', 'yellow']
        for c in xrange(len(clusts)):
            for v in clusts[c]:
                pylab.axhline(v, c=col[c], lw=lw, alpha=0.3)
                pylab.axvline(v, c=col[c], lw=lw, alpha=0.3)

    if title is not None:
        pylab.title(title)

    if save is not None:
        pylab.savefig('%s.png' % save, dpi=100, bbox_inches='tight')
    else:
        pylab.show()

def discretize(graphon, N):
    """
    Given a graphon, discretizes it along N points.

    graphon: a symmetric function from [0,1]^2 -> [0,1], see above
    N: number of points along each axis at which to discretize
    """
    rg = np.arange(0, 1, 1./N)
    return np.array([[graphon(x,y) for x in rg] for y in rg])

def _discretize(graphon, N):
    """
    Given a graphon, discretizes it along N points.

    graphon: a symmetric function from [0,1]^2 -> [0,1], see above
    N: number of points along each axis at which to discretize
    """
    rg = np.arange(0, 1, 1./N)
    return np.array([[graphon(x,y) for x in rg] for y in rg])


################################################################################
# graph plotting utilties, requires networkx
################################################################################

def draw_graph(graph, axis=None):
    """
    Given a graph, draws it using networkx's draw.
    """
    import networkx as nx
    G = nx.Graph(graph)
    return nx.draw(G, ax=axis)

def sample_and_draw_graphs(graphon, N, num_graphs):
    """
    Sample a (or several) graph(s) from a graphon and draw it.
    """
    graphs = [sample_subgraph(graphon, N) for _ in xrange(num_graphs)]
    draw_graphs(graphs)

def draw_graphs(graphs):
    """
    Given several graphs, draws them using networkx's draw.
    """
    num_graphs = len(graphs)
    k = int(np.ceil(np.sqrt(num_graphs)))
    fig = pylab.figure(figsize=(18, 12))
    gs = pylab.gridspec.GridSpec(k, k)

    for i in xrange(k):
        for j in xrange(k):
            if i*k + j < num_graphs:
                ax = fig.add_subplot(gs[i, j])
                draw_graph(graphs[i*k + j], ax)
    gs.tight_layout(fig)

