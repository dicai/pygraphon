"""
    Utilities for doing things with graphs, e.g., computing statistics.
"""

import numpy as np

def closed_triangle_to_square(triangle):
    """
        triangle: a numpy array with k entries in its k'th element
    """
    N = len(triangle)
    square = np.zeros((N,N))
    for i in xrange(N):
        for j in xrange(N):
            square[i,j] = triangle[j][i] if i < j \
                    else triangle[i][j] if j < i \
                    else triangle[i][j]
    return square

def triangle_to_square(triangle):
    """
        triangle: a numpy array with k-1 entries in its k'th element
    """
    N = len(triangle)
    square = np.zeros((N,N))
    for i in xrange(N):
        for j in xrange(N):
            square[i,j] = triangle[j][i] if i < j \
                    else triangle[i][j] if j < i \
                    else 0
    return square


def subdivisions(bins):
    """
    Creates subdivision for plotting the graphon.
    """
    lengths = np.array([len(x) for x in bins])
    return 1.*lengths.cumsum() / lengths.sum()


def num_edges_across_cut(G, S, T, networkx=False, mode='undirected'):
    """
    Counts the number edges between vertices in S and T.

    Arguments:
        G: a graph (either NxN numpy array or networkx graph object)
        S: cut 1 (set of vertices)
        T: cut 2 (set of vertices)
        networkx: whether or not G is a networkx objet
        mode: default is 'undirected'
            other options 'directed', 'weighted' (undirected)
    """

    if mode == 'undirected':
        if networkx:
            return int(sum([G.has_edge(x,y) for x in S for y in T]))
        else:
            return int(sum([G[x][y] for x in S for y in T]))
    elif mode == 'directed':
        pass
    elif mode == 'weighted':
        if networkx:
            # TODO: multiply by weights??? or is it the same
            return int(sum([G.has_edge(x,y) for x in S for y in T]))

            val = 0
            for x in S:
                for y in T:
                    if G.has_edge(x,y):
                        val += G.get_edge_data(x,y)['weight']
            return val

        else:
            return sum([G[x][y] for x in S for y in T])
    else:
        pass


def edge_density_across_cut(G, S, T, networkx=False, mode='undirected'):
    """
    Gets the average edge density from vertices in S and T.

    G: a graph (either NxN numpy array or networkx graph object)
    S: cut 1 (set of vertices)
    T: cut 2 (set of vertices)
    networkx: whether or not G is a networkx objet
    """
    return num_edges_across_cut(G, S, T, networkx=networkx, mode=mode) \
            / float(len(S) * len(T))


def densities_dist(A, B):
    return np.abs(A - B).sum() / A.size

