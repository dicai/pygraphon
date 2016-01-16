import pylab

import numpy as np
from numpy.linalg import svd

import networkx as nx
from bisect import bisect_left

from scipy.special import betaln
from scipy.special import gammaln

def usvt(graph, eta=0.9):

    N = graph.shape[0]

    # compute SVD
    U, S, V = svd(graph)

    # threshold
    thresh = 2 * eta * np.sqrt(N*1)

    # matrix with S on the diagonal
    s1 = np.diag(S)

    # zero values below threshold on diagonal
    s1[s1 < thresh] = 0

    # compute W matrix
    W = np.dot(U, np.dot(s1, V))

    # here we constrain values to be in [0,1]
    W[W > 1] = 1
    W[W < 0] = 0
    return W


def generate_symmetric_graph(wts):
    graph = np.random.binomial(1, wts)
    N = graph.shape[0]
    for i in range(N):
        for j in range(i):
            graph[j,i] = graph[i,j]
        graph[i,i] = 0
    return graph

def mbetaln(alpha):
    """
    performs log B(alpha) for a single vector alpha (doesn't support
        vectorization currently)
    alpha: vector of values
    """

    return gammaln(alpha).sum() - gammaln(alpha.sum())

def DP_stick(alpha, K):
    """
    alpha: concentration parameter
    K: truncation level
    """

    betas = np.random.beta(1, alpha, K)
    prods = np.cumprod(1-betas)

    sticks = np.zeros(K); sticks[0] = betas[0]
    for k in range(1, K):
        sticks[k] = betas[k] * prods[k-1]

    # normalize so they sum to 1 instead of approx 1
    return sticks / sticks.sum()

def sample_CRP(N, alpha):
    """
    N: number of people
    alpha: dispersion parameter
    """

    if N <= 0:
        return np.array([])

    # table assignments of N
    people = np.zeros(N, dtype='i4')
    # stores number of people at table i
    tables = [1]
    table_clust = [[0]]

    next_table = 1

    # generate table for 1, ..., N-1
    for n in xrange(1,N):
        # new table
        if np.random.random() < 1.*alpha / (alpha + n):
            # initialize new things
            tables.append(0)
            table_clust.append([])

            # assign person n to new table
            people[n] = next_table
            # update number of people at new table
            tables[next_table] += 1
            # update list of people at new table
            table_clust[next_table].append(n)

            # update bookkeeping
            next_table += 1
        # existing table
        else:
            # go to table with prob given by prop of people at each table
            ind = np.random.multinomial(1, 1.*np.array(tables) / sum(tables)).argmax()
            people[n] = ind
            tables[ind] += 1
            table_clust[ind].append(n)

    return people, np.array(tables), np.array(table_clust)


if __name__=="__main__":


    print 'testing mbetaln function:',
    alpha = np.array([1,2])

    if np.abs(betaln(alpha[0], alpha[1]) - mbetaln(alpha)) <= 0.001:
        print 'works'
