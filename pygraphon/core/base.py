import numpy as np
import pylab

class Graphon(object):
    def __init__(self, graphon):
        '''
        graphon: function that takes 2 arguments in [0,1], returns value in [0,1]
        '''
        self.graphon = graphon

    def sample(self, N, seed=None):
        '''
        Samples a graph of N vertices from the graphon.

        N: integer representing number of vertices to sample
        seed: set a random seed
        '''
        from pygraphon.utils.graph_utils import triangle_to_square
        from pygraphon.utils.stats import generate_symmetric_graph

        np.random.seed(seed)
        U = np.random.uniform(0,1,N)
        wts = [[self.graphon(U[i], U[j]) for i in xrange(N)] for j in xrange(N)]
        graph = generate_symmetric_graph(wts)

        return RandomGraph(graph, U, seed, Graphon(self.graphon))


    def plot(self, N=200, colorbar=False, wts=None, save=None, title=None):
        from pygraphon.core.graphon_utils import plot_graphon
        plot_graphon(self.graphon, N, colorbar, wts, save, title=title)


class RandomGraphon(Graphon):
    """
    Graphon with extra stuff to keep track of randomness.
    """
    def __init__(self, graphon, generative_process, seed):
        """
        generative_process: a function that takes x, y, an array of args, and a
            random seed and returns a symmetric function [0,1]^2 --> [0,1]
            generates a random graphon function

        seed: random seed used to generate this graphon
        """
        super(self.__class__, self).__init__(graphon)

        self.generative_process = generative_process
        self.seed = seed


class GraphonFunctionEstimator(Graphon):
    def __init__(self, graphon, graph_obj):
        super(self.__class__, self).__init__(graphon)
        self.graph_obj = graph_obj

    def to_graphon_value(self):
        from pygraphon.core.graphon_utils import discretize
        n = self.graph_obj.shape[0]
        value = discretize(self.graphon, n)
        return GraphonValueEstimator(value, self.graph_obj)

    def compute_MSE(self, weighted, permutation):
        """
        weighted: a WeightedGraph containing
        """
        value_est = self.to_graphon_value().value_est
        n = weighted.shape[0]
        weighted_new = weighted.permute(permutation).graph
        return ((value_est - weighted_new) ** 2).sum() / float(n**2)

class GraphonValueEstimator(object):
    def __init__(self, value_est, graph_obj):
        self.value_est = value_est
        self.graph_obj = graph_obj
        self.shape = value_est.shape

    def to_graphon_function(self):
        from pygraphon.core.graphon_utils import step_function
        return GraphonFunctionEstimator(step_function(self.value_est),
            self.graph_obj)

    def compute_MSE(self, weighted, permutation):
        """
        weighted: a WeightedGraph containing
        """
        n = weighted.shape[0]
        # take the permutation applied to graph to get value est to weighted
        weighted_new = weighted.permute(permutation).graph
        return ((self.value_est - weighted_new) ** 2).sum() / float(n**2)

    def plot(self, wts=None, cumsum=False, clusts=None, colorbar=None,
            save=None, title=None):
        """
        Plots the adjacency matrix. If an argument for graph is passed in, plots
        the new adjacency matrix (e.g., a resorted array).

        Arguments:
        wts: weights of division, either must sum to 1 or be the cumsum of it
        cumsum: whether weights are cumsum or not
        clusts: vertex assignment? TODO: need to pass colors
        colorbar: whether or not to display a colorbar
        save: by default, no save, but if a string is passed, saves it with that name
        """
        from pygraphon.core.graphon_utils import plot_graph
        plot_graph(self.value_est, colorbar=colorbar, save=save, title=title)

    def __repr__(self):
        print('GraphonValueEstimator: \n' + str(self.value_est))

class Graph(object):
    def __init__(self, graph):
        """
        graph: 2D numpy array that represents the adjacency matrix of the graph
        """
        self.graph = graph
        self.shape = graph.shape

    def to_step_function(self):
        """
        Returns the step function representation of this finite graph.
        """
        from graphon_utils import step_function
        return Graphon(step_function(self.graph))

    def plot(self, graph=None, wts=None, cumsum=False, clusts=None,
            colorbar=None, save=None, title=None):
        """
        Plots the adjacency matrix. If an argument for graph is passed in, plots
        the new adjacency matrix (e.g., a resorted array).

        Arguments:
        graph: use a different graph (2D numpy array) than the original sample
        wts: weights of division, either must sum to 1 or be the cumsum of it
        cumsum: whether weights are cumsum or not
        clusts: vertex assignment? TODO: need to pass colors
        colorbar: whether or not to display a colorbar
        save: by default, no save, but if a string is passed, saves it with that name
        """
        from pygraphon.core.graphon_utils import plot_graph

        if graph is not None:
            plot_graph(graph, wts, cumsum, clusts, colorbar, save, title)
        else:
            plot_graph(self.graph, colorbar=colorbar, save=save, title=title)

    def sort_by_degree(self):
        from pygraphon.core.graphon_utils import sort_rows_columns_full
        from pygraphon.core.graphon_utils import get_permutations

        order = self.graph.sum(0).argsort()
        forward, reverse = get_permutations(order)
        graph, m = sort_rows_columns_full(self.graph)
        return PermutedGraph(graph, forward, reverse)

    def sort_by_triangle(self):
        from pygraphon.core.graphon_utils import sort_triangle_only
        from pygraphon.core.graphon_utils import sort_graph_given_order
        from pygraphon.core.graphon_utils import get_permutations

        order = sort_triangle_only(self.graph)
        forward, reverse = get_permutations(order)
        graph = sort_graph_given_order(self.graph, order)
        return PermutedGraph(graph, forward, reverse)

    def sort_by_gray(self):
        from pygraphon.core.graphon_utils import order_unlabel
        ## TODO: implement for permuted graph
        return Graph(np.array(sorted(self.graph, cmp=order_unlabel)))

    def smooth(self, graph=None, type='usvt'):
        from pygraphon.utils.stats import usvt

        if graph is not None:
            return GraphonValueEstimator(usvt(graph), Graph(graph))
        else:
            return GraphonValueEstimator(usvt(self.graph), Graph(self.graph))

    def hist(self, m, graph=None):
        """
        Divides the vertices into m equally-sized bins and returns a histogram
            step-function estimator
        """
        from pygraphon.core.graphon_utils import cluster_graphon

        N = self.shape[0]
        val = int(N / m)
        assert N == (m-1)*val + (val + N - val*m)
        bins = [range((a)*val,(a*val) +(val)) for a in range(0,m-1)] + [range(N
            - (val + N - val*m) - 1, N)]

        return GraphonFunctionEstimator(cluster_graphon(self.graph, bins),
                Graph(self.graph))

    def __repr__(self):
        return 'Graph: \n' + str(self.graph)


class RandomGraph(Graph):
    """
    A Graph with a few extra things keeping track of randomness, etc.
    In particular, these are W-random graphs (generated from some graphon W).

    graph: 2D numpy array representing the adjacency matrix of the graph
    unifs: sequence of uniform random variables from which this graph was generated
    seed: random seed used to generate uniform random variable sequence (can be None)
    graphon_obj: Graphon object from which this RandomGraph instance was generated
    """

    def __init__(self, graph, unifs, seed, graphon_obj):
        super(RandomGraph, self).__init__(graph)

        self.unifs = unifs
        self.seed = seed
        self.graphon_obj = graphon_obj

    def sort_by_unifs(self):
        """
        Sort the adjacency matrix using the order of the uniform random
        variables.

        Returns a PermutedGraph object, which is a Graph with information about
            the permutation that was applied to it.

        What if any sorted graph is a SortedGraph object where you can access
        the reverse permutation? bad for large graphs.

        """
        from pygraphon.core.graphon_utils import sort_graph_by_unifs
        from pygraphon.core.graphon_utils import get_permutations

        order = self.unifs.argsort()
        forward, reverse = get_permutations(order)

        return PermutedGraph(sort_graph_by_unifs(self.graph, self.unifs),
                forward, reverse)

    def get_weighted_graph(self):
        """
        Returns the weighted graph that this was sampled from
        """
        g = self.graphon_obj.graphon
        U = self.unifs
        N = len(U)

        weighted = np.array([np.array([g(U[i], U[j]) for i in xrange(N)])
                for j in range(N)])

        return WeightedGraph(weighted, self.unifs, self.seed, self.graphon_obj)

    def __repr__(self):
        return 'RandomGraph: \n' + str(self.graph)


class WeightedGraph(RandomGraph):
    def __init__(self, graph, unifs, seed, graphon_obj):
        super(WeightedGraph, self).__init__(graph, unifs, seed, graphon_obj)

    def permute(self, permutation):

        from pygraphon.core.graphon_utils import sort_given_permutation
        # apply the permutation

        #return PermutedWeightedGraph(sort_given_permutation(self.graph,
        #    permutation), permutation,
        #    self.permutation)
        pass

        ### FIXME
        return WeightedGraph(sort_given_permutation(self.graph, permutation),
                self.unifs, self.seed, self.graphon_obj)

    def __repr__(self):
        return 'WeightedGraph: \n' + str(self.graph)


class PermutedGraph(Graph):
    def __init__(self, graph, permutation, reverse_permutation):
        super(PermutedGraph, self).__init__(graph)

        self.permutation = permutation
        self.reverse_permutation = reverse_permutation

    def reverse(self):
        from pygraphon.core.graphon_utils import sort_given_permutation
        return PermutedGraph(sort_given_permutation(self.graph,
            self.reverse_permutation), self.reverse_permutation,
            self.permutation)

class PermutedWeightedGraph(PermutedGraph):
    def __init__(self, graph, unifs, seed, graphon_obj, permutation, reverse_permutation):
        super(PermutedWeightedGraph, self).__init__(graph, permutation,
                reverse_permutation)

        self.unifs = unifs
        self.seed = seed
        self.graphon_obj = graphon_obj
