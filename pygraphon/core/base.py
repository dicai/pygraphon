import numpy as np
import pylab

###############################################################################
# Graphons
###############################################################################
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


###############################################################################
# Digraphons
###############################################################################
NUM_TYPES = 4
class Digraphon(object):

    def __init__(self, digraphon):
        '''
	pass single function --- assume only 4 things for now
	IGNORING SELF LOOPS
        '''
        self.digraphon = digraphon


    # TODO MODIFY
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
	G = np.zeros((N,N))
        types = np.zeros((N, NUM_TYPES))
	mats = np.zeros((4,N,N))

	#### TODO
	# generate all the weights first
	# then flip those
	W = self.digraphon
	w = lambda x: 0
	for i in xrange(N):
	    for j in xrange(N):
		# self-loops
		if i == j:
		    # TODO: make sure this is right
		    G[i,j] = np.random.binomial(1, w(U[i]))
		    #G[i,j] = w(U[i])
		if i < j:
		    # should return 4 tuple summing to 1
		    val = W(U[i], U[j])

		    # warning if not close to 1
		    if np.abs(val.sum() - 1) > 0.01:
			raise Exception('Value did not sum to 1: ', val)

		    k = int(np.random.multinomial(1, val).argmax())

		    G[i,j] = k / 2
		    G[j,i] = k % 2
                    # note: only counts the top triangle
                    types[i,k] += 1
                    mats[k,i,j] = mats[k,j,i] = 1

	return RandomDigraph(G, U, seed, Digraphon(self.digraphon), types, mats)

#        np.random.seed(seed)
#        U = np.random.uniform(0,1,N)
#        wts = [[self.graphon(U[i], U[j]) for i in xrange(N)] for j in xrange(N)]
#        graph = generate_symmetric_graph(wts)
#
#        return RandomGraph(graph, U, seed, Graphon(self.graphon))

    def sample_digraph_quad(self, N, W_00, W_01, W_10, W_11, w, seed=None):
	"""
	Wrapper for sample_digraph
	"""

	def convert_quadruple_to_W(W_00, W_01, W_10, W_11):
	    return lambda x,y: np.array([W_00(x,y), W_01(x,y), W_10(x,y), W_11(x,y)])

	W = convert_quadruple_to_W(W_00, W_01, W_10, W_11)
	return self.sample(N, W, w, seed=seed)


    # TODO MODIFY
    def plot(self, N=200, colorbar=False, wts=None, save=None, title=None):
        from pygraphon.core.digraphon import plot_digraphon
        plot_digraphon(self.digraphon, N, w=None, save=save, wts=wts)

class RandomDigraphon(Digraphon):
    pass


###############################################################################
# Graphs
###############################################################################
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

    # TODO: allow argument for sorting by 4 types of directions
    def sort_by_degree(self, out=False):
        from pygraphon.core.graphon_utils import sort_rows_columns_full
        from pygraphon.core.graphon_utils import get_permutations

        if out:
            order = self.graph.sum(1).argsort()
        else:
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

###############################################################################
# Directed Graphs
###############################################################################
## TODO Can this inherit from Graph?
## FOR NOW TREAT AS OWN OBJECT BUT CHANGE LATER
class Digraph(object):
    def __init__(self, graph, types=None, mats=None):
        """
        graph: 2D numpy array that represents the adjacency matrix of the graph
        """
        self.graph = graph
        self.shape = graph.shape

	# if generated by digraphon, store 4 types of directions
	# TODO: generate this object when reading in the graph
	self.types = types
        self.mats = mats

    #### TODO: is there an equivalent ?
#    def to_step_function(self):
#        """
#        Returns the step function representation of this finite graph.
#        """
#        from graphon_utils import step_function
#        return Digraphon(step_function(self.graph))

    def generate_types(self):
        if self.types is not None:
            # generate 4 directions matrix if this is called
            G = self.graph
            N = G.shape[0]
            types = np.zeros((N, NUM_TYPES))
            for i in range(N):
                # TODO ignores self-loops
                for j in range(i):
                    if G[i,j] == 0 and G[j,i] == 0:
                        types[i, 0] += 1
                    if G[i,j] == 1 and G[j,i] == 0:
                        types[i, 1] += 1
                    if G[i,j] == 0 and G[j,i] == 1:
                        types[i, 2] += 1
                    if G[i,j] == 1 and G[j,i] == 1:
                        types[i, 3] += 1
                    else:
                        raise Exception("Non-binary graph")
        # TODO: confirm this is correct
        return types

    def generate_mats(self):
        pass

    def set_types(self, types):
        self.types = types

    def set_mats(self, mats):
        self.mats = mats

    def get_mats(self):
        return (Graph(self.mats[0]), Graph(self.mats[1]),
            Graph(self.mats[2]), Graph(self.mats[3]))

    # TODO update --- probably the same as in the graph case
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

        types = self.types
        inds = types.argsort(0)
        N = types.shape[0]
        graph = self.graph

        graphs = [np.zeros((N,N)), np.zeros((N,N)),
                np.zeros((N,N)),np.zeros((N,N))]

        for i in range(N):
            for j in range(i):
                inds0 = inds[:,0]
                inds1 = inds[:,1]
                inds2 = inds[:,2]
                inds3 = inds[:,3]

                graphs[0][i, j] = graph[inds0[i], inds0[j]]
                graphs[0][j, i] = graph[inds0[j], inds0[i]]

                graphs[1][i, j] = graph[inds1[i], inds1[j]]
                graphs[1][j, i] = graph[inds1[j], inds1[i]]

                graphs[2][i, j] = graph[inds2[i], inds2[j]]
                graphs[2][j, i] = graph[inds2[j], inds2[i]]

                graphs[3][i, j] = graph[inds3[i], inds3[j]]
                graphs[3][j, i] = graph[inds3[j], inds3[i]]

        return (PermutedGraph(graphs[0], None, None),
                PermutedGraph(graphs[1], None, None),
                PermutedGraph(graphs[2], None, None),
                PermutedGraph(graphs[3], None, None))

        #from pygraphon.core.graphon_utils import sort_rows_columns_full
        #from pygraphon.core.graphon_utils import get_permutations
        #order = self.graph.sum(0).argsort()
        #forward, reverse = get_permutations(order)
        # TODO RETURN PERMUTED GRAPH
        #return PermutedGraph(graph, forward, reverse)

    # TODO: allow argument for sorting by 4 types of directions
    def sort_by_triangle(self):
        from pygraphon.core.graphon_utils import sort_triangle_only
        from pygraphon.core.graphon_utils import sort_graph_given_order
        from pygraphon.core.graphon_utils import get_permutations

        order = sort_triangle_only(self.graph)
        forward, reverse = get_permutations(order)
        graph = sort_graph_given_order(self.graph, order)
        return PermutedGraph(graph, forward, reverse)

    # TODO: allow argument for sorting by 4 types of directions
    def sort_by_gray(self):
        from pygraphon.core.graphon_utils import order_unlabel
        ## TODO: implement for permuted graph
        return Graph(np.array(sorted(self.graph, cmp=order_unlabel)))

    ## TODO: apply USVT to all 4 functions
    def smooth(self, graphs, type='usvt'):
    #def smooth(self, graph=None, type='usvt'):
        from pygraphon.utils.stats import usvt

        graph0 = graphs[0].graph
        graph1 = graphs[1].graph
        graph2 = graphs[2].graph
        graph3 = graphs[3].graph

        return (GraphonValueEstimator(usvt(graph0), Graph(graph0)),
            GraphonValueEstimator(usvt(graph1), Graph(graph1)),
            GraphonValueEstimator(usvt(graph2), Graph(graph2)),
            GraphonValueEstimator(usvt(graph3), Graph(graph3)))

        ## TODO allow for the composibility
        #else:
        #    return GraphonValueEstimator(usvt(self.graph), Graph(self.graph))

    #  do a coarse 4-hist estimator
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


class RandomDigraph(Digraph):
    """
    A Graph with a few extra things keeping track of randomness, etc.
    In particular, these are W-random graphs (generated from some graphon W).

    graph: 2D numpy array representing the adjacency matrix of the graph
    unifs: sequence of uniform random variables from which this graph was generated
    seed: random seed used to generate uniform random variable sequence (can be None)
    graphon_obj: Graphon object from which this RandomGraph instance was generated
    """

    def __init__(self, graph, unifs, seed, digraphon_obj, types=None, mats=None):
        super(RandomDigraph, self).__init__(graph, types, mats)

        self.unifs = unifs
        self.seed = seed
        self.digraphon_obj = digraphon_obj

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
