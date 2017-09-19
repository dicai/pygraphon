from pygraphon.core.base import GraphonFunctionEstimator


def step_function_graphon(partition, graph_nx, graph_samp=None):
    """
    Given a partition (dictionary), returns the step-function graphon
	associated with it.

    partition: a dictionary whose keys correspond to centroids, values form a partition
	note: the centroid values can be arbitrary as this method doesn't use them

    graph_nx: the sample as a networkx graph

    graph_sample: the sample as a pygraphon object
    """
    from pygraphon.utils.graph_utils import edge_density_across_cut, closed_triangle_to_square, subdivisions
    from bisect import bisect_left
    new_bins = partition.values()
    subdivision = subdivisions(new_bins)

    triangle = [[edge_density_across_cut(graph_nx, new_bins[i], new_bins[j], networkx=True)
                 for i in xrange(j+1)] for j in xrange(len(new_bins))]
    wts = closed_triangle_to_square(triangle)
    g = lambda x, y: wts[bisect_left(subdivision, x),bisect_left(subdivision, y)]

    return GraphonFunctionEstimator(g, graph_samp)


