def sim_st(graph_nx, s, t, w):
    setw = set(graph_nx.neighbors(w))
    lensw = len(set(graph_nx.neighbors(s)).intersection(setw))
    lentw = len(set(graph_nx.neighbors(t)).intersection(setw))
    return abs(lensw - lentw)

def d_sim(graph_nx, s,t):
    nodes = graph_nx.nodes()
    n = len(nodes)
    total = 0
    for w in nodes:
        total += sim_st(graph_nx, s, t, w)
    return total / float(n**2)


def d_neigh(graph_nx, s, t):
    sets = set(graph_nx.neighbors(s))
    sett = set(graph_nx.neighbors(t))
    return len(sets.symmetric_difference(sett)) / float(len(graph_nx))
