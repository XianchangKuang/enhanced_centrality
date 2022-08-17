# A collection of functions


def bmi(mass, height):
	bmi_formula = mass//height**2

	if bmi_formula < 18.5: 
		print('Body mass index is ==> ', bmi_formula)
		print('under')

	elif bmi_formula > 23:
		print('Body mass index is ==> ', bmi_formula)
		print('over')

	elif bmi_formula > 30:
		print('Body mass index is ==> ', bmi_formula)
		print('obese')

	return bmi_formula

# From Our Last Project:

%matplotlib inline
%precision 16
import json
import numpy
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.centrality.closeness import closeness_centrality
from networkx.algorithms.centrality.degree_alg import degree_centrality
import networkx.classes.graph
import scipy.linalg as sl
import random

# SOME USEFUL TOOLS:

def random_graph(N,P,a,b):
    """
    This function generates one random graph, both weighted and unweighted version.
    
    parameters:
    N: integer, number of nodes
    P: integer, number of edges
    a: integer, the lower bound of weight
    b: integer, the upper bound of weight
    
    return:
    G: weighted version of the graph
    G2: unweighted version of the graph
    """
    G = nx.fast_gnp_random_graph(N,P)
    G2 = G.copy()
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = random.randint(a,b)
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > (a+b)*0.5]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= (a+b)*0.5]
    pos = nx.spring_layout(G, seed=7)
    nx.draw_networkx_nodes(G,pos)
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    labels = {}
    for i in range(N):
        labels[i] = i
    nx.draw_networkx_labels(G, pos, labels, font_size=22, font_color="whitesmoke")
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return G,G2

def get_traditional(G):
    """
    This function calculates the closeness and degree centrality scores of a given graph
    
    parameters:
    G: the graph in question
    
    return:
    close: closeness centrality scores of the graph
    degree: degree centrality scores of the graph
    """
    c = closeness_centrality(G)
    close = []
    d = degree_centrality(G)
    degree = []
    for i in range(G.number_of_nodes()):
        close.append(c.get(i))
        degree.append(d.get(i))
    return close,degree

def index(rank):
    """
    This function calculates the centrality ranking of a given list of centrality score
    This function is modified from scipy.stats.rankdata
    
    parameters:
    a: list, centrality scores
    
    return:
    rank: the centrality ranking based on the centrality scores
    """

    arr = numpy.ravel(numpy.asarray(rank))
    algo = 'mergesort'
    sorter = numpy.argsort(arr, kind=algo)

    inv = numpy.empty(sorter.size, dtype=numpy.intp)
    inv[sorter] = numpy.arange(sorter.size, dtype=numpy.intp)
    
    arr = arr[sorter]
    obs = numpy.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]
    
    count = numpy.r_[numpy.nonzero(obs)[0], len(obs)]
    
    rank_inverse = count[dense]
    rank = []
    for i in rank_inverse:
        rank.append(max(rank_inverse) + 1 - i)
    
    return rank

def checklist(lst):
    """
    This function checks if every number in a given list is unique.
    
    parameters:
    lst: list, the list that you would like to check (centrality ranking)
    
    return:
    tie: True if there is any tie in the list, False if the ranking is entirely unique
    """
    
    tie = False
    
    for i in lst:
        equal = 0
        for j in lst:
            if i == j:
                equal += 1
        if equal > 1:
            tie = True
            return tie
              
def knbrs(G, start, k):
    """
    This function gives the kth order neighbor of a given point in a certain graph.
    (This one comes from Aric on stackoverflow, the link is here: https://stackoverflow.com/questions/18393842/k-th-order-neighbors-in-graph-python-networkx)
    
    parameters:
    G: the graph in question
    
    start: the point whose kth neighbors you would like to find
    
    k: the order of the neighbors that you would like to find
    
    return:
    nbrs: the neighbors found
    """
    
    nbrs = set([start])
    for l in range(k):
        nbrs = set((nbr for n in nbrs for nbr in G[n]))
    return nbrs

# ACTUAL STUFF:


def degree_centrality_enhanced_smart(G):
    """
    This function calculates the newly-defined centrality scores (based on degree centrality) of a given graph.
    
    Parameter:
    G: the graph in question
    
    Return:
    degree: centrality score of the graph
    """
    
    if len(G) <= 1:
        raise ValueError
    d = degree_centrality(G)
    step = 1
    degree = [0, 0]
    
    while checklist(degree):
        for i in range(len(G)):
            ne = knbrs(G, i, step)
            addi = 0
            for j in ne:
                addi += d.get(j)/(2*len(knbrs(G, i, step)))
            d[i] = d.get(i)+addi    
        degree = []
        for i in range(G.number_of_nodes()):
            degree.append(d.get(i))
        step += 1
    
    return degree
