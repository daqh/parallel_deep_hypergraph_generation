import torch
from torch import nn
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

def powerfit(x, y):
    """line fitting on log-log scale"""
    k, m = np.polyfit(np.log(x[:]), np.log(y), 1)
    return np.exp(m) * x**(k)

def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, r_value**2

def logAnalysis(xaxis, yaxis, filename, xname="x axis", yname="y axis", title="title"):
    regression = powerfit(xaxis, yaxis)

    plt.figure()
    plt.loglog(xaxis, yaxis, 'ro')
    plt.plot(xaxis, regression)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(title)
    plt.savefig(filename, dpi=300)

    #slope, r2 = rsquared(np.log(xaxis), np.log(yaxis))
    #print("The exponent is", slope)
    #print("R^2 =", r2)

def load_dataset(dataset_name: str) -> list[list[int]]:
    with open(f"data/{dataset_name}.unique-hyperedges.txt", "r") as f:
        hyperedges = [list(map(int, x.split())) for x in f.readlines()]
    return hyperedges

def compute_edge_index(hyperedges: list[int]):
    edge_index = [[], []]
    for i, hyperedge in enumerate(hyperedges):
        for node in hyperedge:
            edge_index[0].append(node - 1)
            edge_index[1].append(i)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index

def compute_embeddings(hyperedges: list[int]):
    num_nodes = max([max(x) for x in hyperedges])
    hyperedge_sizes = torch.tensor([len(x) for x in hyperedges])
    hyperedge_sizes = hyperedge_sizes - 1 # Riduciamo di uno perché non ci sono hyperedge di dimensione 0
    hyperedge_sizes_one_hot = nn.functional.one_hot(hyperedge_sizes).type(torch.float32)
    # Calculate hyperedge embeddings
    hyperedges_one_hot = -torch.ones(len(hyperedges), num_nodes)
    for i in range(len(hyperedges)):
        indexes = torch.tensor(hyperedges[i]) - 1
        hyperedges_one_hot[i, indexes] = 1
    X = torch.eye(num_nodes, dtype=torch.float32)
    return (X, hyperedges_one_hot, hyperedge_sizes_one_hot)
