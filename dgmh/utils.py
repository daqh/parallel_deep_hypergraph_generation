import torch
from torch import nn

def load_dataset(dataset_name: str) -> list[list[int]]:
    with open(f"data/{dataset_name}.unique-hyperedges.txt", "r") as f:
        hyperedges = [list(map(int, x.split())) for x in f.readlines()]
    return hyperedges

def compute_edge_index(hyperedges: list[int]):
    edge_index = [[], []]
    edge_index = [[], []]
    for i, hyperedge in enumerate(hyperedges):
        for node in hyperedge:
            edge_index[0].append(node - 1)
            edge_index[1].append(i)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index

def compute_embeddings(hyperedges: list[int]):
    num_nodes = max([max(x) for x in hyperedges])
    num_hyperedges = len(hyperedges)
    hyperedge_sizes = torch.tensor([len(x) for x in hyperedges])
    hyperedge_sizes = hyperedge_sizes - 1 # Riduciamo di uno perché non ci sono hyperedge di dimensione 0
    hyperedge_sizes_one_hot = nn.functional.one_hot(hyperedge_sizes).type(torch.float32)
    # Calculate hyperedge embeddings
    X = torch.zeros(num_hyperedges, num_nodes)
    for i in range(len(hyperedges)):
        indexes = torch.tensor(hyperedges[i]) - 1
        X[i, indexes] = 1
    return (X, hyperedge_sizes_one_hot)
