import argparse
import torch
from torch import nn
from dgmh.utils import load_dataset
from dgmh.analysis import analyze_k_decomposition

def main(dataset_name: str, k: int):
    hyperedges = load_dataset(dataset_name)

    num_nodes = max([max(x) for x in hyperedges])
    num_hyperedges = len(hyperedges)

    hyperedge_sizes = torch.tensor([len(x) for x in hyperedges])
    hyperedge_sizes = hyperedge_sizes - 1 # Riduciamo di uno perché non ci sono hyperedge di dimensione 0
    hyperedge_sizes = nn.functional.one_hot(hyperedge_sizes).type(torch.float32)
    max_hyperedge_size = hyperedge_sizes.shape[1]

    print("Number of nodes:", num_nodes)
    print("Number of hyperedges:", num_hyperedges)
    print("Max hyperedge size:", max_hyperedge_size)

    if not k:
        raise Exception("k is required for analysis")
    print()
    with open(f"generated/{dataset_name}.generated.txt", "r") as f:
        hyperedges = [list(map(int, x.split())) for x in f.readlines()]
    print("Number of generated hyperedges:", len(hyperedges))
    analyze_k_decomposition(hyperedges, k, dataset_name)

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_descriptor')
    torch.multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description="Train a Hyperedge Autoencoder.")
    parser.add_argument("--dataset", type=str, help="Dataset to use.", required=True)
    parser.add_argument("-k", type=int, help="Level of decomposition.", required=True)
    args = parser.parse_args()

    dataset_name, k = args.dataset, args.k

    main(dataset_name, k)
