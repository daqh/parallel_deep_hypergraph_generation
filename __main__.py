import argparse
import torch
from torch import nn
from dgmh import DGMH
from dgmh.models import Decoder, HGCNEncoder, HyperedgeAutoEncoder, HyperedgeSizeDecisionModule
from dgmh.utils import load_dataset, compute_embeddings, compute_edge_index

def main(dataset_name, device):
    hyperedges = load_dataset(dataset_name)

    num_nodes = max([max(x) for x in hyperedges])
    print("Number of nodes:", num_nodes)
    num_hyperedges = len(hyperedges)
    print("Number of hyperedges:", num_hyperedges)

    X, y = compute_embeddings(hyperedges)
    edge_index = compute_edge_index(hyperedges)

    X = X.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)

    hyperedge_sizes = torch.tensor([len(x) for x in hyperedges], device=device)
    hyperedge_sizes = hyperedge_sizes - 1 # Riduciamo di uno perché non ci sono hyperedge di dimensione 0
    hyperedge_sizes = nn.functional.one_hot(hyperedge_sizes).type(torch.float32)
    max_hyperedge_size = hyperedge_sizes.shape[1]

    print("Max hyperedge size:", max_hyperedge_size)

    print()
    autoencoder = HyperedgeAutoEncoder(HGCNEncoder(num_nodes, 64, 32, hyperedges), Decoder(32, 2048, num_nodes))
    print(autoencoder)

    hsdm = HyperedgeSizeDecisionModule(num_nodes, 32, max_hyperedge_size)
    print()
    print(hsdm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Hyperedge Autoencoder.")
    parser.add_argument("--dataset", type=str, default="email-Eu", help="Dataset to use.")
    parser.add_argument("--device", default="cpu", type=str, help="Device to use.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for.")
    args = parser.parse_args()
    main(args.dataset, args.device)
