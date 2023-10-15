import argparse
import torch
from torch import nn
from dgmh import DGMH
from dgmh.models import Decoder, HGCNEncoder, HyperedgeAutoEncoder, HyperedgeSizeDecisionModule
from dgmh.utils import load_dataset
from dgmh.utils import compute_embeddings, compute_edge_index
from time import time

def main(n, dataset_name: str, processes: int, device: str):

    hyperedges = load_dataset(dataset_name)

    num_nodes = max([max(x) for x in hyperedges])
    num_hyperedges = len(hyperedges)

    hyperedge_sizes = torch.tensor([len(x) for x in hyperedges], device=device)
    hyperedge_sizes = hyperedge_sizes - 1 # Riduciamo di uno perché non ci sono hyperedge di dimensione 0
    hyperedge_sizes = nn.functional.one_hot(hyperedge_sizes).type(torch.float32)
    max_hyperedge_size = hyperedge_sizes.shape[1]

    autoencoder: HyperedgeAutoEncoder = torch.load(f'models/{dataset_name}.autoencoder.pt')
    autoencoder.eval()

    hsdm = HyperedgeSizeDecisionModule(num_nodes, 512, 512, max_hyperedge_size)
    hsdm.load_state_dict(torch.load(f'models/{dataset_name}.hyperedge_size_decision_module.pt'));
    hsdm.eval()

    print()
    print(f"Number of processes: {processes}")

    nodes_one_hot, X, y = compute_embeddings(hyperedges)
    edge_index = compute_edge_index(hyperedges)

    X = X.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)

    Z = torch.randn(n, 64) * autoencoder.logstd + autoencoder.mu

    begin = time()
    hyperedges = DGMH(autoencoder.decoder, hsdm, Z, processes)
    # Remove duplicate nodes
    hyperedges = [sorted(list(set(hyperedge))) for hyperedge in hyperedges]
    hyperedges = [list(set(g)) for g in hyperedges]
    # Remove duplicate hyperedges
    hyperedges = [list(g) for g in list(set([tuple(g) for g in hyperedges]))]
    print(f'Time: {time() - begin}')

    with open(f"generated/{dataset_name}.generated.txt", "w") as f:
        for hyperedge in hyperedges:
            f.write(" ".join(map(str, hyperedge)) + "\n")

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_descriptor')
    torch.multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description="Generate a new hypergraph.")
    parser.add_argument("--dataset", type=str, help="Dataset to use.", required=True)
    parser.add_argument("--device", default="cpu", type=str, help="Device to use.")
    parser.add_argument("-n", type=int, help="Number of hyperedges.", required=True)
    parser.add_argument("--processes", type=int, default=1, help="Number of processes to use.")
    args = parser.parse_args()

    dataset_name, device, processes, n = args.dataset, args.device, args.processes, args.n

    main(n, dataset_name, processes, device)
