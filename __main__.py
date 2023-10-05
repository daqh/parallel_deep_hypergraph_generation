import argparse
import torch
from torch import nn
from dgmh import DGMH
from dgmh.models import Decoder, HGCNEncoder, HyperedgeAutoEncoder, HyperedgeSizeDecisionModule
from dgmh.utils import load_dataset, compute_embeddings, compute_edge_index
from time import time

def evaluate_speed(autoencoder: HyperedgeAutoEncoder, hsdm, F, processes, num_experiments=5):
    result = []
    for j in range(num_experiments):
        print(f"Running experiment {j + 1} for process count {processes}")
        Z = torch.randn(25000, 32)
        begin = time()
        DGMH(autoencoder.decoder, F, hsdm, Z, processes)
        end = time() - begin
        result.append(end)
    print(result)
    print(f"Average time: {sum(result) / len(result)}")
    print(f"Max time: {max(result)}")
    print(f"Min time: {min(result)}")

def main(dataset_name, device, action, processes):
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

    autoencoder = HyperedgeAutoEncoder(HGCNEncoder(num_nodes, 64, 32, hyperedges), Decoder(32, 2048, num_nodes))
    if action == "train":
        autoencoder.load_state_dict(torch.load(f'models/{dataset_name}.autoencoder.pth'));
    autoencoder.eval()

    hsdm = HyperedgeSizeDecisionModule(num_nodes, 512, 512, max_hyperedge_size)
    if action == "train":
        pass
        # autoencoder.load_state_dict(torch.load(f'models/{dataset_name}.hyperedge_size_decision_module.pth'));
    hsdm.eval()


    F = nn.Linear(32, 32).to(device)

    if action == "generate":
        print()
        print(f"Number of processes: {processes}")

        Z = torch.randn(25000, 32)

        begin = time()
        DGMH(autoencoder.decoder, F, hsdm, Z, processes)
        print(f'Time: {time() - begin}')

    if action == "evaluate_speed":
        hsdm.share_memory()
        autoencoder.share_memory()
        evaluate_speed(autoencoder, hsdm, F, processes)

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_descriptor')
    torch.multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description="Train a Hyperedge Autoencoder.")
    parser.add_argument("--dataset", type=str, default="email-Eu", help="Dataset to use.")
    parser.add_argument("--device", default="cpu", type=str, help="Device to use.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for.")
    parser.add_argument("--action", type=str, default="train", help="Action to perform.")
    parser.add_argument("--processes", type=int, default=1, help="Number of processes to use.")
    args = parser.parse_args()
    main(args.dataset, args.device, args.action, args.processes)
