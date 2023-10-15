import argparse
import torch
from torch import nn
from dgmh.models import Decoder, HGCNEncoder, HyperedgeAutoEncoder, HyperedgeSizeDecisionModule
from dgmh.utils import load_dataset
from dgmh.training import train_models

def main(dataset_name, device, epochs: int):
    hyperedges = load_dataset(dataset_name)

    num_nodes = max([max(x) for x in hyperedges])
    num_hyperedges = len(hyperedges)

    hyperedge_sizes = torch.tensor([len(x) for x in hyperedges], device=device)
    hyperedge_sizes = hyperedge_sizes - 1 # Riduciamo di uno perché non ci sono hyperedge di dimensione 0
    hyperedge_sizes = nn.functional.one_hot(hyperedge_sizes).type(torch.float32)
    max_hyperedge_size = hyperedge_sizes.shape[1]

    autoencoder = HyperedgeAutoEncoder(HGCNEncoder(num_nodes, 2048, 1024, 512, 256, 64, hyperedges=hyperedges), Decoder(64, 4096, 2048, 2048, 4096, num_nodes, sigmoid=True, dropout=0.1)).to(device)

    hsdm = HyperedgeSizeDecisionModule(num_nodes, 512, 512, max_hyperedge_size)

    train_models(autoencoder, hsdm, hyperedges, epochs, device)
    torch.save(autoencoder, f"models/{dataset_name}.autoencoder.pt")
    torch.save(hsdm.state_dict(), f"models/{dataset_name}.hyperedge_size_decision_module.pt")

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_descriptor')
    torch.multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description="Train a Hyperedge Autoencoder.")
    parser.add_argument("--dataset", type=str, help="Dataset to use.", required=True)
    parser.add_argument("--device", default="cpu", type=str, help="Device to use.")
    parser.add_argument("--epochs", type=int, default=250, help="Number of epochs to train for.")
    args = parser.parse_args()
    main(args.dataset, args.device, epochs=args.epochs)
