import torch
from torch import nn
from multiprocessing.pool import ThreadPool

def DGMH(decoder: nn.Module, F: nn.Module, A:nn.Module, Z: torch.Tensor, n: int):
    '''
    decoder: nn.Module
        Trained decoder module
    F: nn.Module
        Trained representation learning module
    A: nn.Module
        Trained hyperedge size decision module
    Z: torch.Tensor
        Encoded hyperedges
    n: int
        Number of hyperedges
    '''
    hyperedges = []
    for i in range(n):
        print(f'Hyperedge {i+1}')
        X_i = decoder(Z[i]) # X_i = f(F(Z_i))
        k = torch.argmax(A(X_i.view(1, -1)), dim=1).item() + 1
        print(f'Size k of the hyperedge: {k}')
        theta_i = torch.softmax(X_i, dim=0)
        hyperedge = []
        for j in range(k):
            e_ij = torch.distributions.Categorical(theta_i).sample()
            hyperedge.append(e_ij.item())
        hyperedges.append(hyperedge)
    # Remove duplicate nodes
    hyperedges = [sorted(list(set(hyperedge))) for hyperedge in hyperedges]
    return hyperedges
