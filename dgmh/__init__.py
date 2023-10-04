import torch
from torch import nn
from multiprocessing.pool import Pool
from time import time

def generate_hyperedge(decoder: nn.Module, F: nn.Module, A: nn.Module):
    def wrapper(z):
        X_i = decoder(z) # X_i = f(F(Z_i))
        k = torch.argmax(A(X_i.view(1, -1)), dim=1).item() + 1
        print(f'Size k of the hyperedge: {k} - {time()}', end='\r')
        theta_i = torch.softmax(X_i, dim=0)
        hyperedge = []
        for j in range(k):
            e_ij = torch.distributions.Categorical(theta_i).sample()
            hyperedge.append(e_ij.item())
        return hyperedge
    return wrapper

def generate_hyperedge(args):
    decoder, F, A, z = args
    X_i = decoder(z) # X_i = f(F(Z_i))
    k = torch.argmax(A(X_i.view(1, -1)), dim=1).item() + 1
    print(f'Size k of the hyperedge: {k} - {time()}', end='\r')
    theta_i = torch.softmax(X_i, dim=0)
    hyperedge = []
    for j in range(k):
        e_ij = torch.distributions.Categorical(theta_i).sample()
        hyperedge.append(e_ij.item())
    return hyperedge

def DGMH(decoder: nn.Module, F: nn.Module, A:nn.Module, Z: torch.Tensor):
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
    args = [(decoder, F, A, z) for z in Z]
    with Pool(3) as pool:
        for hyperedge in pool.map(generate_hyperedge, args):
            hyperedges.append(hyperedge)
    # Remove duplicate nodes
    hyperedges = [sorted(list(set(hyperedge))) for hyperedge in hyperedges]
    return hyperedges
