import torch
from torch import nn
from multiprocessing.pool import Pool
from time import time
import os, psutil


def generate_hyperedge(args):
    decoder, F, A, z = args
    X_i = decoder(z) # X_i = f(F(Z_i))
    k = torch.argmax(A(X_i.view(1, -1)), dim=1).item() + 1
    # print(f'Size k of the hyperedge: {k} - {time()}', end='\r')
    theta_i = torch.softmax(X_i, dim=0)
    cat = torch.distributions.Categorical(theta_i)
    hyperedge = list(cat.sample((k,)).detach().cpu().numpy())
    return hyperedge

def DGMH(decoder: nn.Module, F: nn.Module, A:nn.Module, Z: torch.Tensor, processes: int = 1):
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
    max_memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    if processes == 1:
        for z in Z:
            max_memory_usage = max(max_memory_usage, psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
            hyperedge = generate_hyperedge((decoder, F, A, z))
            hyperedges.append(hyperedge)
    else:
        args = [(decoder, F, A, z) for z in Z]
        with Pool(processes) as pool:
            for hyperedge in pool.map(generate_hyperedge, args):
                max_memory_usage = max(max_memory_usage, psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
                hyperedges.append(hyperedge)
    print(f'Max memory usage: {max_memory_usage} MB')
    # Remove duplicate nodes
    hyperedges = [sorted(list(set(hyperedge))) for hyperedge in hyperedges]
    return hyperedges
