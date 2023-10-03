import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv, HypergraphConv
from torch_geometric.nn import VGAE

EPS = 1e-15

class Decoder(nn.Module):
    '''
    A simple decoder module for the Hyperedge Autoencoder.
    '''

    def __init__(self, in_features: int, hidden_features: int, out_features: int, sigmoid: bool = True):
        super(Decoder, self).__init__()
        self.linear_1 = nn.Linear(in_features, hidden_features)
        self.linear_2 = nn.Linear(hidden_features, hidden_features)
        self.linear_3 = nn.Linear(hidden_features, out_features)
        self.sigmoid = sigmoid

    def forward(self, x):
        x = self.linear_1(x)
        x = nn.functional.leaky_relu(x)
        x = self.linear_2(x)
        x = nn.functional.leaky_relu(x)
        x = self.linear_3(x)
        if self.sigmoid:
            x = nn.functional.sigmoid(x)
        return x

class GCNEncoder(nn.Module):
    '''
    A Graph Convolutional Encoder.
    '''

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(GCNEncoder, self).__init__()
        self.conv_1 = GCNConv(in_channels, hidden_channels)
        self.conv_2 = GCNConv(hidden_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv_1(x, edge_index)
        x = torch.relu(x)
        x = self.conv_2(x, edge_index)
        x = torch.relu(x)
        x = self.conv_3(x, edge_index)
        return x

class HGCNEncoder(nn.Module):
    '''
    A Hypergraph Convolutional Encoder.
    '''

    def __init__(self, input_size: int, hidden_size: int, output_size: int, hyperedges):
        super(HGCNEncoder, self).__init__()
        self.hconv_0 = HypergraphConv(input_size, hidden_size)
        self.hconv_1 = HypergraphConv(hidden_size, hidden_size)
        self.hconv_2 = HypergraphConv(hidden_size, output_size)
        self.hyperedges = hyperedges

    def forward(self, x, edge_index):
        x = self.hconv_0(x, edge_index)
        x = F.leaky_relu(x)
        x = self.hconv_1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.hconv_2(x, edge_index)
        x = torch.stack([x[torch.tensor(h) - 1].sum(dim=0) for h in self.hyperedges])
        return x

class HyperedgeAutoEncoder(nn.Module):
    '''
    A Hyperedge Autoencoder.
    '''

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(HyperedgeAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, Z, edge_index):
        x = self.encode(Z, edge_index)
        x = self.decode(x)
        return x

    def encode(self, Z, edge_index):
        x = self.encoder(Z, edge_index)
        return x

    def decode(self, Z):
        x = self.decoder(Z)
        return x

    def recon_loss(self, X_, positive_nodes, negative_nodes):

        positive_loss = -torch.log(X_[positive_nodes] + EPS).mean()

        negative_loss = -torch.log(1 - X_[negative_nodes] + EPS).mean()

        reconstruction_loss = positive_loss + negative_loss

        return reconstruction_loss

class HyperedgeSizeDecisionModule(nn.Module):
    '''
    A Hyperedge Size Decision Module.
    '''

    def __init__(self, in_channels, *hidden_channels, activation=nn.ReLU(), softmax=True, dropout=0.1):
        super(HyperedgeSizeDecisionModule, self).__init__()
        self.dropout = nn.Dropout(dropout)
        if len(hidden_channels) == 0:
            hidden_channels = [in_channels]
        self.linear_0 = nn.Linear(in_channels, hidden_channels[0])
        if len(hidden_channels) > 1:
            self.activation_0 = activation
        for i in range(1, len(hidden_channels)):
            setattr(self, f"linear_{i}", nn.Linear(hidden_channels[i - 1], hidden_channels[i]))
            setattr(self, f"activation_{i}", activation)
        self.hidden_channels = hidden_channels
        self.softmax = softmax

    def forward(self, x):
        x = self.linear_0(x)
        for i in range(1, len(self.hidden_channels)):
            x = getattr(self, f"activation_{i - 1}")(x)
            x = getattr(self, f"linear_{i}")(x)
        if self.softmax:
            x = torch.softmax(x, dim=1)
        return x
