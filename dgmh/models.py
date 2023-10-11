import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv, HypergraphConv
from torch_geometric.nn import VGAE
from typing import Optional

EPS = 1e-15
MAX_LOGSTD = 10

class Decoder(nn.Module):

    def __init__(self, in_channels, *hidden_channels, activation=nn.LeakyReLU(), sigmoid=True, dropout=0.1):
        super(Decoder, self).__init__()
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
        self.sigmoid = sigmoid

    def forward(self, x):
        x = self.linear_0(x)
        for i in range(1, len(self.hidden_channels)):
            x = getattr(self, f"activation_{i - 1}")(x)
            x = getattr(self, f"linear_{i}")(x)
        if self.sigmoid:
            x = torch.sigmoid(x)
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

    def __init__(self, in_channels, *hidden_channels, hyperedges, activation=nn.LeakyReLU()):
        super(HGCNEncoder, self).__init__()
        if len(hidden_channels) == 0:
            hidden_channels = [in_channels]
        self.hconv_0 = HypergraphConv(in_channels, hidden_channels[0])
        if len(hidden_channels) > 1:
            self.activation_0 = activation
        for i in range(1, len(hidden_channels)):
            setattr(self, f"hconv_{i}", HypergraphConv(hidden_channels[i - 1], hidden_channels[i]))
            setattr(self, f"activation_{i}", activation)
        self.hconv_mu = HypergraphConv(hidden_channels[-1], hidden_channels[-1])
        self.hconv_logvar = HypergraphConv(hidden_channels[-1], hidden_channels[-1])
        self.hidden_channels = hidden_channels
        self.hyperedges = hyperedges

    def forward(self, x, edge_index):
        x = self.hconv_0(x, edge_index)
        for i in range(1, len(self.hidden_channels)):
            x = getattr(self, f"activation_{i - 1}")(x)
            x = getattr(self, f"hconv_{i}")(x, edge_index)
        mu = self.hconv_mu(x, edge_index)
        logvar = self.hconv_logvar(x, edge_index)
        mu = torch.stack([mu[torch.tensor(h) - 1].sum(dim=0) for h in self.hyperedges])
        logvar = torch.stack([logvar[torch.tensor(h) - 1].sum(dim=0) for h in self.hyperedges])
        return mu, logvar

class HyperedgeAutoEncoder(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(HyperedgeAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def recon_loss(self, X_, positive_nodes, negative_nodes):

        positive_loss = -torch.log(X_[positive_nodes] + EPS).mean()

        negative_loss = -torch.log(1 - X_[negative_nodes] + EPS).mean()

        reconstruction_loss = positive_loss + negative_loss

        return reconstruction_loss

    def forward(self, Z, edge_index):
        x = self.encode(Z, edge_index)
        x = self.decode(x)
        return x
    
    @property
    def mu(self) -> torch.Tensor:
        return self.__mu__
    
    @property
    def logstd(self) -> torch.Tensor:
        return self.__logstd__

    def decode(self, *args, **kwargs) -> torch.Tensor:
        return self.decoder(*args, **kwargs)

    def encode(self, *args, **kwargs) -> torch.Tensor:
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def reparametrize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def kl_loss(self, mu: Optional[torch.Tensor] = None, logstd: Optional[torch.Tensor] = None) -> torch.Tensor:
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(max=MAX_LOGSTD)
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
    

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
