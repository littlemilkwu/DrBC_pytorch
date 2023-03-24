import torch
from torch import nn
from torch.nn import Module, Linear, Parameter, GRUCell, Sequential, LeakyReLU, ReLU, functional as t_F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        # self.lin = torch.nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        # self.lin.reset_parameters()
        # self.bias.data.zero_()
        pass
    
    def forward(self, x, edge_index):
        # x = self.lin(x)
        row, col = edge_index
        rc = torch.cat([row, col], axis=0)
        deg = degree(rc, x.shape[0], dtype=x.dtype)
        deg = torch.add(deg, 1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class DrBC(Module):
    def __init__(self, embedding_size=128, depth=6):
        super(DrBC, self).__init__()
        self.embedding_size = embedding_size
        self.depth = depth
        # encoder
        self.linear0 = Linear(3, self.embedding_size)
        self.gcn = GCNConv(self.embedding_size, self.embedding_size)
        self.gru = GRUCell(self.embedding_size, self.embedding_size)

        # decoder
        self.mlp = Sequential(
            Linear(self.embedding_size, self.embedding_size // 2),
            LeakyReLU(),
            Linear(self.embedding_size // 2, 1)
        )
        

    def forward(self, X, edge_index):
        all_h = []
        h = self.linear0(X)
        h = LeakyReLU()(h)
        # h = t_F.normalize(h, p=2, dim=1) # l2-norm

        # GRUCell
        # neighborhood aggregation
        # GRUCell
        for i in range(self.depth-1):
            # neighborhood aggregation
            h_aggre = self.gcn(h, edge_index)
            h = self.gru(h_aggre, h)
            # h = t_F.normalize(h, p=2, dim=-1) # l2-norm
            all_h.append(torch.unsqueeze(h, dim=0))
        # max pooling
        all_h = torch.cat(all_h, dim=0)
        h_max = torch.max(all_h, dim=0).values
        

        # Decoder
        out = self.mlp(h_max)
        out = torch.squeeze(out)
        return out

if __name__ == "__main__":
    pass