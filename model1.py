import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        # Step 3: Compute normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # Step 4-6: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 6: Return new node embeddings.
        return aggr_out

class DrBC(nn.Module):
    def __init__(self,):
        super(DrBC, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(3, 128)
        self.relu = nn.LeakyReLU()

        self.gcn1 = GCNConv(128, 128)
        self.gru1 = nn.GRU(128, 128)

        self.gcn2 = GCNConv(128, 128)
        self.gru2 = nn.GRU(128, 128)

        self.gcn3 = GCNConv(128, 128)
        self.gru3 = nn.GRU(128, 128)

        self.gcn4 = GCNConv(128, 128)
        self.gru4 = nn.GRU(128, 128)

        self.gcn5 = GCNConv(128, 128)
        self.gru5 = nn.GRU(128, 128)

        # Decoder
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, edge_index):
        x = self.fc1(x)
        x = self.relu(x)        
        x_n = self.gcn1(x, edge_index)
        x1, _ = self.gru1(x_n.view(1, *x_n.shape), x.view(1, *x.shape))
        x_n = self.gcn2(x1[0], edge_index)
        x2, _ = self.gru2(x_n.view(1, *x_n.shape), x1)
        x_n = self.gcn3(x2[0], edge_index)
        x3, _ = self.gru3(x_n.view(1, *x_n.shape), x2)
        x_n = self.gcn4(x3[0], edge_index)
        x4, _ = self.gru4(x_n.view(1, *x_n.shape), x3)
        x_n = self.gcn5(x4[0], edge_index)
        x5, _ = self.gru5(x_n.view(1, *x_n.shape), x4)
        
        # max
        l = [x1[0],x2[0],x3[0],x4[0],x5[0]]
        l = torch.stack(l)
        x = torch.max(l, dim=0).values
        # l = torch.tensor(l) 

        # decoder
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return torch.squeeze(x)

# forward 不使用 unsqueeze把 cat 改成 stack -> 沒差
# row col 改掉

# GCONV 是對的
# decoder 是對的
# encoder前半段 