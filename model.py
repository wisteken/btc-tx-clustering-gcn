import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = GCNConv(165, 128)
        self.conv2 = GCNConv(128, 1)

    def forward(self, data, adj=None):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)

        return F.sigmoid(x)
