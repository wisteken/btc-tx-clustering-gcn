import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Classifier(torch.nn.Module):
    def __init__(self, n_features, n_classes, n_hidden=128):
        super(Classifier, self).__init__()
        self.conv1 = GCNConv(n_features, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
