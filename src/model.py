import torch
from torch import nn
from torch_geometric.nn import GATConv


class GraphClassifier(torch.nn.Module):
    def __init__(self, n_features, n_classes, n_hidden1=128, n_hidden2=16):
        super(GraphClassifier, self).__init__()
        self.conv = GATConv(n_features, n_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(2 * n_features, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_embedded = self.conv(x, edge_index)
        x_e = self.relu(x_embedded)
        x_e = self.dropout(x_e)
        x = torch.cat((x, x_e), 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x_embedded, self.sigmoid(x)


class MLPClassifier(torch.nn.Module):
    def __init__(self, n_features, n_classes, n_hidden1=128, n_hidden2=16):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x = data.x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x, self.sigmoid(x)
