import os
import torch
import random
import argparse
import numpy as np
import configparser
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.nn import ARGVA
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score

from utils import Logger, load_data
from model import Encoder, Discriminator

config = configparser.ConfigParser()
config.read('./config.ini')
seed = config['DEFAULT']['seed']
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
random.seed(seed)

n_features = 6
n_classes = 1


def train():
    # logger
    # logger = Logger(name='train')

    # load datasets
    features, edges = load_data()
    node_features = torch.tensor(features, dtype=torch.double, device=device)
    edge_index = torch.tensor(edges, dtype=torch.long, device=device)
    train_data = Data(x=node_features, edge_index=edge_index)

    # load model
    encoder = Encoder(n_features, hidden_channels=32, out_channels=32)
    discriminator = Discriminator(in_channels=32, hidden_channels=64, out_channels=32)
    model = ARGVA(encoder, discriminator).to(device)
    model.double()

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.005)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    # train model
    model.train()
    n_epochs = 100
    for epoch in range(n_epochs):
        encoder_optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)

        # We optimize the discriminator more frequently than the encoder.
        for i in range(10):
            discriminator_optimizer.zero_grad()
            discriminator_loss = model.discriminator_loss(z)
            discriminator_loss.backward()
            discriminator_optimizer.step()

        loss = model.recon_loss(z, train_data.edge_index)
        loss = loss + model.reg_loss(z)
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
        loss.backward()
        encoder_optimizer.step()
        print(f'epoch: {epoch}/{n_epochs} loss: {loss}')
        torch.save(model.state_dict(), f'../models/weights-{epoch}.pth')


@torch.no_grad()
def plot():
    features, edges = load_data()
    node_features = torch.tensor(features, dtype=torch.double, device=device)
    edge_index = torch.tensor(edges, dtype=torch.long, device=device)
    data = Data(x=node_features, edge_index=edge_index)

    encoder = Encoder(n_features, hidden_channels=32, out_channels=32)
    discriminator = Discriminator(in_channels=32, hidden_channels=64, out_channels=32)
    model = ARGVA(encoder, discriminator).to(device)
    model.double()
    trained_model_path = '../models/weights.pth'
    model.load_state_dict(torch.load(trained_model_path, map_location=device))

    model.eval()
    z = model.encode(data.x, data.edge_index)
    z = z.cpu().numpy()
    z_selected = random.sample(list(range(z.shape[0])), 30000)
    z_selected = TSNE(n_components=2).fit_transform(np.array(z_selected).reshape(-1, 1))
    # y = data.y.cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.scatter(z_selected[:, 0], z_selected[:, 1])
    # for i in range(dataset.num_classes):
    #     plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    plt.axis('off')
    plt.savefig('../results/plot.png')


if __name__ == '__main__':
    # train()
    plot()
