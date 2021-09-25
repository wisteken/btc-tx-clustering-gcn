import os
import torch
import argparse
import numpy as np
import configparser
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from logger import Logger
from utils import EllipticDataset
from model import ClusteringModel

config = configparser.ConfigParser()
config.read('./config.ini')
seed = config['DEFAULT']['seed']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(seed)
if device == 'cuda':
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

n_features = 165
n_classes = 2


def eval():
    # logger
    logger = Logger(name='eval_cls')

    # load config
    th_timestep = int(config['CLUSTERING']['th_timestep'])

    # load datasets
    datasets = EllipticDataset()

    # load trained model
    trained_model_path = '../models/classifier_weights.pth'
    if not os.path.exists(path=trained_model_path):
        raise FileNotFoundError('Trained model is not found. Please execute train classifier before.')
    model = ClusteringModel(n_features, n_classes).to(device)
    model.double()
    model.load_state_dict(torch.load(trained_model_path, map_location=device))

    # eval model performance
    accuracies = []
    for timestep in range(th_timestep):
        data = datasets[timestep].to(device)
        with torch.no_grad():
            _, out = model(data)
            y_data = data.y.cpu().numpy()
            ids = np.where(y_data != 2)
            y_pred = out.max(1)[1].cpu().numpy()
            acc = accuracy_score(y_pred[ids], y_data[ids])
            accuracies.append(acc)
            logger.info(f'timestep {timestep + 1}/{th_timestep} | eval_acc: {acc: .3f}')
    logger.info(f'average evaluation accuracy: {sum(accuracies)/len(accuracies): .3f}')


def run():
    # logger
    logger = Logger(name='run_cls')

    # load config
    th_timestep = int(config['CLUSTERING']['th_timestep'])

    # load datasets
    datasets = EllipticDataset()
    n_timestep = len(datasets)

    # load trained model
    trained_model_path = '../models/classifier_weights.pth'
    if not os.path.exists(path=trained_model_path):
        raise FileNotFoundError('Trained model is not found. Please execute train classifier before.')
    model = ClusteringModel(n_features, n_classes).to(device)
    model.double()
    model.load_state_dict(torch.load(trained_model_path, map_location=device))

    # clustering txs
    accuracies = []
    for timestep in range(th_timestep + 1, n_timestep):
        data = datasets[timestep].to(device)
        with torch.no_grad():
            embedded, out = model(data)
            y_data = data.y.cpu().numpy()
            ids = np.where(y_data != 2)
            y_pred = out.max(1)[1].cpu().numpy()
            acc = accuracy_score(y_pred[ids], y_data[ids])
            accuracies.append(acc)
            logger.info(f'timestep {timestep + 1}/{th_timestep} | accuracy: {acc: .3f}')

            plot(timestep, out.cpu().numpy(), y_data, True)
    logger.info(f"average accuracy: {sum(accuracies)/len(accuracies): .3f}")


def plot(timestep, x_embedded, y_data, is_limit=False):
    licit_ids = np.where(y_data == 0)
    illicit_ids = np.where(y_data == 1)
    unknown_ids = np.where(y_data == 2)
    print(len(licit_ids[0]), len(illicit_ids[0]), len(unknown_ids[0]))
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('licit')
    plt.scatter(x_embedded[licit_ids][:, 0:1], x_embedded[licit_ids][:, 1:2], c="blue", alpha=0.1)
    if is_limit:
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
    plt.subplot(1, 3, 2)
    plt.title('illicit')
    plt.scatter(x_embedded[illicit_ids][:, 0:1], x_embedded[illicit_ids][:, 1:2], c="red", alpha=0.1)
    if is_limit:
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
    plt.subplot(1, 3, 3)
    plt.title('unknown')
    plt.scatter(x_embedded[unknown_ids][:, 0:1], x_embedded[unknown_ids][:, 1:2], c="green", alpha=0.1)
    if is_limit:
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
    plt.savefig(f'../results/ts{timestep}_embedded.png', bbox_inches='tight')  # , transparent=True)):


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='clustering')
    parser.add_argument('--eval', action='store_true', help='is eval mode')
    args = parser.parse_args()
    if args.eval:
        eval()
    else:
        run()
