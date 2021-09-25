import os
import sys
import torch
import argparse
import configparser
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

from logger import Logger
from utils import EllipticDataset
from model import Classifier

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


def train():
    # logger
    logger = Logger(name='train_clf')

    # load config
    th_timestep = int(config['CLASSIFIER']['th_timestep'])
    n_epochs = int(config['CLASSIFIER']['n_epochs'])

    # load datasets
    datasets = EllipticDataset(is_classification=True)

    # define model and parameters
    model = Classifier(n_features, n_classes).to(device)
    model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # , weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.DoubleTensor([0.7, 0.3]))

    # prepare validation data
    valid_data = datasets[th_timestep].to(device)

    # train model
    model.train()
    for timestep in range(th_timestep):
        logger.info(f"timestep {timestep + 1}/{th_timestep}")

        train_data = datasets[timestep].to(device)
        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad()
            train_out = model(train_data)
            train_loss = criterion(train_out.cpu(), train_data.y.cpu())
            train_acc = accuracy_score(train_data.y.detach().cpu().numpy(), train_out.max(1)[1].detach().cpu().numpy())
            train_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                with torch.no_grad():
                    valid_out = model(valid_data)
                    valid_loss = criterion(valid_out.cpu(), valid_data.y.cpu())
                    valid_acc = accuracy_score(
                        valid_data.y.detach().cpu().numpy(), valid_out.max(1)[1].detach().cpu().numpy())
                logger.info(
                    f"epoch {epoch}/{n_epochs} | train_loss: {train_loss: .3f} train_acc: {train_acc: .3f} "
                    f"| valid_loss: {valid_loss: .3f} valid_acc: {valid_acc: .3f}")
            else:
                logger.debug(f"epoch {epoch}/{n_epochs} | train_loss: {train_loss: .3f} train_acc: {train_acc}")

    torch.save(model.state_dict(), "../models/classifier_weights.pth")


def test():
    # logger
    logger = Logger(name='test_clf')

    # load config
    th_timestep = int(config['CLASSIFIER']['th_timestep'])

    # load datasets
    datasets = EllipticDataset(is_classification=True)
    n_timestep = len(datasets)

    # load trained model
    trained_model_path = '../models/classifier_weights.pth'
    if not os.path.exists(path=trained_model_path):
        raise FileNotFoundError('trained model is not found.')
    model = Classifier(n_features, n_classes).to(device)
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model.double()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.DoubleTensor([0.7, 0.3]))

    test_accuracies = []
    for timestep in range(th_timestep + 1, n_timestep):
        test_data = datasets[timestep].to(device)

        with torch.no_grad():
            test_out = model(test_data)
            test_loss = criterion(test_out.cpu(), test_data.y.cpu())
            test_acc = accuracy_score(test_data.y.detach().cpu(), test_out.max(1)[1].detach().cpu().numpy())
            test_accuracies.append(test_acc)
        logger.info(f"timestep {timestep + 1}/{n_timestep} | test_loss: {test_loss: .3f} test_acc: {test_acc: .3f}")
        logger.debug(
            f"{classification_report(test_data.y.detach().cpu(), test_out.max(1)[1].detach().cpu().numpy())}")
    logger.info(f"average test accuracy: {sum(test_accuracies)/len(test_accuracies): .3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="classification")
    parser.add_argument('--test', action="store_true", help='is test mode')
    args = parser.parse_args()
    if args.test:
        test()
    else:
        train()
