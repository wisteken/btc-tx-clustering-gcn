import os
import torch
import argparse
import configparser
from sklearn.metrics import roc_auc_score

from logger import Logger
from utils import GraphDataset, TabularDatasets
from model import GraphClassifier, MLPClassifier

config = configparser.ConfigParser()
config.read('./config.ini')
seed = config['DEFAULT']['seed']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

n_features = 165
n_classes = 1


def train(is_mlp=False):
    # logger
    if is_mlp:
        logger = Logger(name='train_mlp_clf')
    else:
        logger = Logger(name='train_gcn_clf')

    # load config
    th_timestep = int(config['CLASSIFIER']['th_timestep'])
    n_epochs = int(config['CLASSIFIER']['n_epochs'])

    # load datasets
    if is_mlp:
        datasets = TabularDatasets(is_clf=True)
    else:
        datasets = GraphDataset(is_clf=True)

    # define model and parameters
    if is_mlp:
        model = MLPClassifier(n_features, n_classes).to(device)
    else:
        model = GraphClassifier(n_features, n_classes).to(device)
    model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = torch.nn.BCELoss()

    # prepare validation data
    valid_data = datasets[th_timestep].to(device)

    # train model
    model.train()
    for timestep in range(th_timestep):
        logger.info(f"timestep {timestep + 1}/{th_timestep}")

        train_data = datasets[timestep].to(device)
        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad()
            _, train_out = model(train_data)
            train_out = train_out.reshape((train_data.x.shape[0]))
            train_loss = criterion(train_out, train_data.y)
            train_auc = roc_auc_score(train_data.y.detach().cpu().numpy(), train_out.detach().cpu().numpy())
            train_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                with torch.no_grad():
                    _, valid_out = model(valid_data)
                    valid_out = valid_out.reshape((valid_data.x.shape[0]))
                    valid_loss = criterion(valid_out, valid_data.y)
                    valid_auc = roc_auc_score(valid_data.y.detach().cpu().numpy(), valid_out.detach().cpu().numpy())
                logger.info(
                    f"epoch {epoch}/{n_epochs} | train_loss: {train_loss: .3f} train_auc: {train_auc: .3f} "
                    f"| valid_loss: {valid_loss: .3f} valid_auc: {valid_auc: .3f}")
            else:
                logger.debug(f"epoch {epoch}/{n_epochs} | train_loss: {train_loss: .3f} train_auc: {train_auc}")

    if is_mlp:
        torch.save(model.state_dict(), '../models/mlp_clf_weights.pth')
    else:
        torch.save(model.state_dict(), '../models/gcn_clf_weights.pth')


def test(is_mlp):
    # logger
    if is_mlp:
        logger = Logger(name='test_mlp_clf')
    else:
        logger = Logger(name='test_gcn_clf')

    # load config
    th_timestep = int(config['CLASSIFIER']['th_timestep'])

    # load datasets
    if is_mlp:
        datasets = TabularDatasets(is_clf=True)
    else:
        datasets = GraphDataset(is_clf=True)
    n_timestep = len(datasets)

    # load trained model
    if is_mlp:
        trained_model_path = '../models/mlp_clf_weights.pth'
        if not os.path.exists(path=trained_model_path):
            raise FileNotFoundError('Trained model is not found.')
        model = MLPClassifier(n_features, n_classes).to(device)
    else:
        trained_model_path = '../models/gcn_clf_weights.pth'
        if not os.path.exists(path=trained_model_path):
            raise FileNotFoundError('Trained model is not found.')
        model = GraphClassifier(n_features, n_classes).to(device)
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model.double()
    criterion = torch.nn.BCELoss()

    # test model
    test_aucs = []
    for timestep in range(th_timestep + 1, n_timestep):
        test_data = datasets[timestep].to(device)

        with torch.no_grad():
            _, test_out = model(test_data)
            test_out = test_out.reshape((test_data.x.shape[0]))
            test_loss = criterion(test_out, test_data.y)
            test_auc = roc_auc_score(test_data.y.detach().cpu(), test_out.detach().cpu().numpy())
            test_aucs.append(test_auc)
        logger.info(f"timestep {timestep + 1}/{n_timestep} | test_loss: {test_loss: .3f} test_auc: {test_auc: .3f}")
    logger.info(f"average test roc_auc score: {sum(test_aucs)/len(test_aucs): .3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='classification')
    parser.add_argument('--test', action='store_true', help='is test mode')
    parser.add_argument('--mlp', action='store_true', help='is mlp mode')
    args = parser.parse_args()
    if args.test:
        test(is_mlp=args.mlp)
    else:
        train(is_mlp=args.mlp)
