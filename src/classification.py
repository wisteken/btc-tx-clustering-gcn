import sys
import torch
import argparse
import configparser
from sklearn.metrics import roc_auc_score

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


def train():
    # logger
    logger = Logger(name='classification')

    # load config
    th_timestep = int(config['CLASSIFIER']['th_timestep'])
    n_epochs = int(config['CLASSIFIER']['n_epochs'])

    # load datasets
    datasets = EllipticDataset(is_train=True)

    # define model and parameters
    model = Classifier(n_features=165, n_classes=1).to(device)
    model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
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
            train_out = model(train_data)
            train_out = train_out.reshape((train_data.x.shape[0]))
            train_loss = criterion(train_out, train_data.y)
            train_auc = roc_auc_score(train_data.y.detach().cpu().numpy(), train_out.detach().cpu().numpy())
            train_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                with torch.no_grad():
                    valid_out = model(valid_data)
                    valid_out = valid_out.reshape((valid_data.x.shape[0]))
                    valid_loss = criterion(valid_out, valid_data.y)
                    valid_auc = roc_auc_score(valid_data.y.detach().cpu().numpy(), valid_out.detach().cpu().numpy())
                logger.info(
                    f"epoch {epoch}/{n_epochs} | train_loss: {train_loss: .3f} train_auc: {train_auc: .3f} "
                    f"| valid_loss: {valid_loss: .3f} valid_auc: {valid_auc: .3f}")
            else:
                logger.debug(f"epoch {epoch}/{n_epochs} | train_loss: {train_loss: .3f} train_auc: {train_auc}")

    torch.save(model.state_dict(), "../models/classifier_weights.pth")


def test():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classification")
    parser.add_argument('--test', action="store_true", help='is test mode')
    args = parser.parse_args()
    if args.test:
        test()
    else:
        train()
