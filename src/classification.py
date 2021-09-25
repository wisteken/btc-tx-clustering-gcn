import torch
import configparser
from sklearn.metrics import roc_auc_score

from utils import EllipticDataset
from model import Classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = configparser.ConfigParser()
config.read('./config.ini')


def train():
    # load config
    th_timestep = int(config['TRAIN']['th_timestep'])
    n_epochs = int(config['TRAIN']['n_epochs'])

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
        print(f"timestep {timestep + 1}/{th_timestep}")

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
                print(
                    f"epoch {epoch}/{n_epochs} | train_loss: {train_loss: .3f} train_auc: {train_auc: .3f} "
                    f"| valid_loss: {valid_loss: .3f} valid_auc: {valid_auc: .3f}")

    torch.save(model.state_dict(), "../models/classifier_weights.pth")


def test():
    pass


train()
