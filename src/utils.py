from networkx.algorithms.traversal.depth_first_search import dfs_edges
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch_geometric.data import Data


def load_data():
    # csv path
    features_csv_path = '../datasets/elliptic_txs_features.csv'
    classes_csv_path = '../datasets/elliptic_txs_classes.csv'
    edges_csv_path = '../datasets/elliptic_txs_edgelist.csv'

    # load node features
    df_features = pd.read_csv(features_csv_path, header=None)
    df_features.columns = ['txId', 'time step'] + \
        [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in range(72)]

    # load edge lists
    df_edges = pd.read_csv(edges_csv_path)

    # load classes
    df_classes = pd.read_csv(classes_csv_path)
    df_classes['class'] = df_classes['class'].map({'unknown': 2, '1': 1, '2': 0})  # 1 for illicit tx, 0 for licit tx

    # marge classes into features
    df_merged_features = pd.merge(df_features, df_classes, how='left', right_on='txId', left_on='txId')
    del df_features, df_classes

    # mapping txId to idx
    mapping_table = {txId: idx for idx, txId in enumerate(df_merged_features['txId'])}
    df_edges['txId1'] = df_edges['txId1'].map(mapping_table)
    df_edges['txId2'] = df_edges['txId2'].map(mapping_table)
    df_merged_features = df_merged_features.drop('txId', axis=1)

    features = []
    classes = []
    edges = []
    for _, df_time_step_features in df_merged_features.groupby('time step'):
        classes.append(df_time_step_features['class'])
        features.append(df_time_step_features.drop(['time step', 'class'], axis=1))
        edges.append(df_edges.loc[df_edges['txId1'].isin(df_time_step_features.index)])
    del df_merged_features, df_edges

    return features, edges, classes


class EllipticDataset(data.Dataset):
    def __init__(self, is_train=False):
        self.is_train = is_train
        self.features, self.edges, self.classes = load_data()

    def __getitem__(self, index):
        if (self.is_train):
            ids = self.classes[index].loc[(self.classes[index] == 0) | (self.classes[index] == 1)].index
            node_features = torch.tensor(self.features[index].iloc[ids].values, dtype=torch.double)
            edge_index = torch.tensor(
                self.edges[index].loc[(self.edges[index]['txId1'].isin(ids)) | (self.edges[index]['txId2'].isin(ids))].values, dtype=torch.long)
            labels = torch.tensor(self.classes[index].iloc[ids].values, dtype=torch.long)
            return Data(x=node_features, edge_index=edge_index, y=labels)
        else:
            node_features = torch.tensor(self.features[index].values, dtype=torch.double)
            edge_index = torch.tensor(self.edges[index].values, dtype=torch.long)
            labels = torch.tensor(self.classes[index].values, dtype=torch.long)
            return Data(x=node_features, edge_index=edge_index, y=labels)

    def __len__(self):
        return len(self.classes)


if __name__ == '__main__':
    datasets = EllipticDataset(is_train=True)
    print(datasets[0])
