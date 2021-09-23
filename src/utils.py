import torch
from torch_geometric import data, loader
import pandas as pd


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

    node_features = []
    for _, df_time_step_features in df_merged_features.groupby('time step'):
        node_features.append(df_time_step_features.drop('time step', axis=1).values)
    del df_merged_features

    edge_index = df_edges.T.values
    del df_edges

    return node_features, edge_index


if __name__ == '__main__':
    load_data()