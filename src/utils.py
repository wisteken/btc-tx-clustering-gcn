import sys
import bisect
import logging
import joblib
import warnings
warnings.simplefilter('ignore')
import configparser
import numpy as np
import pandas as pd
from tqdm import tqdm
from pytz import timezone
from datetime import datetime

config = configparser.ConfigParser()
config.read('./config.ini')
seed = config['DEFAULT']['seed']


class Logger:
    def __new__(cls, name: str = None, logdir: str = '../logs'):
        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_handler.setLevel(level=logging.INFO)
        stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
        filename = f"{logdir}/{name+'_' if name else ''}{datetime.now(timezone('Asia/Tokyo')).strftime('%Y-%m-%d_%H:%M:%S')}.log"
        logfile_handler = logging.FileHandler(filename=filename)
        logfile_handler.setLevel(level=logging.DEBUG)
        logfile_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))
        logger = logging.getLogger()
        logger.setLevel(level=logging.DEBUG)
        logger.addHandler(stdout_handler)
        logger.addHandler(logfile_handler)

        return logger


def cropping_data():
    start_timestamp = int(config['DATASETS']['START_TIMESTAMP'])
    end_timestamp = int(config['DATASETS']['END_TIMESTAMP'])

    bh_table_path = '../datasets/bitcoin_2018_bh.dat'
    bh_df = pd.read_table(bh_table_path, header=None, index_col=0)
    bh_df.columns = ['hash', 'timestamp', 'n_txs']
    timestamps = bh_df['timestamp'].values
    start_bid = bisect.bisect_right(timestamps, start_timestamp)
    end_bid = bisect.bisect_left(timestamps, end_timestamp) - 1
    print(start_bid, end_bid)
    print(f"target n_blocks: {end_bid - start_bid}")
    bh_df.loc[start_bid: end_bid].to_csv('../datasets/bitcoin_2018/bh.csv')
    del bh_df

    tx_table_path = '../datasets/2018_tx/tx_eh'
    tx_df = pd.read_table(tx_table_path, header=None, index_col=0)
    tx_df.columns = ['blockID', 'n_inputs', 'n_outputs']
    block_ids = tx_df['blockID'].values
    start_txid = tx_df.index[0] + bisect.bisect_right(block_ids, start_bid)
    end_txid = tx_df.index[0] + bisect.bisect_left(block_ids, end_bid) - 1
    print(start_txid, end_txid)
    print(f"target n_tx: {end_txid - start_txid}")
    tx_df.loc[start_txid: end_txid].to_csv('../datasets/bitcoin_2018/tx.csv')
    del tx_df

    txin_table_path = '../datasets/2018_txin/txineh'
    txin_df = pd.read_table(txin_table_path, header=None)
    txin_df.columns = ['txID', 'input_seq', 'prev_txID', 'prev_output_seq', 'addrID', 'sum']
    tx_ids = txin_df['txID'].values
    start_txinid = txin_df.index[0] + bisect.bisect_right(tx_ids, start_txid)
    end_txinid = txin_df.index[0] + bisect.bisect_left(tx_ids, end_txid)
    print(start_txinid, end_txinid)
    print(f"target n_txins: {end_txinid - start_txinid}")
    txin_df.loc[start_txinid: end_txinid].to_csv('../datasets/bitcoin_2018/txin.csv')
    del txin_df

    txout_table_path = '../datasets/2018_txout/xed'
    txout_df = pd.read_table(txout_table_path, header=None)
    txout_df.columns = ['txID', 'output_seq', 'addrID', 'sum']
    tx_ids = txout_df['txID'].values
    start_txoutid = txout_df.index[0] + bisect.bisect_right(tx_ids, start_txid)
    end_txoutid = txout_df.index[0] + bisect.bisect_left(tx_ids, end_txid) - 1
    print(start_txoutid, end_txoutid)
    print(f"target n_txouts: {end_txoutid - start_txoutid}")
    txout_df.loc[start_txoutid: end_txoutid].to_csv('../datasets/bitcoin_2018/txout.csv')
    del txout_df


def preprocess_data():
    tx_path = '../datasets/bitcoin_2018/tx.csv'
    df_tx = pd.read_csv(tx_path, index_col=0, header=0)
    df_tx = df_tx.reset_index().rename(columns={'0': 'txID'})
    mapping = {txID: idx + 1 for idx, txID in enumerate(df_tx['txID'])}

    edges = []
    txin_path = '../datasets/bitcoin_2018/txin.csv'
    df_txin = pd.read_csv(txin_path, index_col=0, header=0)
    for target_tx, gdf_txin in tqdm(df_txin.groupby('txID')):
        if mapping.get(target_tx, -1) != -1:
            n_groups = len(gdf_txin)
            input_seq_mean = 0
            # prev_output_seq_mean = 0
            in_sum_mean = 0
            for idx in gdf_txin.index:
                source_tx = gdf_txin.loc[idx, 'prev_txID']
                if mapping.get(source_tx, -1) != -1:
                    edges.append([mapping[source_tx], mapping[target_tx]])
                input_seq_mean += gdf_txin.loc[idx, 'input_seq']
                # prev_output_seq_mean += gdf_txin.loc[idx, 'prev_output_seq']
                in_sum_mean += gdf_txin.loc[idx, 'sum']
            df_tx.loc[mapping[target_tx], 'input_seq_mean'] = input_seq_mean / n_groups
            # df_tx.loc[mapping[target_tx], 'prev_output_seq_mean'] = prev_output_seq_mean / n_groups / 10**3
            df_tx.loc[mapping[target_tx], 'in_sum_mean'] = in_sum_mean / n_groups

    txout_path = '../datasets/bitcoin_2018/txout.csv'
    df_txout = pd.read_csv(txout_path, index_col=0, header=0)
    for target_tx, gdf_txout in tqdm(df_txout.groupby('txID')):
        if mapping.get(target_tx, -1) != -1:
            n_groups = len(gdf_txout)
            output_seq_mean = 0
            out_sum_mean = 0
            for idx in gdf_txout.index:
                output_seq_mean += gdf_txout.loc[idx, 'output_seq']
                out_sum_mean += gdf_txout.loc[idx, 'sum']
            df_tx.loc[mapping[target_tx], 'output_seq_mean'] = output_seq_mean / n_groups
            df_tx.loc[mapping[target_tx], 'out_sum_mean'] = out_sum_mean / n_groups
    edge_index = np.array(edges)
    df_tx = df_tx.fillna(0)
    node_features = df_tx[
        ['n_inputs', 'n_outputs', 'input_seq_mean', 'in_sum_mean', 'output_seq_mean', 'out_sum_mean']
    ].values

    joblib.dump(edge_index, '../datasets/bitcoin_2018/edge_index.bin')
    joblib.dump(node_features, '../datasets/bitcoin_2018/node_features.bin')

    return node_features, edge_index


def load_data():
    # load node features
    features_path = '../datasets/bitcoin_2018/node_features.bin'
    features = joblib.load(features_path)

    # load edge lists
    edges_path = '../datasets/bitcoin_2018/edge_index.bin'
    edges = joblib.load(edges_path).reshape(2, -1)

    return features, edges


def export_to_csv():
    # load node features
    features_path = '../datasets/bitcoin_2018/node_features.bin'
    features = joblib.load(features_path)
    pd.DataFrame(features).to_csv('../datasets/bitcoin_2018/node_features.csv')

    # load edge lists
    edges_path = '../datasets/bitcoin_2018/edge_index.bin'
    edges = joblib.load(edges_path).reshape(2, -1)
    pd.DataFrame(edges).to_csv('../datasets/bitcoin_2018/edge_index.csv')


if __name__ == '__main__':
    # cropping_data()
    # node_features, edge_index = preprocess_data()
    # features, edges = load_data()
    export_to_csv()
