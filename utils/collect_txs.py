import os
import time
import json
import requests
from tqdm import tqdm
import pandas as pd


def wait_for_result(url):
    success = False
    response = requests.get(url)
    while not success:
        time.sleep(1)
        if str(response) == "<Response [200]>":
            time.sleep(1)
            success = True
        else:
            print(response)
            print(response.headers)
            time.sleep(5)
            response = requests.get(url)
    return response


if not os.path.exists('../datasets/addresses.csv'):
    raise Exception('Error: before you run collect_richest.py')

started_at = 1627776000
ended_at = 1630454400

addresses = pd.read_csv('../datasets/addresses.csv', index_col=0).values
result = pd.DataFrame(columns=['address', 'balance', 'tx_count', 'received', 'sent'], index=range(1, len(addresses) + 1))

for i, [address] in enumerate(tqdm(addresses)):
    address_api = f'https://chain.api.btc.com/v3/address/{address}'
    res = wait_for_result(address_api)
    res_json = json.loads(res.text)['data']
    result.loc[i + 1]['address'] = address
    result.loc[i + 1]['balance'] = res_json['balance']
    result.loc[i + 1]['tx_count'] = res_json['tx_count']
    result.loc[i + 1]['received'] = res_json['received']
    result.loc[i + 1]['sent'] = res_json['sent']

    all_tx_hash = []
    for j in range(10**3):
        address_tx_api = f'https://chain.api.btc.com/v3/address/{address}/tx?page={j}&pagesize=50'
        res = wait_for_result(address_tx_api)
        res_json = json.loads(res.text)['data']
        txs = res_json['list']
        filtered_txs = [tx['hash'] for tx in txs if (started_at <= tx['created_at'] <= ended_at)]
        if (len(filtered_txs) == 0):
            break
        else:
            all_tx_hash += filtered_txs

    all_txs = []
    for tx_hash in all_tx_hash:
        tx_api = f'https://chain.api.btc.com/v3/tx/{tx_hash}'
        res = wait_for_result(tx_api)
        res_json = json.loads(res.text)['data']
        all_txs += [{"inputs": res_json["inputs"], "outputs":res_json["outputs"]}]

    with open(f'../datasets/transactions/{address}.json', 'w') as f:
        json.dump(all_txs, f)

result.to_csv('../datasets/accounts.csv', index=True)
