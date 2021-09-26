#!/bin/bash

gunzip ./datasets/bitcoin_2018_bh.dat.gz
gunzip ./datasets/bitcoin_2018_addresses.dat.gz
gunzip ./datasets/bitcoin_2018_multiple.dat.gz
gunzip ./datasets/bitcoin_2018_nonstandard.dat.gz
xz --decompress ./datasets/bitcoin_2018_tx.dat.xz
gunzip ./datasets/bitcoin_2018_txh.dat.gz
xz --decompress ./datasets/bitcoin_2018_txin.dat.xz
xz --decompress ./datasets/bitcoin_2018_txout.dat.xz
gunzip ./datasets/bitcoin_2018_addr_sccs.dat.gz