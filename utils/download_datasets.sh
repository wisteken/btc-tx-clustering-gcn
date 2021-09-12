#!/bin/bash

https://datadryad.org/stash/downloads/file_stream/583428 -O ../datasets/bitcoin_2020_addresses.dat.gz
https://datadryad.org/stash/downloads/file_stream/583429 -O ../datasets/bitcoin_2020_bh.dat.gz
https://datadryad.org/stash/downloads/file_stream/583430 -O ../datasets/bitcoin_2020_multiple.dat.gz
https://datadryad.org/stash/downloads/file_stream/583431 -O ../datasets/bitcoin_2020_nonstandard.dat.gz
https://datadryad.org/stash/downloads/file_stream/583485 -O ../datasets/bitcoin_2020_tx.dat.xz
https://datadryad.org/stash/downloads/file_stream/583432 -O ../datasets/bitcoin_2020_txh.dat.gz
https://datadryad.org/stash/downloads/file_stream/583433 -O ../datasets/bitcoin_2020_txin.dat.xz
https://datadryad.org/stash/downloads/file_stream/583434 -O ../datasets/bitcoin_2020_txout.dat.xz

gunzip bitcoin_2020_addresses.dat.gz
gunzip bitcoin_2020_bh.dat.gz
gunzip bitcoin_2020_multiple.dat.gz
gunzip bitcoin_2020_nonstandard.dat.gz
xz --decompress bitcoin_2020_tx.dat.xz
gunzip bitcoin_2020_txh.dat.gz
xz --decompress bitcoin_2020_txin.dat.xz
xz --decompress bitcoin_2020_txout.dat.xz
