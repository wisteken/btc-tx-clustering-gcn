# bitcoin-tx-clustering-gcn

## 1. Preparation

- Download the [datasets](https://doi.org/10.5061/dryad.qz612jmcf).

```shell
chmod +x ./script/download.sh
./script/download.sh
```

- Extract the downloaded data.

```shell
chmod +x ./script/extract.sh
./script/extract.sh
```

## 2. Build

- Build docker image

```shell
docker-compose up --build -d
```

## 3. Dive into the container

```shell
docker-compose exec python bash
cd src
```

- Train clustering model

```shell
python3 clustering.py
```

- Test clustering model

```shell
python3 clustering.py --test
```

## 4. Analyze results

Please check results and logs directory.
