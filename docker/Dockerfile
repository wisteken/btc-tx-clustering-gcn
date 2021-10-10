FROM nvidia/cuda:11.0-devel-ubuntu20.04

RUN apt-get update
RUN apt-get install -y python3 python3-pip

WORKDIR /work

COPY requirements.txt .

RUN pip install -r requirements.txt

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
