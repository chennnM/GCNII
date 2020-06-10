# Simple and Deep Graph Convolutional Networks
This repository contains a PyTorch implementation of ICML 2020 Paper "Simple and Deep Graph Convolutional Networks".

## Dependencies
- CUDA 10.1
- python 3.6.9
- pytorch 1.3.1
- networkx 2.1
- scikit-learn

## Datasets

The `data` folder contains three benchmark datasets(Cora, Citeseer, Pubmed), and the `newdata` folder contains four datasets(Chameleon, Cornell, Texas, Wisconsin) from [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn). We use the same semi-supervised setting as [GCN](https://github.com/tkipf/gcn) and the same full-supervised setting as Geom-GCN. PPI can be downloaded from [GraphSAGE](http://snap.stanford.edu/graphsage/).


## Usage

- To replicate the semi-supervised results, run the following script
```sh
sh semi.sh
```
- To replicate the full-supervised results, run the following script
```sh
sh full.sh
```
- To replicate the inductive results of PPI, run the following script
```sh
sh ppi.sh
```
