# Simple and Deep Graph Convolutional Networks
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/simple-and-deep-graph-convolutional-networks/node-classification-on-cora-full-supervised)](https://paperswithcode.com/sota/node-classification-on-cora-full-supervised?p=simple-and-deep-graph-convolutional-networks)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/simple-and-deep-graph-convolutional-networks/node-classification-on-pubmed-full-supervised)](https://paperswithcode.com/sota/node-classification-on-pubmed-full-supervised?p=simple-and-deep-graph-convolutional-networks)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/simple-and-deep-graph-convolutional-networks/node-classification-on-cora-with-public-split)](https://paperswithcode.com/sota/node-classification-on-cora-with-public-split?p=simple-and-deep-graph-convolutional-networks)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/simple-and-deep-graph-convolutional-networks/node-classification-on-ppi)](https://paperswithcode.com/sota/node-classification-on-ppi?p=simple-and-deep-graph-convolutional-networks)

This repository contains a PyTorch implementation of "Simple and Deep Graph Convolutional Networks".(https://arxiv.org/abs/2007.02133)

## Dependencies
- CUDA 10.1
- python 3.6.9
- pytorch 1.3.1
- networkx 2.1
- scikit-learn

## Datasets

The `data` folder contains three benchmark datasets(Cora, Citeseer, Pubmed), and the `newdata` folder contains four datasets(Chameleon, Cornell, Texas, Wisconsin) from [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn). We use the same semi-supervised setting as [GCN](https://github.com/tkipf/gcn) and the same full-supervised setting as Geom-GCN. PPI can be downloaded from [GraphSAGE](http://snap.stanford.edu/graphsage/).

## Results
Testing accuracy summarized below.
| Dataset | Depth |  Metric | Dataset | Depth |  Metric |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Cora       | 64 | 85.5  | Cham | 8  | 62.48 |
| Cite       | 32 | 73.4  | Corn | 16 | 76.49 |
| Pubm       | 16 | 80.3  | Texa | 32 | 77.84 |
| Cora(full) | 64 | 88.49 | Wisc | 16 | 81.57 |
| Cite(full) | 64 | 77.13 | PPI  | 9  | 99.56 |
| Pubm(full) | 64 | 90.30 | obgn-arxiv | 16 | 72.74 |


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
## Reference implementation
The `PyG` folder includes a simple *PyTorch Geometric* implementation of GCNII.
Requirements: `torch-geometric >= 1.5.0` and  `ogb >= 1.2.0`.
- Running examples
```
python cora.py
python arxiv.py
```

## Citation
```
@article{chenWHDL2020gcnii,
  title = {Simple and Deep Graph Convolutional Networks},
  author = {Ming Chen, Zhewei Wei and Zengfeng Huang, Bolin Ding and Yaliang Li},
  year = {2020},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
}
```
