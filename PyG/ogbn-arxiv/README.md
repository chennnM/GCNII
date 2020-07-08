# ogbn-arxiv
This repository includes a simple *PyTorch Geometric* implementation of "Simple and Deep Graph Convolutional Networks".

## Requirements
- CUDA 10.1
- python 3.6.10
- pytorch 1.5.1
- torch-geometric 1.5.0
- ogb 1.2.0

## Training & Evaluation
```
python arxiv.py
```
- Accuracy: 0.7274 ± 0.0016
- Params: 2,148,648
- Hardware: Quadro RTX 8000 (48GB GPU)