# DisenE-coling

Codes for paper "DisenE: Disentangling Knowledge Graph Embeddings".<br>
Xiaoyu Kou, Yankai Lin, Yuntao Li, Jiahao Xu, Peng Li, Jie Zhou, Yan Zhang.<br>
In COLING 2020
### [arXiv](https://arxiv.org/abs/2010.14730)


## Requirement

* pytorch: 1.6.0
* python: 3.6
* numpy: 1.18.5

## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/KXY-PUBLIC/DisenE.git
cd DisenE
```

### Dataset

- DisenE/data: FB15K-237 and WN18RR are two well-known KG benchmark datasets. 


## Training Examples:

For example, training on FB15k-237 datasets:
```
CUDA_VISIBLE_DEVICES=0 nohup python -u run.py --dataset=FB15k-237  --epochs=800 --model_name=DisenE --k_factors=6 --step_size=50 --embedding_size=200 --w1=0.1 --w2=0.1 --sample_num=50  &> log/DisenE_fb_k_6.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --dataset=FB15k-237  --epochs=800 --model_name=DisenE_Trans --k_factors=4 --embedding_size=200 --w1=0.1 &> log/DisenE_Trans_fb_k_4.out &
```

 