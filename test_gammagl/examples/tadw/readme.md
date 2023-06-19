# Network Representation Learning with Rich Text Information (TADW)

- Paper link: [https://www.ijcai.org/Proceedings/15/Papers/299.pdf](https://www.ijcai.org/Proceedings/15/Papers/299.pdf)
- Author's code repo: [https://github.com/albertyang33/TADW](https://github.com/albertyang33/TADW). Note that the original code is 
  implemented with MATLAB for the paper.

# Dataset Statics

| Dataset  | # Nodes | # Edges | # Classes |
|----------|---------|---------|-----------|
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |

Refer to [Planetoid](https://gammagl.readthedocs.io/en/latest/api/gammagl.datasets.html#gammagl.datasets.Planetoid).

# Performance
> For all the datasets: The training ratio is 50% for linear SVM.

| Dataset  | Paper(10%) | Paper(20%) | Paper(30%) | Paper(40%) | Paper(50%) | Our(tf)      | Our(th)      | Our(pd)     | Our(ms)     |
|----------|------------|------------|------------|------------|------------|--------------|--------------|-------------|-------------|
| Cora     | 82.4       | 85.0       | 85.6       | 86.0       | 86.7       | 90.90±0.74%  | 90.90±0.74%  | 91.18±0.66% | 80.31±0.68% |
| Citeseer | 70.6       | 71.9       | 73.3       | 73.7       | 74.2       | 90.90±0.74%  | 55.22±1.21%  | 56.08±2.13% | 56.08±2.13% |

```bash
TL_BACKEND="tensorflow" python3 tadw_trainer.py --dataset Cora --n_epoch 50 --lr 0.3 --embedding_dim 500 --lamda 0.5 --svdft 300
TL_BACKEND="torch" python3 tadw_trainer.py --dataset Cora --n_epoch 50 --lr 0.3 --embedding_dim 500 --lamda 0.5 --svdft 300
TL_BACKEND="paddle" python3 tadw_trainer.py --dataset Cora --n_epoch 50 --lr 0.3 --embedding_dim 500 --lamda 0.5 --svdft 300
TL_BACKEND="mindspore" python3 tadw_trainer.py --dataset Cora --n_epoch 50 --lr 0.3 --embedding_dim 500 --lamda 0.5 --svdft 300

TL_BACKEND="torch" python3 tadw_trainer.py --dataset Citeseer --n_epoch 50 --lr 0.3 --embedding_dim 500 --lamda 0.5 --svdft 300
TL_BACKEND="tensorflow" python3 tadw_trainer.py --dataset Citeseer --n_epoch 50 --lr 0.3 --embedding_dim 500 --lamda 0.5 --svdft 300
TL_BACKEND="paddle" python3 tadw_trainer.py --dataset Citeseer --n_epoch 50 --lr 0.3 --embedding_dim 500 --lamda 0.5 --svdft 300
TL_BACKEND="mindspore" python3 tadw_trainer.py --dataset Citeseer --n_epoch 50 --lr 0.3 --embedding_dim 500 --lamda 0.5 --svdft 300
```