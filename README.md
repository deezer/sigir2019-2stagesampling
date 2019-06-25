# Improving Collaborative Metric Learning with Efficient Negative Sampling

This repository contains implementation for the following paper:

> V-A. Tran, R. Hennequin, J. Royo-Letelier and M. Moussallam. Improving Collaborative Metric Learning with Efficient Negative Sampling. In: *Proceedings of SIGIR 2019*, July 2019.


## Environment
- python 3.6
- tensorflow 1.12
- numpy 1.15.4
- scipy 1.1.0
- sklearn 0.20.2
- pandas 0.24.2

## Example dataset
- Download the Echonest dataset (TRIPLETS FOR 1M USERS) from [here](http://millionsongdataset.com/tasteprofile/).
- Put the unzipped files into directory data/echonest

## Run example script
Run experiments/exp_echonest.sh for an example:
- UNIFORM ndcg/map: [0.067834   0.03042152], MMR: 358.485
- POPULAR ndcg/map: [0.08187394 0.04360328], MMR: 30.783
- 2-STAGE ndcg/map: [0.09179041  0.04784787], MMR: 146.327
