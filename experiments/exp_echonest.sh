#!/usr/bin/env bash
# Best UNIFORM score: [0.067834   0.03042152], MMR: 358.4845
# Best POPULAR score: [0.08187394 0.04360328], MMR: 30.7825
# Best 2-STAGE score: [0.09179041 0.04784787], MMR: 146.327

CUDA_VISIBLE_DEVICES=0 python experiments/exp_echonest.py \
    -learning_rate=0.0001 \
    -embedding_dim=128 \
    -n_epochs=50 \
    -initialized_std=0.1 \
    -eval_every_n_batches=2000 \
    -mode=train \
    -sampler=uniform \
    -models_home=results/models >& logs/echonest_lr0.0001_dim128_nepoch50_uniform_batchsize4096_negatives1.log

CUDA_VISIBLE_DEVICES=0 python experiments/exp_echonest.py \
    -learning_rate=0.0001 \
    -embedding_dim=128 \
    -n_epochs=50 \
    -initialized_std=0.1 \
    -eval_every_n_batches=2000 \
    -mode=mmr \
    -sampler=uniform \
    -models_home=results/models

#######

CUDA_VISIBLE_DEVICES=0 python experiments/exp_echonest.py \
    -learning_rate=0.0001 \
    -embedding_dim=128 \
    -n_epochs=50 \
    -initialized_std=0.1 \
    -eval_every_n_batches=2000 \
    -mode=train \
    -sampler=popular \
    -models_home=results/models >& logs/echonest_lr0.0001_dim128_nepoch50_popular_batchsize256_negatives1.log

CUDA_VISIBLE_DEVICES=0 python experiments/exp_echonest.py \
    -learning_rate=0.0001 \
    -embedding_dim=128 \
    -n_epochs=50 \
    -initialized_std=0.1 \
    -eval_every_n_batches=2000 \
    -mode=mmr \
    -sampler=popular \
    -models_home=results/models

#######

CUDA_VISIBLE_DEVICES=0 python experiments/exp_echonest.py \
    -learning_rate=0.0001 \
    -activate_l2_norm=True \
    -embedding_dim=128 \
    -n_epochs=50 \
    -initialized_std=0.1 \
    -eval_every_n_batches=2000 \
    -mode=train \
    -sampler=spreadout \
    -update_embeddings_every_n_batches=100 \
    -spreadout_weight=0.01 \
    -num_neg_candidates=2000 \
    -models_home=results/models >& logs/echonest_lr0.0001_dim128_nepoch50_spreadout_batchsize256_negatives1.log

CUDA_VISIBLE_DEVICES=0 python experiments/exp_echonest.py \
    -learning_rate=0.0001 \
    -activate_l2_norm=True \
    -embedding_dim=128 \
    -n_epochs=50 \
    -initialized_std=0.1 \
    -eval_every_n_batches=2000 \
    -mode=mmr \
    -sampler=spreadout \
    -update_embeddings_every_n_batches=100 \
    -spreadout_weight=0.01 \
    -num_neg_candidates=2000 \
    -models_home=results/models
