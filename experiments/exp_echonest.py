from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import json
import logging
import argparse
import toolz
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.contrib.training import HParams

from datasets import split_data
from datasets.echonest import fetch_echonest
from models.triplet import Triplet
from samplers import SamplerFactory

logger = logging.getLogger(__name__)

RANDOM_SEED = 1234567

params = HParams(
    # dataset
    min_playcount=10,
    min_interactions=10,

    # model
    optimizer='adam',
    learning_rate=0.0001,
    embedding_dim=32,
    clip_norm=1.0,
    margin=1.0,
    initialized_std=1.0,
    activate_l2_norm=False,

    # training iter & batch
    n_epochs=20,
    eval_every_n_batches=2000,

    # valid & test eval parameters
    n_users_in_chunk=100,
    n_users_in_validation=3000,
    n_users_in_test=3000,

    n_negatives=1,
    mode='train'
)

sampler_params = {
    'uniform': {
        'sampler': 'uniform',
        'batch_size': 4096
    },
    'popular': {
        'sampler': 'popular',
        'max_count': None,
        'batch_size': 256
    },
    'spreadout': {
        'sampler': 'spreadout',
        'spreadout_weight': 1.0,
        'n_try': 10,
        'num_neg_candidates': 300,
        'max_count': None,
        'batch_size': 256,
        'update_embeddings_every_n_batches': 100
    }
}


def update_params_from_parser(params, parser):
    """
    Update tensorflow's hparams object using values from
    python's ArgumentParser.

    Parameters
    ----------
    params: tf.contrib.training.python.training.hparam.HParams
        tensorflow's hparams
    parser: argparse.ArgumentParser
        python argparse
    """
    for param_name, param_value in json.loads(params.to_json()).items():
        parser.add_argument("-%s" % param_name, help=param_name,
                            type=type(param_value), default=param_value)
    pargs = parser.parse_args()
    for k, v in pargs.__dict__.items():
        if k != 'sampler':
            if hasattr(params, k):
                params.set_hparam(k, v)
            else:
                params.add_hparam(k, v)
        else:
            for sampler_k, sampler_v in sampler_params[v].items():
                if hasattr(params, sampler_k):
                    print(sampler_k)
                    print(sampler_v)
                    params.set_hparam(sampler_k, sampler_v)
                else:
                    print(sampler_k)
                    print(sampler_v)
                    params.add_hparam(sampler_k, sampler_v)
    return params, pargs


def train_model(params, train_interactions, valid_interactions, eval_users):
    """
    Train model
    :param params:
    :param train_interactions:
    :param valid_interactions:
    :return:
    """
    train_interactions = sp.lil_matrix(train_interactions)

    # create a tensorflow session and train model
    with tf.Session() as sess:
        model = Triplet(sess=sess, params=params)
        model.build_graph()

        extra_sampler_args = {}
        if params.sampler != 'uniform':
            extra_sampler_args['max_count'] = params.max_count
        if params.sampler == 'spreadout':
            extra_sampler_args['model'] = model
            extra_sampler_args['embedding_dim'] = params.embedding_dim
            extra_sampler_args['n_try'] = params.n_try
            extra_sampler_args['num_neg_candidates'] = params.num_neg_candidates

        sampler = SamplerFactory.generate_sampler(
            sampler_name=params.sampler,
            interactions=train_interactions,
            n_negatives=params.n_negatives,
            batch_size=params.batch_size,
            n_workers=5,
            **extra_sampler_args)
        # training the model
        model.fit(sampler, train_interactions, valid_interactions, eval_users,
                  metrics=['ndcg', 'map'], k=50)
        sampler.close()
    tf.reset_default_graph()


def get_item_popularity(interactions, max_count=None):
    interactions = sp.lil_matrix(interactions)
    popularity_dict = defaultdict(int)
    for uid, iids in enumerate(interactions.rows):
        for iid in iids:
            popularity_dict[iid] += 1
            if max_count is not None and popularity_dict[iid] > max_count:
                popularity_dict[iid] = max_count
    return popularity_dict


def get_median_rank_recommended_items(train_interactions,
                                      eval_users,
                                      item_popularities,
                                      n_users_in_chunk):
    train_interactions = sp.lil_matrix(train_interactions)
    max_train_interaction_count = max(len(row) for row in train_interactions.rows)
    train_user_items = {u: set(train_interactions.rows[u])
                        for u in eval_users if train_interactions.rows[u]}

    recommended_items = []
    with tf.Session() as sess:
        model = Triplet(sess=sess, params=params)
        logger.info('Get recommended items for users in evaluation')
        for user_chunk in toolz.partition_all(n_users_in_chunk, eval_users):
            recommended_items = recommended_items + \
                                model.get_recommended_items(
                                    user_chunk, train_user_items,
                                    max_train_interaction_count,
                                    k=50)
        logger.info('Get rank for recommended items')
        item_ranks = []
        for iids in recommended_items:
            item_ranks.append([item_popularities[iid] for iid in iids])
        median_ranks = np.median(item_ranks, axis=1)
    tf.reset_default_graph()
    return np.mean(median_ranks)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG)

    tf.set_random_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Add parameter for results folder
    params.add_hparam('models_home', '')
    params.add_hparam('sampler', '')
    params.add_hparam('spreadout_weight', 0.)
    params.add_hparam('n_try', 10)
    params.add_hparam('num_neg_candidates', 0)
    params.add_hparam('max_count', None)
    params.add_hparam('update_embeddings_every_n_batches', -1)

    # update tensorflow's hparams object using values from
    # python's ArgumentParser.
    parser = argparse.ArgumentParser()
    params, pargs = update_params_from_parser(params, parser)

    logger.info('Load echonest dataset')
    dataset = fetch_echonest(data_home='data/echonest',
                             min_playcount=params.min_playcount,
                             min_interactions=params.min_interactions)

    n_users = dataset['interactions'].shape[0]
    n_items = dataset['interactions'].shape[1]

    # update n_users, n_items parameters
    if hasattr(params, 'n_users'):
        params.set_hparam('n_users', n_users)
    else:
        params.add_hparam('n_users', n_users)

    if hasattr(params, 'n_items'):
        params.set_hparam('n_items', n_items)
    else:
        params.add_hparam('n_items', n_items)

    model_path = os.path.join(
        'echonest_pltcnt{}_minint{}'.format(
            params.min_playcount,
            params.min_interactions),
        'batch_{}'.format(params.batch_size),
        'lr{}_dim{}_nepochs{}_sampler-{}_batchsize{}_negs{}'.format(
            params.learning_rate,
            params.embedding_dim,
            params.n_epochs,
            params.sampler,
            params.batch_size,
            params.n_negatives
        ))

    if params.sampler == 'spreadout':
        model_path = '{}_spreadout{}'.format(model_path,
                                             params.spreadout_weight)
    model_path = os.path.join(pargs.models_home, model_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if hasattr(params, 'model_dir'):
        params.set_hparam('model_dir', model_path)
    else:
        params.add_hparam('model_dir', model_path)

    logging.info('\nPARAMETERS:\n--------------\n{}\n--------------\n'.format(
            params.to_json()))

    # split data into train/valid/test
    train_interactions, valid_interactions, test_interactions = split_data(
        interactions=dataset['interactions'],
        split_ratio=(3, 1, 1),
        random_seed=RANDOM_SEED)

    eval_users = np.array(np.random.choice(np.arange(len(dataset['user_ids'])),
                                           size=params.n_users_in_validation,
                                           replace=False))

    if params.mode == 'train':
        train_model(params, train_interactions, valid_interactions, eval_users)
    else:
        logging.info('Get item popularities')
        item_popularities = get_item_popularity(train_interactions)

        mmr = get_median_rank_recommended_items(train_interactions,
                                                eval_users,
                                                item_popularities,
                                                params.n_users_in_chunk)
        logger.info('Mean Median Rank: {}'.format(mmr))
