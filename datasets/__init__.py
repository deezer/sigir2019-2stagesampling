import logging
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
from collections import defaultdict

logger = logging.getLogger(__name__)


def split_data(interactions, split_ratio=(3, 1, 1), random_seed=42):
    """
    Split interactions data into train/validation/test set.

    Parameters
    ----------
    interactions: sp.dok_matrix
        The total interactions matrix in sparse format
    split_ratio: tuple
        The ratio for each set
    random_seed: int
        The random seed

    Returns
    -------
        A tuple of (train, valid, test) sets.
    """
    np.random.seed(random_seed)

    train_interactions = sp.dok_matrix(interactions.shape)
    valid_interactions = sp.dok_matrix(interactions.shape)
    test_interactions = sp.dok_matrix(interactions.shape)

    logger.info('Get user item positive interactions dict')
    user_item_pairs = np.asarray(interactions.nonzero()).T
    user_item_dict = defaultdict(set)
    for uid, iid in user_item_pairs:
        user_item_dict[uid].add(iid)

    description = 'Split data into train/valid/test'
    for uid in tqdm(np.arange(interactions.shape[0]), desc=description):
        items = list(user_item_dict[uid])
        if len(items) >= sum(split_ratio):
            # shuffle items first
            np.random.shuffle(items)

            # split
            train_count = int(len(items) * split_ratio[0] / sum(split_ratio))
            valid_count = int(len(items) * split_ratio[1] / sum(split_ratio))

            for i in items[0: train_count]:
                train_interactions[uid, i] = 1
            for i in items[train_count: train_count + valid_count]:
                valid_interactions[uid, i] = 1
            for i in items[train_count + valid_count:]:
                test_interactions[uid, i] = 1

    logger.info("{}/{}/{} train/valid/test samples".format(
        train_interactions.count_nonzero(),
        valid_interactions.count_nonzero(),
        test_interactions.count_nonzero()))
    return train_interactions.tocsr(), valid_interactions.tocsr(), \
           test_interactions.tocsr()
