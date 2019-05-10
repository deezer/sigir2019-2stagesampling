# fetch_echonest('data/echonest', min_playcount=10, min_interactions=10)
# DEBUG : Number of users: 64774
# DEBUG : Number of items: 130791
# DEBUG : Number of interactions: 1103040
# DEBUG : Mapping interactions density: 0.013020050972692179

import pandas as pd
import os
import logging
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def _filter_data(data, key, min_inter):
    """
    Filter out data by minimum number of items or users
    :param data: input dataframe
    :param key: user_id | item_id
    :param min_inter:
    :return: filtered data
    """
    notkey = 'user_ID' if key == 'song_ID' else 'song_ID'
    min_inter_data = data[['user_ID', 'song_ID']].groupby(key) \
        .count() \
        .reset_index() \
        .rename(columns={notkey: 'count'}) \
        .query('count >= %i' % min_inter)[key].tolist()
    min_inter_data = {rid: 1 for rid in min_inter_data}
    data = data[data[key].isin(min_inter_data)]
    return data


def _build_interactions(data, min_interactions):
    """
    Build interaction matrix
    :param data:
    :param min_interactions:
    :return:
    """
    logger.debug('Filter users has less than {} '
                 'interactions'.format(min_interactions))
    data = _filter_data(data, 'user_ID', min_interactions)

    # create mappings
    user_ids = data.user_ID.unique()
    item_ids = data.song_ID.unique()
    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_id_map = {iid: idx for idx, iid in enumerate(item_ids)}

    data.loc[:, 'user_id_map'] = data.user_ID.apply(lambda x: user_id_map[x])
    data.loc[:, 'item_id_map'] = data.song_ID.apply(lambda x: item_id_map[x])

    # convert to sparse and save
    interaction_mat = sp.dok_matrix((len(user_ids), len(item_ids)),
                                    dtype=np.float32)
    interaction_mat[data.user_id_map.tolist(), data.item_id_map.tolist()] = 1

    return interaction_mat, user_ids, item_ids


def fetch_echonest(data_home, min_playcount=5, min_interactions=10):
    """
    Fetch echonest dataset
    :param data_home:
    :param min_playcount:
    :param min_interactions:
    :return:
    """
    echo_data_path = os.path.join(data_home, 'train_triplets.txt')
    echo_interactions_path = os.path.join(
        data_home,
        'interactions_pltcnt{}_minint{}.npz'.format(
            min_playcount, min_interactions))
    echo_user_item_ids_path = os.path.join(
        data_home,
        'user_item_ids_pltcnt{}_minint{}.npz'.format(
            min_playcount, min_interactions))

    if not os.path.exists(echo_interactions_path) or \
            not os.path.exists(echo_user_item_ids_path):
        # read raw data
        data = pd.read_csv(echo_data_path,
                           sep='\t',
                           names=['user_ID', 'song_ID', 'play_count'])
        # filter out only tracks with playcount more than min_playcount
        data = data[data.play_count >= min_playcount]
        interactions, user_ids, item_ids = _build_interactions(data,
                                                               min_interactions)

        sp.save_npz(echo_interactions_path, interactions.tocsr())
        np.savez(echo_user_item_ids_path, user_ids=user_ids, item_ids=item_ids)
    else:
        interactions = sp.load_npz(echo_interactions_path)
        user_item_ids = np.load(echo_user_item_ids_path)
        user_ids = user_item_ids['user_ids']
        item_ids = user_item_ids['item_ids']

    logger.debug('Number of users: {}'.format(len(user_ids)))
    logger.debug('Number of items: {}'.format(len(item_ids)))
    logger.debug(
        'Number of interactions: {}'.format(interactions.count_nonzero()))
    logger.debug('Mapping interactions density: {}'.format(
        float(interactions.count_nonzero()) * 100 / (
                len(user_ids) * len(item_ids))))
    return {
        'interactions': interactions,
        'user_ids': user_ids,
        'item_ids': item_ids
    }


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG)
    fetch_echonest(data_home='data/echonest',
                   min_playcount=10,
                   min_interactions=10)
