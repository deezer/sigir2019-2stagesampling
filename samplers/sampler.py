from multiprocessing import Process, Queue
import logging
import numpy as np
from collections import defaultdict
logger = logging.getLogger(__name__)


class Sampler:
    """
    A sampler is responsible for triplet sampling within a specific strategy
    :param name: sampler name
    :param model: current training model
    :param interactions: input user interactions in
           scipy.sparse.lil_matrix format
    :param n_workers: number of workers
    :param n_negatives: number of negatives
    :param batch_size: batch size
    :param kwargs: optional keyword arguments
    """

    @classmethod
    def _get_popularity(cls, interactions, max_count=None):
        logger.debug('Get Item Popularity')
        popularity_dict = defaultdict(set)
        for uid, iids in enumerate(interactions.rows):
            for iid in iids:
                popularity_dict[iid].add(uid)

        popularity_dict = {
            key: max_count if max_count is not None and 0 < max_count < len(val)
            else len(val) for key, val in popularity_dict.items()
        }
        return popularity_dict

    def __init__(self, name, interactions, n_negatives=-1,
                 batch_size=None, n_workers=5, **kwargs):
        self.name = name
        self.interactions = interactions
        self.n_negatives = n_negatives
        self.batch_size = batch_size
        self.neg_alpha = 1.0
        self.result_queue = Queue(maxsize=n_workers * 2)
        self.processors = []
        if kwargs is not None:
            self.__dict__.update(kwargs)

        # user positive item dictionary, use to check
        # if an item is a positive one
        self.user_items = {uid: set(iids) for uid, iids in enumerate(
            self.interactions.rows)}

        # get item popularities
        if 'pop' in self.name or 'spreadout' in self.name:
            self.item_counts = self._get_popularity(interactions,
                                                    max_count=self.max_count)

            if self.neg_alpha != 1.0:
                logger.debug('NEG ALPHA: {}'.format(self.neg_alpha))
                self.item_counts = {iid: np.power(freq, self.neg_alpha)
                                    for iid, freq in self.item_counts.items()}

            total_count = np.sum(list(self.item_counts.values()))

            self.item_popularities = np.zeros(interactions.shape[1],
                                              dtype=np.float32)
            for iid in range(interactions.shape[1]):
                if iid in self.item_counts:
                    self.item_popularities[iid] = float(
                        self.item_counts[iid]) / total_count
                else:
                    self.item_popularities[iid] = 0.0

        for i in range(n_workers):
            self.processors.append(Process(target=self.sampling,
                                           args=(self.result_queue, i+1)))
            self.processors[-1].start()

    def next_batch(self):
        """
        Get next batch training samples
        :return: None
        """
        return self.result_queue.get()

    def close(self):
        """
        Close the sampler
        :return: None
        """
        for p in self.processors:
            p.terminate()
            p.join()

    def sampling(self, result_queue, random_seed):
        """
        Sampling a batch of training samples and put it into result_queue
        :param result_queue:
        :param random_seed:
        :return: batch (user, pos_item, neg_items)
        """
        rng = np.random.RandomState(random_seed)

        # positive user item pairs
        user_positive_item_pairs = np.asarray(self.interactions.nonzero()).T

        # number of batch per iteration
        n_batches_per_epoch = int(
            len(user_positive_item_pairs) / self.batch_size)

        while True:
            rng.shuffle(user_positive_item_pairs)
            # for each batch
            for batch_index in range(n_batches_per_epoch):
                batch_samples = self._batch_sampling(user_positive_item_pairs,
                                                     batch_index)
                result_queue.put(batch_samples)

    def _batch_sampling(self, user_positive_item_pairs, batch_index):
        batch_user_anchor_items_pairs = user_positive_item_pairs[
                                        batch_index * self.batch_size:
                                        (batch_index + 1) * self.batch_size, :]

        # generate batch users
        batch_user_ids = np.array(
            [uid for uid, _ in batch_user_anchor_items_pairs])

        # generate batch positives
        batch_pos_ids = np.array(
            [iid for _, iid in batch_user_anchor_items_pairs])

        # preselect n_negative_candidates items due to large number of items
        candidate_neg_ids = self._candidate_neg_ids(batch_pos_ids)

        # generate batch negatives
        batch_neg_ids = self._negative_sampling(batch_user_ids, batch_pos_ids,
                                                candidate_neg_ids)
        return batch_user_ids, batch_pos_ids, batch_neg_ids

    def _candidate_neg_ids(self, pos_ids):
        """
        Candidate for negative ids
        :param pos_ids: batch positive ids
        :return:
        """
        raise NotImplementedError(
            '_candidate_neg_ids method should be implemented in child class')

    def _negative_sampling(self, user_ids, pos_ids, neg_ids):
        """
        Negative sampling
        :param user_ids:
        :param pos_ids:
        :param neg_ids:
        :return:
        """
        raise NotImplementedError(
            '_negative_sampling method should be implemented in child class')
