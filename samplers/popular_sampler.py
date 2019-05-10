import numpy as np
import logging
from samplers import Sampler

logger = logging.getLogger(__name__)


class PopularSampler(Sampler):
    """
    Sampler based on popularity.
    The negative examples are chosen from all items
    """

    def _candidate_neg_ids(self, pos_ids):
        """
        Candidate for negative ids
        :param pos_ids: batch positive ids
        :return:
        """
        return np.arange(self.interactions.shape[1])

    def _negative_sampling(self, user_ids, pos_ids, neg_ids):
        neg_samples = np.random.choice(neg_ids,
                                       size=(len(pos_ids), self.n_negatives),
                                       replace=False,
                                       p=self.item_popularities)
        for i, uid, negatives in zip(range(len(user_ids)),
                                     user_ids, neg_samples):
            for j, neg in enumerate(negatives):
                while neg in self.user_items[uid]:
                    neg_samples[i, j] = neg = np.random.choice(
                        neg_ids, p=self.item_popularities)
        return neg_samples
