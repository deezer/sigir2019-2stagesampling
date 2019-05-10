import numpy as np
import logging
from samplers import Sampler

logger = logging.getLogger(__name__)

BETA_VOLUMN_CONST = 0.2228655673209082014292


class SpreadoutSampler(Sampler):
    """
    Spreadout weighted sampling

    Reference:
    W., Chao-Yuan, R., Manmatha, A. J., Smola, and P., Krahenbuhl.
    Sampling matters in deep embedding learning. Proc.ICCV. 2017.
    """
    def _candidate_neg_ids(self, pos_ids):
        """
        Candidate for negative ids
        :param pos_ids: batch positive ids
        :return:
        """
        return np.random.choice(range(self.interactions.shape[1]),
                                size=self.num_neg_candidates,
                                replace=False,
                                p=self.item_popularities)

    def _negative_sampling(self, user_ids, pos_ids, neg_ids):
        neg_samples = np.zeros(shape=(len(pos_ids), self.n_negatives))
        pos_item_embeddings = self.model.dense_item_embeddings[pos_ids]
        neg_item_embeddings = self.model.dense_item_embeddings[neg_ids]

        item_spreadout_distances = np.linalg.multi_dot([pos_item_embeddings,
                                                        neg_item_embeddings.T])

        for i, uid, in enumerate(user_ids):
            weights = self._calculate_weights(item_spreadout_distances[i])
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                weights = weights / weights_sum

            if weights_sum > 0:
                neg_samples[i] = np.random.choice(neg_ids,
                                                  size=self.n_negatives,
                                                  p=weights,
                                                  replace=False)
            else:
                neg_samples[i] = np.random.choice(neg_ids,
                                                  size=self.n_negatives,
                                                  replace=False)

            for j, neg in enumerate(neg_samples[i]):
                current_try = 0
                while neg in self.user_items[uid] and current_try < self.n_try:
                    if weights_sum > 0:
                        neg_samples[i, j] = neg = np.random.choice(neg_ids,
                                                                   p=weights)
                    else:
                        neg_samples[i, j] = neg = np.random.choice(neg_ids)
                    current_try += 1
        return neg_samples

    def _calculate_weights(self, spreadout_distance):
        mask = spreadout_distance > 0
        log_weights = (1.0 - (float(self.embedding_dim) - 1) / 2) * np.log(
            1.0 - np.square(spreadout_distance) + 1e-8) + np.log(
            BETA_VOLUMN_CONST)
        weights = np.exp(log_weights)
        weights[np.isnan(weights)] = 0.
        weights[~mask] = 0.
        weights_sum = np.sum(weights)

        if weights_sum > 0:
            weights = weights / weights_sum
        return weights
