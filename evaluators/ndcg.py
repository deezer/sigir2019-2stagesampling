from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from evaluators.evaluator import Evaluator
import numpy as np


class NDCGEvaluator(Evaluator):
    """
    nDCG@k score evaluator.
    """

    @classmethod
    def dcg_at_k(cls, r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.

    def __init__(self, train_interactions, test_interactions):
        super(NDCGEvaluator, self).__init__(train_interactions, test_interactions)

    def eval(self, reco_items, k=50, on_train=False):
        """
        Compute the Top-K nDCG for a particular user given the predicted scores to items.
        :param reco_items:
        :param k:
        :param on_train:
        :return: ndcg@k
        """
        ndcg = []
        for user_id, tops in reco_items.items():
            train_set = self.train_user_items.get(user_id, set())
            test_set = self.test_user_items.get(user_id, set())
            ref_set = train_set if on_train else test_set
            top_n_items = 0
            user_hits = []
            for i in tops:
                # ignore item in the training set
                if not on_train and i in train_set:
                    continue
                pred = 1 if i in ref_set else 0
                user_hits.append(pred)
                top_n_items += 1
                if top_n_items == k:
                    break
            user_hits = np.array(user_hits, dtype=np.float32)
            if len(ref_set) >= k:
                ideal_rels = np.ones(k)
            else:
                ideal_rels = np.pad(np.ones(len(ref_set)), (0, k - len(ref_set)), 'constant')
            ndcg.append(self.dcg_at_k(user_hits, k) / self.dcg_at_k(ideal_rels, k))
        return ndcg
