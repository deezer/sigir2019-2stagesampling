from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from evaluators.evaluator import Evaluator


class MAPEvaluator(Evaluator):
    """
    MAP@k score evaluator.
    """
    def __init__(self, train_interactions, test_interactions):
        super(MAPEvaluator, self).__init__(train_interactions, test_interactions)

    @classmethod
    def user_ap_at_k(cls, label_iids, pred_iids, k):
        if len(pred_iids) > k:
            pred_iids = pred_iids[:k]
        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(pred_iids):
            if p in label_iids and p not in pred_iids[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not label_iids:
            return 0.0

        return score / min(len(label_iids), k)

    def eval(self, reco_items, k=50, on_train=False):
        """
        Compute the Top-K MAP
        :param reco_items:
        :param k:
        :param on_train:
        :return: map@k
        """
        # calculate the map
        final_ap = []
        for user_id, pred_iids in reco_items.items():
            train_set = self.train_user_items.get(user_id, set())
            test_set = self.test_user_items.get(user_id, set())
            ref_set = train_set if on_train else test_set
            if on_train:
                top = [iid for i, iid in enumerate(pred_iids) if i < k]
            else:
                top = []
                for iid in pred_iids:
                    if iid not in train_set and len(top) < k:
                        top.append(iid)
            user_ap = self.user_ap_at_k(ref_set, top, k)
            final_ap.append(user_ap)
        return np.array(final_ap)
