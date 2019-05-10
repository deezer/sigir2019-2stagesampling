from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import scipy.sparse as sp


class Evaluator(object):
    """
    Evaluator for recommendation algorithms

    Parameters
    ----------
    train_interactions:
        The user-item pairs used in the training set.
        These pairs will be ignored in the recall calculation
    test_interactions:
        The held-out user-item pairs we make prediction against
    """
    def __init__(self, train_interactions, test_interactions):
        self.train_interactions = sp.lil_matrix(train_interactions)
        self.test_interactions = sp.lil_matrix(test_interactions)

        # number of users
        n_users = train_interactions.shape[0]

        self.train_user_items = {
            u: set(self.train_interactions.rows[u])
            for u in range(n_users) if self.train_interactions.rows[u]}

        self.test_user_items = {
            u: set(self.test_interactions.rows[u])
            for u in range(n_users) if self.test_interactions.rows[u]}

    def eval(self, reco_items, k=50, on_train=False):
        raise NotImplementedError('eval method should be implemented '
                                  'in concrete evaluator')
