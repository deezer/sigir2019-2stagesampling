from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize
from models.model import Model

logger = logging.getLogger(__name__)


_SUPPORTED_OPTIMIZERS = {
    'adam': tf.train.AdamOptimizer,
    'sgd': tf.train.GradientDescentOptimizer
}


class Triplet(Model):
    """
    Collaborative Metric Learning based on triplet loss.

    Adapted from original code here:
    https://github.com/changun/CollMetric

    References
    ----------
    [1]. Hsieh, C.-K., Yang, L. et al., Collaborative Metric Learning,
    Proceedings of the 26th WWW, 2017.
    """
    def __init__(self, sess, params):
        super(Triplet, self).__init__(sess, params)
        self.spreadout_weight = getattr(params, 'spreadout_weight', 1.0)
        self.clip_norm = getattr(params, 'clip_norm', 1.0)
        self.activate_l2_norm = getattr(params, 'activate_l2_norm', False)
        self.margin = getattr(params, 'margin', 1.0)
        self.n_negatives = getattr(params, 'n_negatives', 1)
        self.dense_user_embeddings = np.random.normal(
            0.0, self.initialized_std,
            size=[self.n_users, self.embedding_dim]).astype(np.float32)
        self.dense_user_embeddings = normalize(self.dense_user_embeddings,
                                               axis=1, norm='l2')

        self.dense_item_embeddings = np.random.normal(
            0.0, self.initialized_std,
            size=[self.n_items, self.embedding_dim]).astype(np.float32)
        self.dense_item_embeddings = normalize(self.dense_item_embeddings,
                                               axis=1, norm='l2')

    def _create_placeholders(self):
        logger.debug('--> Define Triplet placeholders')
        self.anchor_ids = tf.placeholder(name='anchor_ids', dtype=tf.int32,
                                         shape=[None])
        self.pos_ids = tf.placeholder(name='pos_ids', dtype=tf.int32,
                                      shape=[None])
        self.neg_ids = tf.placeholder(name='neg_ids', dtype=tf.int32,
                                      shape=[None, self.n_negatives])

        # list of users for validation
        self.eval_ids = tf.placeholder(tf.int32, [None])

    def _create_variables(self):
        logger.debug('--> Define Triplet variables')
        # user embeddings
        self.user_embeddings = tf.get_variable(
            name='user_embedding_matrix',
            initializer=self.dense_user_embeddings,
            dtype=tf.float32
        )

        # item embeddings
        self.item_embeddings = tf.get_variable(
            name='item_embedding_matrix',
            initializer=self.dense_item_embeddings,
            dtype=tf.float32
        )

        if self.activate_l2_norm:
            logger.debug('----> Active L2 normalization')
            self.user_embeddings = tf.math.l2_normalize(self.user_embeddings,
                                                        axis=1)
            self.item_embeddings = tf.math.l2_normalize(self.item_embeddings,
                                                        axis=1)

    def _pos_distances(self):
        logger.debug('--> Define Triplet positive distances')
        distances = tf.reduce_sum(
            tf.squared_difference(self.anchors, self.positives),
            axis=1,
            name='pos_distances')
        return tf.maximum(distances, 0.0)

    def _neg_distances(self):
        logger.debug('--> Define Triplet negative distances')
        expanded_anchors = tf.expand_dims(self.anchors, -1,
                                          name='expanded_anchors')
        distances = tf.reduce_sum(
            tf.squared_difference(expanded_anchors, self.negatives),
            axis=1,
            name='neg_distances')
        return tf.maximum(distances, 0.0)

    def _create_loss(self):
        logger.debug('--> Define Triplet loss')
        self.loss = self._embedding_loss()
        if self.enable_spreadout_loss:
            self.spreadout_loss = self._spread_out_loss()
            self.loss += self.spreadout_loss

    def _embedding_loss(self):
        logger.debug('--> Define embedding loss')
        # anchor user embedding: shape=(N, K)
        self.anchors = tf.nn.embedding_lookup(self.user_embeddings,
                                              self.anchor_ids,
                                              name='batch_anchor_embeddings')
        # positive item embedding: shape=(N, K)
        self.positives = tf.nn.embedding_lookup(
            self.item_embeddings,
            self.pos_ids,
            name='batch_positive_embeddings')
        # negative item embedding: shape=(N, K, W)
        negatives = tf.nn.embedding_lookup(self.item_embeddings, self.neg_ids)

        self.negatives = tf.transpose(negatives, (0, 2, 1),
                                      name='batch_negative_embeddings')
        # positive distances
        self.pos_distances = self._pos_distances()

        # negative distances
        self.neg_distances = self._neg_distances()

        # get only the closest negative distance to the anchor
        min_neg_distances = tf.reduce_min(self.neg_distances, axis=1,
                                          name='min_neg_distances')

        loss = tf.maximum(self.pos_distances - min_neg_distances + self.margin,
                          0.0, name="pair_loss")

        return tf.reduce_sum(loss, name='embedding_loss')

    def _spread_out_loss(self):
        logger.debug('--> Define Triplet spreadout loss')
        expanded_anchors = tf.expand_dims(self.anchors, -1,
                                          name='expanded_spreadout_anchors')

        non_match_anchor_dot = tf.matmul(expanded_anchors,
                                         self.negatives,
                                         transpose_a=True)

        expanded_positives = tf.expand_dims(self.positives, -1,
                                            name='expanded_spreadout_positives')
        non_match_positive_dot = tf.matmul(expanded_positives,
                                           self.negatives,
                                           transpose_a=True)

        return self._spread_out_weights(non_match_anchor_dot) + \
               self._spread_out_weights(non_match_positive_dot)

    def _spread_out_weights(self, non_match_dot):
        non_match_dot = tf.reduce_max(tf.reshape(
            non_match_dot, shape=[-1, self.n_negatives]), axis=1)

        m1 = tf.reduce_sum(non_match_dot)
        m2 = tf.reduce_sum(tf.square(non_match_dot))
        n_samples = tf.cast(tf.shape(self.anchor_ids)[0], tf.float32)
        return tf.maximum(
            (tf.square(m1) + tf.maximum(0.0, m2 - n_samples /
                                        self.embedding_dim)) *
            self.spreadout_weight, 0.)

    def _create_train_ops(self):
        logger.debug('--> Define Triplet train ops')
        self.train_ops = self._optimizer_ops()

    def _clip_by_norm_ops(self):
        return [tf.assign(self.user_embeddings, tf.clip_by_norm(
            self.user_embeddings, self.clip_norm, axes=[1])),
                tf.assign(self.item_embeddings, tf.clip_by_norm(
                    self.item_embeddings, self.clip_norm, axes=[1]))]

    def _optimizer_ops(self):
        logger.debug('--> Define Triplet Optimizer ops')
        try:
            ops = [_SUPPORTED_OPTIMIZERS[self.optimizer](
                self.learning_rate).minimize(self.loss)]
        except KeyError:
            raise KeyError('Current Triplet implementation does not support '
                           'optimizer: {}'.format(self.optimizer))
        return ops

    def _create_score_items(self):
        logger.debug('--> Define Triplet ranking scores')
        # get users need to score
        users = tf.expand_dims(self.anchors, 1)

        # reshape items to (1, M, K)
        items = tf.expand_dims(self.item_embeddings, 0)

        # scores = minus distance (N, M)
        self.scores = -tf.reduce_sum(tf.squared_difference(users, items),
                                     axis=2, name='scores')

    def _update_embeddings(self):
        self.dense_user_embeddings, self.dense_item_embeddings = \
            self.sess.run([self.user_embeddings, self.item_embeddings])
