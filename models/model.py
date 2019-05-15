from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import logging
import toolz
from tqdm import tqdm
import os
from evaluators import EvaluatorFactory
import math

logger = logging.getLogger(__name__)


class Model(object):
    """
    Base class for recommendation algorithm.

    Parameters
    ----------
    sess: tf.Session
        Session that the model will be attached with
    params: tensorflow.contrib.training import HParams
        Hyper parameters for the model
    """
    def __init__(self, sess, params):
        self.sess = sess
        self.learning_rate = getattr(params, 'learning_rate', 0.001)
        self.embedding_dim = getattr(params, 'embedding_dim', 32)
        self.model_dir = getattr(params, 'model_dir')
        self.n_users = getattr(params, 'n_users')
        self.n_items = getattr(params, 'n_items')
        self.initialized_std = getattr(params, 'initialized_std', 0.01)
        self.n_epochs = getattr(params, 'n_epochs', 20)
        self.eval_every_n_batches = getattr(params, 'eval_every_n_batches', 100)
        self.update_embeddings_every_n_batches = getattr(
            params, 'update_embeddings_every_n_batches', 100)
        self.max_to_keep = getattr(params, 'max_to_keep', 3)
        self.enable_spreadout_loss = getattr(
            params, 'enable_spreadout_loss', False)

        self.n_users_in_validation = getattr(
            params, 'n_users_in_validation', 1000)
        self.n_users_in_test = getattr(params, 'n_users_in_test', -1)
        self.n_users_in_chunk = getattr(params, 'n_users_in_chunk', 100)

        # optimizer
        self.optimizer = getattr(params, 'optimizer', 'adam')
        self.checkpoint = None
        self.scores = None

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_train_ops()
        self._create_score_items()
        self.sess.run(tf.global_variables_initializer())

    def fit(self, sampler, train_interactions, valid_interactions, eval_users,
            metrics, k=50):
        """
        Training the model
        :param sampler:
        :param train_interactions:
        :param valid_interactions:
        :param eval_users:
        :param metrics:
        :param k:
        :return:
        """
        saver = tf.train.Saver(max_to_keep=self.max_to_keep)

        n_evals_per_epoch = math.ceil(
            train_interactions.count_nonzero() * 1.0 /
            (self.eval_every_n_batches * sampler.batch_size))

        max_train_interaction_count = max(len(row)
                                          for row in train_interactions.rows)
        best_valid_scores = np.zeros(len(metrics))
        best_ep = -1

        metrics_path = '{}/metrics.csv'.format(self.model_dir)
        if os.path.isfile(metrics_path):
            os.remove(metrics_path)

        with open(metrics_path, 'w') as f:
            metrics_header = ','.join(['train_{}_{},val_{}_{}'.format(
                metric, k, metric, k) for metric in metrics])
            f.write('epoch,train_loss,{}\n'.format(metrics_header))

            for ep in range(1, self.n_epochs + 1):
                for it in range(n_evals_per_epoch):
                    losses = []
                    spreadout_losses = []

                    for b_idx in tqdm(range(self.eval_every_n_batches),
                                      desc="Optimizing {}/{}...".format(it + 1,
                                                                        ep)):
                        batch_samples = sampler.next_batch()
                        feed_dict = {
                            self.anchor_ids: batch_samples[0],
                            self.pos_ids: batch_samples[1],
                            self.neg_ids: batch_samples[2]
                        }

                        if self.enable_spreadout_loss:
                            _, spreadout_loss, loss = self.sess.run(
                                (self.train_ops, self.spreadout_loss, self.loss),
                                feed_dict=feed_dict)
                            spreadout_losses.append(spreadout_loss)
                        else:
                            _, loss = self.sess.run((self.train_ops, self.loss),
                                                    feed_dict=feed_dict)
                        losses.append(loss)
                        if self.update_embeddings_every_n_batches > 0 and \
                            b_idx > 0 and \
                            b_idx % self.update_embeddings_every_n_batches == 0:
                            self._update_embeddings()
                    log_info = '\nTraining loss: {}'.format(np.mean(losses))
                    if self.enable_spreadout_loss:
                        log_info = '{}, spreadout_loss: {}'.format(
                            log_info, np.mean(spreadout_losses))
                    logger.info(log_info)

                    # create evaluator on validation set
                    evaluators = [EvaluatorFactory.create_evaluator(
                        train_interactions, valid_interactions, metric)
                                  for metric in metrics]

                    # compute recall on train & validate set
                    train_eval_scores = np.empty(shape=(0, len(evaluators)),
                                                 dtype=np.float32)
                    valid_eval_scores = np.empty(shape=(0, len(evaluators)),
                                                 dtype=np.float32)
                    for user_chunk in toolz.partition_all(self.n_users_in_chunk,
                                                          eval_users):
                        # get topk items (more than k so that we can excluded
                        # items already in train)
                        reco_items = self._topk_items(
                            user_chunk, k=k + max_train_interaction_count)
                        train_eval_scores = np.append(train_eval_scores,
                                                      self._get_eval_scores(
                                                          reco_items,
                                                          evaluators, k,
                                                          on_train=True),
                                                      axis=0)

                        valid_eval_scores = np.append(valid_eval_scores,
                                                      self._get_eval_scores(
                                                          reco_items,
                                                          evaluators, k),
                                                      axis=0)

                    current_train_scores = np.mean(train_eval_scores, axis=0)
                    current_valid_scores = np.mean(valid_eval_scores, axis=0)
                    loss = 0.0 if not losses else np.mean(losses)

                    self._write_stats(f, ep, loss, current_train_scores,
                                      current_valid_scores)

                    logger.info('\n{} on (sampled) validation set at '
                                'iter/epoch #{}/{}: {}'.format(
                        '|'.join(metrics), it + 1, ep, current_valid_scores))

                    # save best recall model
                    if current_valid_scores[0] > best_valid_scores[0]:
                        # save new model
                        best_valid_scores = current_valid_scores
                        best_ep = ep
                        save_path = '{}/{}_{}-epoch_{}'.format(
                            self.model_dir, metrics[0],
                            self.__class__.__name__.lower(), ep)
                        saver.save(self.sess, save_path=save_path)
        logger.info('Best {} validation score: {}, on epoch {}'.format(
                    metrics[0], best_valid_scores, best_ep))

    def get_recommended_items(self, user_ids, train_user_items,
                              max_train_interaction_count, k=50):
        # load model
        if self.checkpoint is None:
            self._load_from_checkpoint()
        recommended_items = self._topk_items(user_ids,
                                             k=k + max_train_interaction_count)
        results = []
        for uid, pred_iids in recommended_items.items():
            top = []
            for iid in pred_iids:
                if iid not in train_user_items[uid] and len(top) < k:
                    top.append(iid)
            results.append(top)
        return results

    def _create_placeholders(self):
        self.is_training = tf.placeholder(dtype=tf.bool, shape=None,
                                          name='is_training')

    def _create_variables(self):
        raise NotImplementedError('_create_variables() method should be '
                                  'implemented in concrete model')

    def _create_loss(self):
        raise NotImplementedError('_create_loss() method should be '
                                  'implemented in concrete model')

    def _create_train_ops(self):
        raise NotImplementedError('_create_train_ops() method should be '
                                  'implemented in concrete model')

    def _create_score_items(self):
        raise NotImplementedError('score_items method should be '
                                  'implemented in concrete model')

    def _load_from_checkpoint(self):
        # restore the model if it already exists
        self.checkpoint = tf.train.get_checkpoint_state(self.model_dir)

        if self.checkpoint is not None:
            logger.info('Load {} model from {}'.format(self.__class__,
                                                       self.model_dir))
            self.build_graph()
            saver = tf.train.Saver(max_to_keep=self.max_to_keep)
            saver.restore(self.sess, self.checkpoint.model_checkpoint_path)

    def _topk_items(self, user_ids, k=50):
        """
        Get top k items for the list of users.

        Parameters
        ----------
        user_ids: list
            List of user ids
        k: int
            The number of items on the top of user preferences.
        Returns
        -------
            Top k items for each user.
        """
        _, topk = self.sess.run(tf.nn.top_k(self.scores, k),
                                feed_dict={self.eval_ids: user_ids})
        return dict(zip(user_ids, topk))

    @classmethod
    def _get_eval_scores(cls, reco_items, evaluators, k, on_train=False):
        """
        Get eval scores
        :param reco_items:
        :param evaluators:
        :param k:
        :param on_train:
        :return:
        """
        scores = np.zeros(shape=(len(reco_items), len(evaluators)))
        for i, evaluator in enumerate(evaluators):
            scores[:, i] = evaluator.eval(reco_items, k, on_train=on_train)

        return scores

    @classmethod
    def _write_stats(cls, f, ep, loss, current_train_scores,
                     current_valid_scores):
        metric_scores = ['{},{}'.format(train_score, val_score)
                         for train_score, val_score in
                         zip(current_train_scores, current_valid_scores)]
        f.write('{},{},{}\n'.format(ep, loss, ','.join(metric_scores)))
        f.flush()

    def _update_embeddings(self):
        """
        Update embeddings
        :return:
        """
        raise NotImplementedError('get_embeddings method should be implemented '
                                  'in concrete model')
