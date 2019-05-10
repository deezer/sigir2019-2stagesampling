from evaluators.evaluator import Evaluator
from evaluators.ndcg import NDCGEvaluator
from evaluators.map import MAPEvaluator


class EvaluatorFactory(object):
    """
    Evaluator factory.
    """
    @classmethod
    def create_evaluator(cls, train_interactions, test_interactions,
                         metric='ndcg'):
        if metric == 'ndcg':
            eva = NDCGEvaluator(train_interactions, test_interactions)
        elif metric == 'map':
            eva = MAPEvaluator(train_interactions, test_interactions)
        else:
            raise ValueError('does not support evaluator: {}'.format(metric))
        return eva
