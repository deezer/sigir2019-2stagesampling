from samplers.sampler import Sampler
from samplers.uniform_sampler import UniformSampler
from samplers.popular_sampler import PopularSampler
from samplers.spreadout_sampler import SpreadoutSampler

_SUPPORTED_SAMPLERS = {
    'uniform': UniformSampler,
    'popular': PopularSampler,
    'spreadout': SpreadoutSampler
}


class SamplerFactory(object):
    @classmethod
    def generate_sampler(cls, sampler_name, interactions,
                         n_negatives=None, batch_size=None, n_workers=5,
                         **kwargs):
        """
        Generate a sampler
        :param sampler_name:
        :param interactions:
        :param n_negatives:
        :param batch_size:
        :param n_workers:
        :param kwargs:
        :return:
        """
        try:
            spl = _SUPPORTED_SAMPLERS[sampler_name](sampler_name, interactions,
                                                    n_negatives, batch_size,
                                                    n_workers, **kwargs)
            return spl
        except KeyError:
            raise KeyError('Do not support sampler {}'.format(sampler_name))
