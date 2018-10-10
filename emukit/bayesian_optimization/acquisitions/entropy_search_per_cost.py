from typing import Union
import numpy as np

from ...core.acquisition import Acquisition
from ...core.interfaces import IModel, IDifferentiable
from ...core.parameter_space import ParameterSpace

from emukit.bayesian_optimization.acquisitions import EntropySearch
from emukit.bayesian_optimization.interfaces import IEntropySearchModel
from ..util.mcmc_sampler import McmcSampler


class EntropySearchPerCost(Acquisition):

    def __init__(self, model: Union[IModel, IDifferentiable, IEntropySearchModel], cost_model: IModel,
                 space: ParameterSpace, sampler: McmcSampler = None, num_samples: int = 400,
                 num_representer_points: int = 50, proposal_function: Acquisition = None,
                 burn_in_steps: int = 50) -> None:

        """
        Entropy Search acquisition function approximates the distribution of the global
        minimum and tries to decrease its entropy. See this paper for more details:

        P. Hennig and C. J. Schuler
        Entropy search for information-efficient global optimization
        Journal of Machine Learning Research, 13, 2012

        :param model: GP model to compute the distribution of the minimum dubbed pmin.
        :param space: Domain space which we need for the sampling of the representer points
        :param sampler: mcmc sampler for representer points
        :param num_samples: integer determining how many samples to draw for each candidate input
        :param num_representer_points: integer determining how many representer points to sample
        :param proposal_function: Function that defines an unnormalized log proposal measure from which to sample the
        representer points. The default is expected improvement.
        :param burn_in_steps: integer that defines the number of burn-in steps when sampling the representer points
        """
        super().__init__()

        self.cost_model = cost_model
        self.es = EntropySearch(model, space, sampler,
                                num_samples, num_representer_points,
                                proposal_function, burn_in_steps)

    def evaluate(self, x: np.ndarray):
        a = self.es.evaluate(x)
        cost, _ = self.cost_model.predict(x)
        return a / cost

    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return False
