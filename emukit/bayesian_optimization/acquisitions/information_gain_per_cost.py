from typing import Union
from copy import deepcopy
import scipy
import numpy as np

from ...core.acquisition import Acquisition
from ..acquisitions import EntropySearch
from ...core.interfaces import IModel, IDifferentiable
from ...core.parameter_space import ParameterSpace

from emukit.bayesian_optimization.acquisitions.expected_improvement import ExpectedImprovement
from ..interfaces import IEntropySearchModel
from ..util import epmgp
from ..util.mcmc_sampler import AffineInvariantEnsembleSampler, McmcSampler


class InformationGainPerCost(EntropySearch):

    def __init__(self, model: Union[IModel, IDifferentiable, IEntropySearchModel],
                 cost_model: Union[IModel, IDifferentiable, IEntropySearchModel],
                 space: ParameterSpace, sampler: McmcSampler = None,
                 num_samples: int = 400, num_representer_points: int = 50,
                 proposal_function: Acquisition = None, burn_in_steps: int = 50) -> None:

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

        if not isinstance(model, IEntropySearchModel):
            raise RuntimeError("Model is not supported for Entropy Search")

        self.model = model
        self.cost_model = cost_model
        self.space = space
        self.num_representer_points = num_representer_points
        self.burn_in_steps = burn_in_steps
        self.s_max = self.space.parameters[-1].max
        subspace = deepcopy(space)
        subspace.parameters.pop(-1)
        self.subspace = subspace
        if sampler is None:
            self.sampler = AffineInvariantEnsembleSampler(subspace)
        else:
            self.sampler = sampler

        # (unnormalized) density from which to sample the representer points to approximate pmin
        self.proposal_function = proposal_function
        if self.proposal_function is None:

            ei = ExpectedImprovement(model)

            def prop_func(x):

                if len(x.shape) == 1:
                    x_ = x[None, :]
                else:
                    x_ = x
                if self.subspace.check_points_in_domain(x_):

                    x_proj = np.append(x_, [self.s_max])[None, :]
                    a = ei.evaluate(x_proj)[0]
                    if np.isnan(a):
                        return np.array([np.NINF])
                    else:
                        return a
                else:
                    return np.array([np.NINF])

            self.proposal_function = prop_func

        # This is used later to calculate derivative of the stochastic part for the loss function
        # Derived following Ito's Lemma, see for example https://en.wikipedia.org/wiki/It%C3%B4%27s_lemma
        self.W = scipy.stats.norm.ppf(np.linspace(1. / (num_samples + 1),
                                                  1 - 1. / (num_samples + 1),
                                                  num_samples))[np.newaxis, :]

        # Initialize parameters to lazily compute them once needed
        self.representer_points = None
        self.representer_points_log = None
        self.logP = None
        self.need_update = True

    def _sample_representer_points(self) -> tuple:
        """ Samples a new set of representer points from the proposal measurement"""

        repr_points, repr_points_log = self.sampler.get_samples(self.num_representer_points, self.proposal_function,
                                                                self.burn_in_steps)

        if np.any(np.isnan(repr_points_log)) or np.any(np.isposinf(repr_points_log)):
            raise RuntimeError(
                "Sampler generated representer points with invalid log values: {}".format(repr_points_log))

        # Removing representer points that have 0 probability of being the minimum
        idx_to_remove = np.where(np.isneginf(repr_points_log))[0]
        if len(idx_to_remove) > 0:
            idx = list(set(range(self.num_representer_points)) - set(idx_to_remove))
            repr_points = repr_points[idx, :]
            repr_points_log = repr_points_log[idx]

        # project repr points
        repr_points = np.concatenate((repr_points, np.ones([repr_points.shape[0], 1]) * self.s_max), axis=1)

        return repr_points, repr_points_log


