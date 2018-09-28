import DIRECT
import numpy as np

from .. import ParameterSpace
from ..acquisition import Acquisition


class DirectOptimizer(object):
    """ Optimizes the acquisition function with DIRECT"""

    def __init__(self, space: ParameterSpace, n_iters: int = 500, n_func_evals: int = 400, **kwargs) -> None:
        b = np.array(space.convert_to_gpyopt_design_space().get_bounds())
        self.lower = b[:, 0]
        self.upper = b[:, 1]
        self.n_iters = n_iters
        self.n_func_evals = n_func_evals

    def _direct_acquisition_fkt_wrapper(self, acq_f):
        def _l(x, user_data):
            return -acq_f.evaluate(np.array([x])), 0

        return _l

    def optimize(self, acquisition: Acquisition):
        """
        acquisition - The acquisition function to be optimized
        """
        x, _, _ = DIRECT.solve(self._direct_acquisition_fkt_wrapper(acquisition),
                               l=[self.lower],
                               u=[self.upper],
                               maxT=self.n_iters,
                               maxf=self.n_func_evals)
        return x[None, :], None
