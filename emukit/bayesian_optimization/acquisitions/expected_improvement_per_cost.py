from typing import Tuple, Union

import numpy as np

from ...core.interfaces import IModel, IDifferentiable
from ...core.acquisition import Acquisition
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement


class ExpectedImprovementPerCost(Acquisition):

    def __init__(self, model: Union[IModel, IDifferentiable], cost_model: Union[IModel, IDifferentiable],
                 jitter: np.float64 = np.float64(0))-> None:
        """
        This acquisition computes for a given input the improvement over the current best observed value in
        expectation. For more information see:

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization

        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """

        self.model = model
        self.cost_model = cost_model
        self.jitter = jitter

        self.ei = ExpectedImprovement(model, jitter)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        improvement = self.ei.evaluate(x)

        mean, _ = self.cost_model.predict(x)

        return improvement / mean

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the Expected Improvement and its derivative.

        :param x: locations where the evaluation with gradients is done.
        """

        improvement, dimprovement_dx = self.ei.evaluate_with_gradients(x)

        mean, _ = self.cost_model.predict(x)

        dmean_dx, _ = self.model.get_prediction_gradients(x)

        return improvement / mean, (dimprovement_dx * mean - dmean_dx * improvement) / (mean ** 2)

    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return True
