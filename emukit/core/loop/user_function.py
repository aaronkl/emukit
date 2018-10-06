"""
This file contains the "UserFunction" base class and implementations

The user function is the objective function in optimization, the integrand in quadrature or the function to be learnt
in experimental design.
"""

import abc
from typing import Callable, List

import numpy as np

from .user_function_result import UserFunctionResult


class UserFunction(abc.ABC):
    """ The user supplied function is interrogated as part of the outer loop """
    @abc.abstractmethod
    def evaluate(self, X: np.ndarray) -> List[UserFunctionResult]:
        pass


class UserFunctionWrapper(UserFunction):
    """ Wraps a user-provided python function. """
    def __init__(self, f: Callable):
        """
        :param f: A python function that takes in a 2d numpy ndarray of inputs and returns a 2d numpy ndarray of outputs.
        """
        self.f = f

    def evaluate(self, inputs: np.ndarray) -> List[UserFunctionResult]:
        """
        Evaluates python function by providing it with numpy types and converts the output
        to a List of UserFunctionResults

        :param inputs: List of function inputs at which to evaluate function
        :return: List of function results
        """
        if inputs.ndim != 2:
            raise ValueError("User function should receive 2d array as an input, actual input dimensionality is {}".format(inputs.ndim))

        outputs = self.f(inputs)

        if outputs.ndim != 2:
            raise ValueError("User function should return 2d array as an output, actual output dimensionality is {}".format(outputs.ndim))

        results = []
        for x, y in zip(inputs, outputs):
            results.append(UserFunctionResult(x, y))
        return results


class UserFunctionWithCostWrapper(UserFunction):
    def __init__(self, f: Callable):
        """
        Wraps a user-provided python function, which returns the function value as well as
        the cost for evaluating the function.

        :param f: A python function that takes in a 2d numpy ndarray of inputs and returns
            two 2d numpy ndarray for the actual function value and the cost.
        """
        self.f = f

    def evaluate(self, inputs: np.ndarray) -> List[UserFunctionResult]:
        """
        Evaluates python function by providing it with numpy types and converts the output to a
        List of UserFunctionResults

        :param inputs: List of function inputs at which to evaluate function
        :return: List of function results
        """
        if inputs.ndim != 2:
            raise ValueError("User function should receive 2d array as an input, actual input dimensionality is {}".format(inputs.ndim))

        outputs, costs = self.f(inputs)

        if outputs.ndim != 2:
            raise ValueError("User function should return 2d array as an output, "
                             "actual output dimensionality is {}".format(outputs.ndim))

        if costs.ndim != 2:
            raise ValueError("User function should return 2d array as for costs, "
                             "actual dimensionality is {}".format(costs.ndim))

        results = []
        for x, y, c in zip(inputs, outputs, costs):
            results.append(UserFunctionResult(x, y, c))
        return results
