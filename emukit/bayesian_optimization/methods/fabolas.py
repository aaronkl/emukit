import numpy as np
from typing import List, Callable
from ...core.parameter_space import ParameterSpace, ContinuousParameter
from ...core.loop import FixedIterationsStoppingCondition, UserFunctionWithCostWrapper, Sequential
from ..acquisitions import EntropySearchPerCost
from ..loops import CostSensitiveBayesianOptimizationLoop
from emukit.multi_fidelity.models.fabolas_model import FabolasModel, quad, linear
from emukit.experimental_design.model_free.random_design import RandomDesign
from emukit.core.optimization import DirectOptimizer
from emukit.core.loop import UserFunctionWithCostWrapper, UserFunctionResult, UserFunction


def transform(s, s_min, s_max):
    s_transform = (np.log2(s) - np.log2(s_min)) / (np.log2(s_max) - np.log2(s_min))
    return s_transform


def retransform(s_transform, s_min, s_max):
    s = np.rint(2**(s_transform * (np.log2(s_max) - np.log2(s_min)) + np.log2(s_min)))
    return s


class FabolasUserFunctionWrapper(UserFunction):

    def __init__(self, wrapper: UserFunctionWithCostWrapper, s_min: int, s_max: int):
        """
        :param f: A python function that takes in a 2d numpy ndarray of inputs and returns a 2d numpy ndarray of outputs.
        """
        self.s_min = s_min
        self.s_max = s_max
        self.wrapper = wrapper

    def evaluate(self, inputs: np.ndarray) -> List[UserFunctionResult]:

        inputs[:, -1] = retransform(inputs[:, -1], self.s_min, self.s_max)
        return self.wrapper.evaluate(inputs)


class Fabolas(CostSensitiveBayesianOptimizationLoop):
    def __init__(self, user_function: UserFunctionWithCostWrapper, space: ParameterSpace,
                 s_min, s_max, n_init: int = 20) -> None:

        """
        Fast Bayesian optimization on large datasets

        :param space: contains the definition of the variables of the input space.

        """

        self.user_function = FabolasUserFunctionWrapper(user_function, s_min, s_max)

        init_design = RandomDesign(space)
        X_init = init_design.get_samples(n_init)
        subsets = [s_max / s_sub for s_sub in [512, 256, 128, 64]]

        s = np.zeros([n_init])
        for i in range(n_init):
            s[i] = transform(subsets[i % len(subsets)], s_min, s_max)

        X_init = np.append(X_init, s[:, None], axis=1)
        res = self.user_function.evaluate(X_init)
        C_init = np.array([ri.C for ri in res])
        Y_init = np.array([ri.Y for ri in res])

        space.parameters.append(ContinuousParameter("s", 0, 1))

        model_objective = FabolasModel(X_init, Y_init, basis_func=quad)
        model_cost = FabolasModel(X_init, C_init, basis_func=linear)
        es = EntropySearchPerCost(model=model_objective, cost_model=model_cost, space=space)
        acquisition_optimizer = DirectOptimizer(space)

        candidate_point_calculator = Sequential(es, acquisition_optimizer)

        super(Fabolas, self).__init__(X_init=X_init, Y_init=Y_init, C_init=C_init, space=space,
                                      acquisition=es, candidate_point_calculator=candidate_point_calculator,
                                      model_objective=model_objective, model_cost=model_cost)

    def suggest_new_locations(self):
        """ Returns one or a batch of locations without evaluating the objective """
        return self.candidate_point_calculator.compute_next_points(self.loop_state)[0].X

    def run_optimization(self, num_iterations: int) -> None:
        """
        :param user_function: The function that we want to optimize
        :param num_iterations: The number of iterations to run the Bayesian optimization loop.
        """
        self.run_loop(self.user_function, FixedIterationsStoppingCondition(num_iterations))
