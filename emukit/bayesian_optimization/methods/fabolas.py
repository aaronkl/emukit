import numpy as np

from copy import deepcopy
from typing import List, Callable
from ...core.parameter_space import ParameterSpace, ContinuousParameter
from ...core.loop import FixedIterationsStoppingCondition, UserFunctionWithCostWrapper, Sequential
from ..acquisitions import EntropySearchPerCost
from ..loops import CostSensitiveBayesianOptimizationLoop
from emukit.multi_fidelity.models.fabolas_model import FabolasModel, quad, linear
from emukit.experimental_design.model_free.random_design import RandomDesign
from emukit.core.optimization import DirectOptimizer
from emukit.core.loop import UserFunctionWithCostWrapper, UserFunctionResult, UserFunction


class Fabolas(CostSensitiveBayesianOptimizationLoop):
    def __init__(self, user_function: UserFunctionWithCostWrapper, space: ParameterSpace,
                 s_min, s_max, n_init: int = 20) -> None:

        """
        Fast Bayesian optimization on large datasets

        :param space: contains the definition of the variables of the input space.

        """
        self.s_min = s_min
        self.s_max = s_max
        self.incumbents = []
        self.user_function = user_function

        init_design = RandomDesign(space)
        X_init = init_design.get_samples(n_init)
        subsets = [s_max / s_sub for s_sub in [256, 128, 64, 32]]
        # TODO: check that subsets are larger than smin
        # s = np.array(subsets)
        s = np.zeros([n_init])
        for i in range(n_init):
            # s[i] = transform(subsets[i % len(subsets)], s_min, s_max)
            s[i] = subsets[i % len(subsets)]
        X_init = np.append(X_init, s[:, None], axis=1)

        C_init = []
        Y_init = []

        curr_inc = None
        curr_inc_val = np.inf
        for x in X_init:
            res = self.user_function.evaluate(x[None, :])
            C_init.append(res[0].C)
            Y_init.append(res[0].Y)

            if res[0].Y < curr_inc_val:
                curr_inc_val = res[0].Y
                curr_inc = x[:-1]
            self.incumbents.append(curr_inc)

        space.parameters.append(ContinuousParameter("s", s_min, s_max))

        Y_init = np.array(Y_init)
        C_init = np.array(C_init)

        model_objective = FabolasModel(X_init, Y_init, s_min, s_max, basis_func=quad)
        model_cost = FabolasModel(X_init, C_init, s_min, s_max, basis_func=linear)
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

    def custom_step(self):
        # identify incumbent
        proj_X = deepcopy(self.loop_state.X)
        proj_X[:, -1] = np.ones(proj_X.shape[0]) * self.s_max
        mean_full_dataset, _ = self.model_updaters[0].model.predict(proj_X)
        best = np.argmin(mean_full_dataset, axis=0)
        self.incumbents.append(proj_X[best, :-1][0])
