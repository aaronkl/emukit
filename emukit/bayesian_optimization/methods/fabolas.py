import numpy as np

from ...core.parameter_space import ParameterSpace, ContinuousParameter
from ...core.loop import FixedIterationsStoppingCondition, UserFunctionWithCostWrapper
from ..acquisitions import EntropySearchPerCost
from ..loops import CostSensitiveBayesianOptimizationLoop
from emukit.multi_fidelity.models.fabolas_model import FabolasModel, quad, linear
from emukit.experimental_design.model_free.random_design import RandomDesign


class Fabolas(CostSensitiveBayesianOptimizationLoop):
    def __init__(self, user_function: UserFunctionWithCostWrapper, space: ParameterSpace, s_min, s_max, n_init: int = 20) -> None:

        """
        Fast Bayesian optimization on large datasets

        :param space: contains the definition of the variables of the input space.

        """

        self.user_function = user_function

        init_design = RandomDesign(space)
        X_init = init_design.get_samples(n_init)

        subsets = [512, 256, 128, 64]
        s = np.log(np.array([[s_max / subsets[i % len(subsets)] for i in range(X_init.shape[0])]]))
        X_init = np.append(X_init, s.T, axis=1)
        Y_init, C_init = self.user_function.f(X_init)
        space.parameters.append(ContinuousParameter("s", np.log(s_min), np.log(s_max)))

        model_objective = FabolasModel(X_init, Y_init, basis_func=quad)
        model_cost = FabolasModel(X_init, C_init, basis_func=linear)
        es = EntropySearchPerCost(model=model_objective, cost_model=model_cost, space=space)

        super(Fabolas, self).__init__(X_init=X_init, Y_init=Y_init, C_init=C_init, space=space,
                                      acquisition=es,
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
