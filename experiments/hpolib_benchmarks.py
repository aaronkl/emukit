import os
import json
import argparse
import numpy as np

from hpolib.benchmarks.surrogates.svm import SurrogateSVM

from emukit.models.bo_gp import BOGP
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop, CostSensitiveBayesianOptimizationLoop
from emukit.core.loop import FixedIterationsStoppingCondition, UserFunctionWithCostWrapper, UserFunctionResult, \
    Sequential, LoopState
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement, ExpectedImprovementPerCost
from emukit.experimental_design.model_free.random_design import RandomDesign
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.optimization import DirectOptimizer
from emukit.bayesian_optimization.methods.fabolas import Fabolas


parser = argparse.ArgumentParser()
parser.add_argument("--method", default="gp_ei", type=str, nargs="?")
parser.add_argument("--benchmark", default="svm_mnist_surrogate", type=str, nargs="?")
parser.add_argument("--num_iterations", default=10, type=int, nargs="?")
parser.add_argument("--output_path", default=".", type=str, nargs="?")
parser.add_argument("--run_id", default=0, type=int, nargs="?")
parser.add_argument("--n_init", default=2, type=int, nargs="?")

args = parser.parse_args()

if args.benchmark == "svm_mnist_surrogate":
    b = SurrogateSVM()
    s_min = b.s_min
    s_max = 50000


def evaluate(X: np.ndarray):
    y = []
    c = []
    for xi in X:
        data = b.objective_function(xi)
        y.append(data["function_value"])
        c.append(data["cost"])

    return np.array(y)[:, None], np.log(np.array(c)[:, None])


def evaluate_subsets(X: np.ndarray):
    y = []
    c = []
    for xi in X:
        s = (np.exp(xi[-1]) - s_min) / (s_max - s_min)
        data = b.objective_function(xi[:-1], dataset_fraction=s)
        y.append(data["function_value"])
        c.append(data["cost"])

    return np.array(y)[:, None], np.log(np.array(c)[:, None])



cs = b.get_configuration_space()

list_params = []

for h in cs.get_hyperparameters():
    list_params.append(ContinuousParameter(h.name, h.lower, h.upper))

space = ParameterSpace(list_params)

if args.method == "gp_ei":

    obj = UserFunctionWithCostWrapper(evaluate)

    init_design = RandomDesign(space)
    X_init = init_design.get_samples(args.n_init)
    Y_init, C_init = evaluate(X_init)

    model = BOGP(X_init=X_init, Y_init=Y_init)

    acquisition = ExpectedImprovement(model)

    acquisition_optimizer = DirectOptimizer(space)

    candidate_point_calculator = Sequential(acquisition, acquisition_optimizer)

    bo = BayesianOptimizationLoop(model=model, space=space, X_init=X_init, Y_init=Y_init, acquisition=acquisition,
                                  candidate_point_calculator=candidate_point_calculator)
    initial_results = []
    for i in range(X_init.shape[0]):
        initial_results.append(UserFunctionResult(X_init[i], Y_init[i], C_init[i]))
    loop_state = LoopState(initial_results)
    bo.loop_state = loop_state

    bo.run_loop(user_function=obj,
                stopping_condition=FixedIterationsStoppingCondition(args.num_iterations - args.n_init))
    C = bo.loop_state.C
    Y = bo.loop_state.Y

elif args.method == "gp_ei_per_cost":

    obj = UserFunctionWithCostWrapper(evaluate)

    init_design = RandomDesign(space)
    X_init = init_design.get_samples(args.n_init)
    Y_init, C_init = evaluate(X_init)

    model = BOGP(X_init=X_init, Y_init=Y_init)
    cost_model = BOGP(X_init=X_init, Y_init=C_init)

    acquisition = ExpectedImprovementPerCost(model, cost_model)

    acquisition_optimizer = DirectOptimizer(space)

    candidate_point_calculator = Sequential(acquisition, acquisition_optimizer)

    bo = CostSensitiveBayesianOptimizationLoop(model_objective=model, model_cost=cost_model,
                                               space=space, X_init=X_init, Y_init=Y_init, C_init=C_init,
                                               acquisition=acquisition,
                                               candidate_point_calculator=candidate_point_calculator)
    bo.run_loop(user_function=obj,
                stopping_condition=FixedIterationsStoppingCondition(args.num_iterations - args.n_init))
    C = bo.loop_state.C
    Y = bo.loop_state.Y

elif args.method == "fabolas":

    obj = UserFunctionWithCostWrapper(evaluate_subsets)

    bo = Fabolas(obj, space, s_min=s_min, s_max=s_max, n_init=args.n_init)
    bo.run_optimization(num_iterations=args.num_iterations - args.n_init)
    # TODO: Return incumbent
    C = bo.loop_state.C
    Y = bo.loop_state.Y


elif args.method == "rs":

    init_design = RandomDesign(space)
    X = init_design.get_samples(args.num_iterations)
    Y, C = evaluate(X)

curr_inc = np.inf
curr_time = 0
error = []
runtime = []
for yi, ci in zip(Y, C):
    if curr_inc > yi:
        curr_inc = yi[0]
    error.append(curr_inc)

    curr_time += ci[0]
    runtime.append(curr_time)

data = dict()
data["error"] = error
data["runtime"] = runtime

print(error, runtime)
path = os.path.join(args.output_path, args.benchmark)
os.makedirs(path, exist_ok=True)

fname = os.path.join(path, "%s_run_%d.json" % (args.method, args.run_id))

fh = open(fname, "w")
json.dump(data, fh)
fh.close()
