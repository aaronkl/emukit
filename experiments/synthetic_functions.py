import os
import json
import argparse
import numpy as np

from hpolib.benchmarks.synthetic_functions import Branin, Hartmann3, Hartmann6, Bohachevsky, Rosenbrock, \
    GoldsteinPrice, Forrester, Camelback, SinOne, SinTwo

from emukit.models.bo_gp import BOGP
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core.loop import FixedIterationsStoppingCondition, UserFunction, UserFunctionResult, Sequential
from emukit.bayesian_optimization.acquisitions import NegativeLowerConfidenceBound, \
    ProbabilityOfImprovement, ExpectedImprovement, LogExpectedImprovement
from emukit.models.bohamiann import Bohamiann
from emukit.models.random_forest import RandomForest
from emukit.models.dngo import DNGO
from emukit.experimental_design.model_free.random_design import RandomDesign
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.optimization import AcquisitionOptimizer, DirectOptimizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", default="gp", type=str, nargs="?")

parser.add_argument("--acquisition_type", default="ei", type=str, nargs="?")
parser.add_argument("--benchmark", default="branin", type=str, nargs="?")
parser.add_argument("--num_iterations", default=10, type=int, nargs="?")
parser.add_argument("--output_path", default=".", type=str, nargs="?")
parser.add_argument("--run_id", default=0, type=int, nargs="?")


args = parser.parse_args()


class Wrapper(UserFunction):

    def __init__(self, b):
        self.b = b

    def evaluate(self, X: np.ndarray):
        res = []
        for xi in X:
            yi = self.b.objective_function(xi)["function_value"]
            res.append(UserFunctionResult(xi, np.array([yi])))
        return res


if args.benchmark == "branin":
    b = Branin()
elif args.benchmark == "hartmann3":
    b = Hartmann3()
elif args.benchmark == "hartmann6":
    b = Hartmann6()
elif args.benchmark == "bohachevsky":
    b = Bohachevsky()
elif args.benchmark == "rosenbrock":
    b = Rosenbrock()
elif args.benchmark == "goldsteinprice":
    b = GoldsteinPrice()
elif args.benchmark == "forrester":
    b = Forrester()
elif args.benchmark == "camelback":
    b = Camelback()
elif args.benchmark == "sinone":
    b = SinOne()
elif args.benchmark == "sintwo":
    b = SinTwo()

obj = Wrapper(b)

f_opt = b.get_meta_information()["f_opt"]

cs = b.get_configuration_space()

list_params = []

for h in cs.get_hyperparameters():
    list_params.append(ContinuousParameter(h.name, h.lower, h.upper))

space = ParameterSpace(list_params)

init_design = RandomDesign(space)
X_init = init_design.get_samples(2)
Y_init = np.array([b.objective_function(xi)["function_value"] for xi in X_init])[:, None]

with_gradients = True

if args.model_type == "bnn":
    model = Bohamiann(X_init=X_init, Y_init=Y_init, verbose=True)

elif args.model_type == "rf":
    model = RandomForest(X_init=X_init, Y_init=Y_init)
    with_gradients = False

elif args.model_type == "dngo":
    model = DNGO(X_init=X_init, Y_init=Y_init)
    with_gradients = False

elif args.model_type == "gp":
    model = BOGP(X_init=X_init, Y_init=Y_init)

if args.acquisition_type == "ei":
    acquisition = ExpectedImprovement(model)
elif args.acquisition_type == "pi":
    acquisition = ProbabilityOfImprovement(model)
elif args.acquisition_type == "nlcb":
    acquisition = NegativeLowerConfidenceBound(model)
elif args.acquisition_type == "logei":
    acquisition = LogExpectedImprovement(model)
    with_gradients = False

# if with_gradients:
#    acquisition_optimizer = AcquisitionOptimizer(space)
# else:
acquisition_optimizer = DirectOptimizer(space)

candidate_point_calculator = Sequential(acquisition, acquisition_optimizer)

bo = BayesianOptimizationLoop(model=model, space=space, X_init=X_init, Y_init=Y_init, acquisition=acquisition,
                              candidate_point_calculator=candidate_point_calculator)
bo.run_loop(user_function=obj, stopping_condition=FixedIterationsStoppingCondition(args.num_iterations))

curr_inc = np.inf
traj = []
regret = []
for yi in bo.loop_state.Y:
    if curr_inc > yi:
        curr_inc = yi[0]
    traj.append(curr_inc)
    regret.append(curr_inc - f_opt)

data = dict()
data["regret"] = regret

path = os.path.join(args.output_path, args.benchmark)
os.makedirs(path, exist_ok=True)

fname = os.path.join(path, "%s_%s_run_%d.json" % (args.model_type, args.acquisition_type, args.run_id))

fh = open(fname, "w")
json.dump(data, fh)
fh.close()
