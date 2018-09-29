import os
import time
import json
import argparse
import numpy as np

from hpolib.benchmarks.synthetic_functions import Branin, Hartmann3, Hartmann6, Bohachevsky, Rosenbrock, \
    GoldsteinPrice, Forrester, Camelback, SinOne, SinTwo

from GPy.models import GPRegression
from GPy.kern import Matern52

from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core.loop import FixedIterationsStoppingCondition, UserFunction, UserFunctionResult, Sequential
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.models.bohamiann import Bohamiann
from emukit.models.random_forest import RandomForest
from emukit.experimental_design.model_free.random_design import RandomDesign
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.optimization import AcquisitionOptimizer, DirectOptimizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", default="gp", type=str, nargs="?")
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

if args.model_type == "bnn":
    model = Bohamiann(X_init=X_init, Y_init=Y_init, verbose=True)

elif args.model_type == "rf":
    model = RandomForest(X_init=X_init, Y_init=Y_init)
    with_gradients = False


elif args.model_type == "gp":
    kernel = Matern52(len(list_params), variance=1., ARD=True)
    gpmodel = GPRegression(X_init, Y_init, kernel)
    gpmodel.optimize()
    model = GPyModelWrapper(gpmodel)

acquisition = ExpectedImprovement(model)
acquisition_optimizer = DirectOptimizer(space)

candidate_point_calculator = Sequential(acquisition, acquisition_optimizer)

bo = BayesianOptimizationLoop(model=model, space=space, X_init=X_init, Y_init=Y_init, acquisition=acquisition,
                              candidate_point_calculator=candidate_point_calculator)

overhead = []
st = time.time()
for i in range(args.num_iterations):
    t = time.time()
    bo.run_loop(user_function=obj, stopping_condition=FixedIterationsStoppingCondition(i + X_init.shape[0]))
    overhead.append(time.time() - t)
    print(i)

data = dict()
data["overhead"] = overhead
data["runtime"] = np.cumsum(overhead).tolist()

path = os.path.join(args.output_path, args.benchmark)
os.makedirs(path, exist_ok=True)

fname = os.path.join(path, "%s_run_%d.json" % (args.model_type, args.run_id))

fh = open(fname, "w")
json.dump(data, fh)
fh.close()
