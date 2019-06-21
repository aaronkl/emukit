import argparse
import json
import os
import ConfigSpace
import numpy as np

from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario

import numpy as np
from emukit.examples.profet.meta_benchmarks import meta_svm, meta_forrester, meta_fcnet, meta_xgboost
from util import estimate_incumbent

parser = argparse.ArgumentParser()

parser.add_argument('--run_id', default=0, type=int, nargs='?')
parser.add_argument('--output_path', default="./", type=str, nargs='?')
parser.add_argument('--sample_path', default="./", type=str, nargs='?')
parser.add_argument('--n_iters', default=50, type=int, nargs='?')
parser.add_argument('--n_init', default=2, type=int, nargs='?')
parser.add_argument('--benchmark', default="meta_svm", type=str, nargs='?')
parser.add_argument('--instance_id', default=0, type=int, nargs='?')
parser.add_argument('--noise', action='store_true')
args = parser.parse_args()

if args.benchmark == "meta_forrester":

    fname_objective = os.path.join(args.sample_path, "sample_objective_%d.pkl" % args.instance_id)
    fcn, parameter_space = meta_forrester(fname_objective=fname_objective)

elif args.benchmark == "meta_svm":

    fname_objective = os.path.join(args.sample_path, "sample_objective_%d.pkl" % args.instance_id)
    fname_cost = os.path.join(args.sample_path, "sample_cost_%d.pkl" % args.instance_id)
    fcn, parameter_space = meta_svm(fname_objective=fname_objective,
                                    fname_cost=fname_cost,
                                    noise=args.noise)
elif args.benchmark == "meta_fcnet":

    fname_objective = os.path.join(args.sample_path, "sample_objective_%d.pkl" % args.instance_id)
    fname_cost = os.path.join(args.sample_path, "sample_cost_%d.pkl" % args.instance_id)
    fcn, parameter_space = meta_fcnet(fname_objective=fname_objective,
                                      fname_cost=fname_cost,
                                      noise=args.noise)

elif args.benchmark == "meta_xgboost":

    fname_objective = os.path.join(args.sample_path, "sample_objective_%d.pkl" % args.instance_id)
    fname_cost = os.path.join(args.sample_path, "sample_cost_%d.pkl" % args.instance_id)
    fcn, parameter_space = meta_xgboost(fname_objective=fname_objective,
                                        fname_cost=fname_cost,
                                        noise=args.noise)

cs = ConfigSpace.ConfigurationSpace()
for p in parameter_space.parameters:
    cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(p.name, lower=p.min, upper=p.max))

scenario = Scenario({"run_obj": "quality",
                     "runcount-limit": args.n_iters,
                     "cs": cs,
                     "deterministic": "false",
                     "initial_incumbent": "RANDOM",
                     "output_dir": ""})


def objective_function(config, **kwargs):
    x = []
    for p in parameter_space.parameters:
        x.append(config[p.name])
    if args.benchmark == "meta_forrester":
        y = fcn(np.array([x]))
        return float(y[0, 0]), 1
    else:
        y, c = fcn(np.array([x]))
        return float(y[0, 0]), float(c[0, 0])


smac = SMAC(scenario=scenario, tae_runner=objective_function)
smac.optimize()
smac.solver.intensifier.maxR = 4
X, y, _ = smac.get_X_y()

costs = []
for d in smac.runhistory.data:
    costs.append(smac.runhistory.data[d].additional_info)
data = dict()
data["method"] = "smac"
data["benchmark"] = args.benchmark
data["trajectory"] = estimate_incumbent(y)
data["run_id"] = args.run_id
data["runtime"] = np.cumsum(np.array(costs)).tolist()
if bool(args.noise):
    output_path = os.path.join(args.output_path, args.benchmark + "_noise", "smac", "instance_%d" % args.instance_id)
else:
    output_path = os.path.join(args.output_path, args.benchmark + "_noiseless", "smac", "instance_%d" % args.instance_id)
os.makedirs(output_path, exist_ok=True)

with open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w') as fh:
    json.dump(data, fh)
