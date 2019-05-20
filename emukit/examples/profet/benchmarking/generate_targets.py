import os
import argparse
import json
import numpy as np
import sobol_seq

from emukit.examples.profet.meta_benchmarks import meta_svm, meta_forrester

from emukit.examples.gp_bayesian_optimization.enums import ModelType, AcquisitionType
from emukit.examples.gp_bayesian_optimization.optimization_loops import create_bayesian_optimization_loop
from emukit.examples.gp_bayesian_optimization.single_objective_bayesian_optimization import GPBayesianOptimization
from emukit.benchmarking.loop_benchmarking.benchmarker import Benchmarker
from emukit.benchmarking.loop_benchmarking.metrics import MinimumObservedValueMetric, TimeMetric, CumulativeCostMetric

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?')
parser.add_argument('--output_path', default="./", type=str, nargs='?')
parser.add_argument('--sample_path', default="./", type=str, nargs='?')
parser.add_argument('--benchmark', default="meta_svm", type=str, nargs='?')

args = parser.parse_args()

targets = []

for instance_id in range(1000):

    if args.benchmark == "meta_forrester":

        fname_objective = os.path.join(args.sample_path, "sample_objective_%d.pkl" % instance_id)
        fcn, parameter_space = meta_forrester(fname_objective=fname_objective)

    elif args.benchmark == "meta_svm":

        fname_objective = os.path.join(args.sample_path, "sample_objective_%d.pkl" % instance_id)
        fname_cost = os.path.join(args.sample_path, "sample_cost_%d.pkl" % instance_id)
        fcn, parameter_space = meta_svm(fname_objective=fname_objective,
                                    fname_cost=fname_cost,
                                    noise=False)

    D = len(parameter_space.parameters)
    X = sobol_seq.i4_sobol_generate(D, D * 100)
    bounds = np.array(parameter_space.get_bounds())

    lower = bounds[:, 0]
    upper = bounds[:, 1]
    X = X * (upper - lower) + lower
    y = []
    for xi in X:
        if args.benchmark == "meta_forrester":
            y.append(float(fcn(xi[None, :])[0, 0]))
        else:
            y.append(float(fcn(xi[None, :])[0][0, 0]))

    targets.append(y)

print(np.array(targets).shape)

os.makedirs(args.output_path, exist_ok=True)

fh = open(os.path.join(args.output_path, args.benchmark + "_targets.json"), "w")
json.dump(targets, fh)
