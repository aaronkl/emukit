import os
import argparse
import json
import numpy as np

from emukit.examples.profet.meta_benchmarks import meta_svm
from robo.fmin import bayesian_optimization, entropy_search, random_search


parser = argparse.ArgumentParser()

parser.add_argument('--run_id', default=0, type=int, nargs='?')
parser.add_argument('--output_path', default="./", type=str, nargs='?')
parser.add_argument('--n_iters', default=50, type=int, nargs='?')
parser.add_argument('--n_init', default=2, type=int, nargs='?')
parser.add_argument('--benchmark', default="meta_svm", type=str, nargs='?')
parser.add_argument('--method', default="random_search", type=str, nargs='?')
parser.add_argument('--instance_id', default=0, type=int, nargs='?')
parser.add_argument('--sample_path', default="./", type=str, nargs='?')
parser.add_argument('--noise', action='store_true')
args = parser.parse_args()

if args.benchmark == "meta_svm":

    fname_objective = os.path.join(args.sample_path, "sample_objective_%d.pkl" % args.instance_id)
    fname_cost = os.path.join(args.sample_path, "sample_cost_%d.pkl" % args.instance_id)
    fcn, parameter_space = meta_svm(fname_objective=fname_objective,
                                    fname_cost=fname_cost,
                                    noise=args.noise)

costs = []


def estimate_incumbent(y):
    curr_inc = np.inf
    traj = []

    for yi in y:
        if yi < curr_inc:
            curr_inc = yi
        traj.append(curr_inc)
    return traj


def wrapper(x):
    y, c = fcn(x[None, :])
    costs.append(c[0, 0])
    return float(y[0, 0])


lower = np.array(parameter_space.get_bounds())[:, 0]
upper = np.array(parameter_space.get_bounds())[:, 1]

if args.method == "entropy_search":
    results = entropy_search(wrapper, lower, upper,
                             num_iterations=args.n_iters, n_init=args.n_init)
elif args.method == "gp_mcmc":
    results = bayesian_optimization(wrapper, lower, upper,
                                    num_iterations=args.n_iters,
                                    n_init=args.n_init, model_type="gp_mcmc")
elif args.method == "gp":
    results = bayesian_optimization(wrapper, lower, upper,
                                    num_iterations=args.n_iters,
                                    n_init=args.n_init, model_type="gp")
elif args.method == "rf":
    results = bayesian_optimization(wrapper, lower, upper,
                                    num_iterations=args.n_iters,
                                    n_init=args.n_init, model_type="rf")
elif args.method == "random_search":
    results = random_search(wrapper, lower, upper,
                            num_iterations=args.n_iters)
elif args.method == "bohamiann":
    results = bayesian_optimization(wrapper, lower, upper,
                                    num_iterations=args.n_iters,
                                    n_init=args.n_init, model_type="bohamiann")


data = dict()
data["method"] = args.method
data["benchmark"] = args.benchmark
data["trajectory"] = estimate_incumbent(results["y"])
data["runtime"] = np.cumsum(costs).tolist()
data["actual_runtime"] = results["runtime"]
data["run_id"] = args.run_id

if bool(args.noise):
    output_path = os.path.join(args.output_path, args.benchmark + "_noise", args.method, "instance_%d" % args.instance_id)
else:
    output_path = os.path.join(args.output_path, args.benchmark + "_noiseless", args.method, "instance_%d" % args.instance_id)
os.makedirs(output_path, exist_ok=True)

with open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w') as fh:
    json.dump(data, fh)