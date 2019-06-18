import argparse
import json
import os

import ConfigSpace
import cma
import numpy as np

from emukit.examples.profet.meta_benchmarks import meta_svm, meta_forrester, meta_fcnet

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


bounds = parameter_space.get_bounds()
lower = np.array(bounds)[:, 0]
upper = np.array(bounds)[:, 1]

print(lower, upper)
all_y = []
costs = []


def wrapper(x):
    if args.benchmark == "meta_forrester":
        y = fcn(np.array([x]))
        c = 1
    else:
        y, c = fcn(np.array([x]))
        c = c[0, 0]
        costs.append(c)
    all_y.append(float(y[0, 0]))

    return float(y[0, 0]), float(c)


x_0 = np.random.rand(lower.shape[0])
es = cma.CMAEvolutionStrategy(x_0, 0.6, {'bounds': [lower, upper], "maxfevals": args.n_iters})
es.optimize(wrapper, args.n_iters)

data = dict()
data["method"] = "cmaes"
data["benchmark"] = args.benchmark
data["trajectory"] = estimate_incumbent(all_y)[:args.n_iters]
data["run_id"] = args.run_id
data["runtime"] = np.cumsum(costs).tolist()[:args.n_iters]

if bool(args.noise):
    output_path = os.path.join(args.output_path, args.benchmark + "_noise", "cmaes", "instance_%d" % args.instance_id)
else:
    output_path = os.path.join(args.output_path, args.benchmark + "_noiseless", "cmaes", "instance_%d" % args.instance_id)
os.makedirs(output_path, exist_ok=True)

with open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w') as fh:
    json.dump(data, fh)
