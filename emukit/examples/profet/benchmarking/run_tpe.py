import argparse
import json
import os

import numpy as np
from emukit.examples.profet.meta_benchmarks import meta_svm, meta_forrester
from hyperopt import fmin, tpe, hp, STATUS_OK
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

all_y = []
costs = []


def wrapper(config):
    x = []
    for p in parameter_space.parameters:
        x.append(config[p.name])

    if args.benchmark == "meta_forrester":
        y = fcn(np.array([x]))
    else:
        y, c = fcn(np.array([x]))
        costs.append(c)
    all_y.append(float(y[0, 0]))
    return {
        'config': config,
        'loss': y,
        'status': STATUS_OK}


space = {}
for p in parameter_space.parameters:
    space[p.name] = hp.uniform(p.name, p.min, p.max)

best = fmin(wrapper,
            space=space,
            algo=tpe.suggest,
            max_evals=args.n_iters,
            trials=None)

data = dict()
data["method"] = "tpe"
data["benchmark"] = args.benchmark
data["trajectory"] = estimate_incumbent(all_y)
data["run_id"] = args.run_id
data["runtime"] = np.cumsum(costs).tolist()
if bool(args.noise):
    output_path = os.path.join(args.output_path, args.benchmark + "_noise", "tpe", "instance_%d" % args.instance_id)
else:
    output_path = os.path.join(args.output_path, args.benchmark + "_noiseless", "tpe", "instance_%d" % args.instance_id)
os.makedirs(output_path, exist_ok=True)

with open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w') as fh:
    json.dump(data, fh)
