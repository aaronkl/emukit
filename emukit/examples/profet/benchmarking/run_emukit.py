import argparse
import os
import pickle

from emukit.benchmarking.loop_benchmarking.benchmarker import Benchmarker
from emukit.benchmarking.loop_benchmarking.metrics import MinimumObservedValueMetric, TimeMetric, CumulativeCostMetric
from emukit.examples.gp_bayesian_optimization.enums import ModelType, AcquisitionType
from emukit.examples.gp_bayesian_optimization.optimization_loops import create_bayesian_optimization_loop
from emukit.examples.gp_bayesian_optimization.random_search import RandomSearch
from emukit.examples.gp_bayesian_optimization.single_objective_bayesian_optimization import GPBayesianOptimization
from emukit.examples.profet.meta_benchmarks import meta_svm, meta_forrester, meta_fcnet

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?')
parser.add_argument('--output_path', default="./", type=str, nargs='?')
parser.add_argument('--sample_path', default="./", type=str, nargs='?')
parser.add_argument('--n_iters', default=50, type=int, nargs='?')
parser.add_argument('--n_repeats', default=20, type=int, nargs='?')
parser.add_argument('--n_init', default=2, type=int, nargs='?')
parser.add_argument('--benchmark', default="meta_svm", type=str, nargs='?')
parser.add_argument('--model_type', default="gp", type=str, nargs='?')
parser.add_argument('--acquisition_type', default="ei", type=str, nargs='?')
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

if args.model_type == "rs":
    name = args.model_type
else:
    name = args.model_type + "_" + args.acquisition_type

if args.acquisition_type == "ei":
    acquisition = AcquisitionType.EI
elif args.acquisition_type == "pi":
    acquisition = AcquisitionType.PI
elif args.acquisition_type == "nlcb":
    acquisition = AcquisitionType.NLCB

if args.model_type == "rf":
    loops = [(name, lambda s: create_bayesian_optimization_loop(s.X, s.Y, parameter_space=parameter_space,
                                                                cost_init=s.cost,
                                                                acquisition_type=acquisition,
                                                                model_type=ModelType.RandomForest))]
elif args.model_type == "bnn":
    loops = [(name, lambda s: create_bayesian_optimization_loop(s.X, s.Y, parameter_space=parameter_space,
                                                                cost_init=s.cost,
                                                                acquisition_type=acquisition,
                                                                model_type=ModelType.BayesianNeuralNetwork))]
elif args.model_type == "gp":
    loops = [(name, lambda s: GPBayesianOptimization(parameter_space.parameters, s.X, s.Y,
                                                     acquisition_type=acquisition, noiseless=False))]

elif args.model_type == "rs":
    loops = [(name, lambda s: RandomSearch(parameter_space, x_init=s.X, y_init=s.Y, cost_init=s.cost))]

metrics = [MinimumObservedValueMetric(), TimeMetric(), CumulativeCostMetric()]

benchmarkers = Benchmarker(loops, fcn, parameter_space, metrics=metrics)
benchmark_results = benchmarkers.run_benchmark(n_iterations=args.n_iters,
                                               n_initial_data=args.n_init,
                                               n_repeats=args.n_repeats)

if bool(args.noise):
    output_path = os.path.join(args.output_path, args.benchmark + "_noise", name, "instance_%d" % args.instance_id)
else:
    output_path = os.path.join(args.output_path, args.benchmark + "_noiseless", name, "instance_%d" % args.instance_id)
os.makedirs(output_path, exist_ok=True)

fh = open(os.path.join(output_path, "run_%d.pkl" % args.run_id), "wb")
pickle.dump(benchmark_results, fh)
