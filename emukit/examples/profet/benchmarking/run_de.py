import argparse
import json
import os

import numpy as np
from emukit.examples.profet.meta_benchmarks import meta_svm, meta_forrester, meta_fcnet

parser = argparse.ArgumentParser()

parser.add_argument('--run_id', default=0, type=int, nargs='?')
parser.add_argument('--output_path', default="./", type=str, nargs='?')
parser.add_argument('--sample_path', default="./", type=str, nargs='?')
parser.add_argument('--n_iters', default=50, type=int, nargs='?')
parser.add_argument('--n_init', default=2, type=int, nargs='?')
parser.add_argument('--benchmark', default="meta_svm", type=str, nargs='?')
parser.add_argument('--instance_id', default=0, type=int, nargs='?')
parser.add_argument('--noise', action='store_true')
parser.add_argument('--strategy', default="rand1", type=str, nargs='?')
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
parameter_space.get_bounds()
lower = np.array(parameter_space.get_bounds())[:, 0]
upper = np.array(parameter_space.get_bounds())[:, 1]
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


class DifferentialevolutionOptimizer:
    def __init__(self, f, lower, upper, fevals, strategy, bin=1):

        self.f = f

        self.lower_bound = lower
        self.upper_bound = upper

        # Define F and CR values
        self.mut = 0.5
        self.crossp = 0.5

        # DEfine popsize and FEs(its*popsize)
        self.popsize = 10
        self.its = fevals // self.popsize - 1
        self.dimensions = len(self.lower_bound)

        self.de_pop = []
        self.fitness = []
        self.fbest = np.float('inf')  # if it is a minimization problem, otherwise use 0 or - inf for maximization
        self.idxbest = 1
        self.strategy = strategy
        self.bin = bin

    def evolve(self, j):
        best_idv = self.de_pop[self.idxbest]
        current_idv = self.de_pop[j]

        # perform mutation operation
        if self.strategy == "rand1":
            idxs = [idx for idx in range(self.popsize) if idx != j]
            r1, r2, r3 = self.de_pop[np.random.choice(idxs, 3, replace=False)]

            # Step 3.1: Perform mutation and checking
            temp = r1 + self.mut * (r2 - r3)
            vi = np.clip(temp, self.lower_bound, self.upper_bound)

        if self.strategy == "best1":
            idxs = [idx for idx in range(self.popsize) if idx != j]
            r1, r2 = self.de_pop[np.random.choice(idxs, 2, replace=False)]
            temp = best_idv + self.mut * (r1 - r2)
            vi = np.clip(temp, self.lower_bound, self.upper_bound)

        if self.strategy == "rand2":
            idxs = [idx for idx in range(self.popsize) if idx != j]
            r1, r2, r3, r4, r5 = self.de_pop[np.random.choice(idxs, 5, replace=False)]
            temp = r1 + self.mut * (r1 - r2) + self.mut * (r3 - r4)
            vi = np.clip(temp, self.lower_bound, self.upper_bound)

        if self.strategy == "best2":
            idxs = [idx for idx in range(self.popsize) if idx != j]
            r1, r2, r3, r4 = self.de_pop[np.random.choice(idxs, 4, replace=False)]
            temp = best_idv + self.mut * (r1 - r2) + self.mut * (r3 - r4)
            vi = np.clip(temp, self.lower_bound, self.upper_bound)

        if self.strategy == "currenttobest1":
            idxs = [idx for idx in range(self.popsize) if idx != j]
            r1, r2 = self.de_pop[np.random.choice(idxs, 2, replace=False)]
            temp = current_idv + self.mut * (best_idv - current_idv) + self.mut * (r1 - r2)
            vi = np.clip(temp, self.lower_bound, self.upper_bound)

        if self.strategy == "randtobest1":
            idxs = [idx for idx in range(self.popsize) if idx != j]
            r1, r2, r3 = self.de_pop[np.random.choice(idxs, 3, replace=False)]
            temp = r1 + self.mut * (best_idv - r1) + self.mut * (r2 - r3)
            vi = np.clip(temp, self.lower_bound, self.upper_bound)

        # perform crossover operation
        if self.bin == 1:
            cross_points = np.random.rand(self.dimensions) < self.crossp

            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dimensions)] = True
            ui = np.where(cross_points, vi, current_idv)

        else:
            i = 0
            ui = []
            fill_point = np.random.randint(0, self.dimensions)
            while (i < self.dimensions and
                   np.random.rand(0, 1) < self.crossp):
                ui[fill_point] = vi[fill_point]
                fill_point = (fill_point + 1) % self.dimensions
                i += 1

        return ui

    def run(self):
        # len_traj = 0
        # start the run
        curr_time = 0
        traj = []
        runtime = []

        # Step1: initialization
        rand_temp = np.random.rand(self.popsize, self.dimensions)
        diff = np.fabs(self.lower_bound - self.upper_bound)
        self.de_pop = self.lower_bound + rand_temp * diff

        c = []

        # Step 2: population evaluation

        for j in range(self.popsize):
            ftemp, ctemp = self.f(self.de_pop[j])
            self.fitness.append(ftemp)

            if ftemp < self.fbest:
                self.fbest = ftemp
                self.idxbest = j
            curr_time += ctemp
            runtime.append(curr_time)
            traj.append(self.fbest)

        # Step 3: Start evolutionary search
        for i in range(self.its):
            for j in range(self.popsize):
                ui = self.evolve(j)

                fit, c = self.f(ui)

                # Step3.5: Perform Selection
                if fit < self.fitness[j]:
                    self.fitness[j] = fit
                    self.de_pop[j] = ui
                    if fit < self.fitness[self.idxbest]:
                        self.idxbest = j
                        best = ui

                curr_time += c
                runtime.append(curr_time)
                traj.append(self.fitness[self.idxbest])
            # print(fitness[best_idx],curr_time)

        return traj, runtime


de = DifferentialevolutionOptimizer(wrapper, lower, upper, strategy=args.strategy, fevals=args.n_iters)
traj, runtime = de.run()

data = dict()
data["method"] = "differential_evolution"
data["benchmark"] = args.benchmark
data["trajectory"] = traj
data["run_id"] = args.run_id
data["runtime"] = runtime

if bool(args.noise):
    output_path = os.path.join(args.output_path, args.benchmark + "_noise", "de_%s" % args.strategy, "instance_%d" % args.instance_id)
else:
    output_path = os.path.join(args.output_path, args.benchmark + "_noiseless", "de_%s" % args.strategy, "instance_%d" % args.instance_id)
os.makedirs(output_path, exist_ok=True)

with open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w') as fh:
    json.dump(data, fh)
