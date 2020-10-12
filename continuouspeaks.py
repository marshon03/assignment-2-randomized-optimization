import six
import sys

sys.modules['sklearn.externals.six'] = six
import mlrose_hiive as mlrose
import numpy as np

import pandas as pd

import os
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

np.random.seed(1)

destination = 'Add destination path'


def run_continuouspeaks():

    if not os.path.exists(destination + '/ContinuousPeaks/'):
        os.mkdir(destination + '/ContinuousPeaks/')

    logger = logging.getLogger(__name__)

    problem_size_space = np.linspace(10, 125, 20, dtype=int)

    best_fit_dict = {}
    best_fit_dict['Problem Size'] = problem_size_space
    best_fit_dict['Random Hill Climbing'] = []
    best_fit_dict['Simulated Annealing'] = []
    best_fit_dict['Genetic Algorithm'] = []
    best_fit_dict['MIMIC'] = []

    times = {}
    times['Problem Size'] = problem_size_space
    times['Random Hill Climbing'] = []
    times['Simulated Annealing'] = []
    times['Genetic Algorithm'] = []
    times['MIMIC'] = []

    fits_per_iteration = {}
    fits_per_iteration['Random Hill Climbing'] = []
    fits_per_iteration['Simulated Annealing'] = []
    fits_per_iteration['Genetic Algorithm'] = []
    fits_per_iteration['MIMIC'] = []

    for prob_size in problem_size_space:
        logger.info("---- Problem size: " + str(prob_size) + " ----")
        prob_size_int = int(prob_size)
        fitness_continuous_peaks = mlrose.ContinuousPeaks()
        generate_continuous_peaks_state = lambda: np.random.randint(2, size=prob_size_int)
        init_state = generate_continuous_peaks_state()
        problem = mlrose.DiscreteOpt(length=prob_size_int, fitness_fn=fitness_continuous_peaks, maximize=True, max_val=2)

        start = datetime.now()
        _, best_fitness_sa, fit_array_sa = mlrose.simulated_annealing(problem,
                                                                      schedule=mlrose.ExpDecay(exp_const=.001,
                                                                                               init_temp=2),
                                                                      max_attempts=20,
                                                                      curve=True,
                                                                      max_iters=1000, init_state=init_state)
        best_fit_dict['Simulated Annealing'].append(best_fitness_sa)
        end = datetime.now()
        times['Simulated Annealing'].append((end - start).total_seconds())

        start = datetime.now()
        _, best_fitness_rhc, fit_array_rhc = mlrose.random_hill_climb(problem, max_attempts=200, max_iters=1000,
                                                                      curve=True, restarts=20)
        best_fit_dict['Random Hill Climbing'].append(best_fitness_rhc)
        end = datetime.now()
        times['Random Hill Climbing'].append((end - start).total_seconds())

        start = datetime.now()
        _, best_fitness_ga, fit_array_ga = mlrose.genetic_alg(problem, pop_size=prob_size_int * 5, curve=True,
                                                              mutation_prob=.025, max_attempts=20, max_iters=1000)
        best_fit_dict['Genetic Algorithm'].append(best_fitness_ga)
        end = datetime.now()
        times['Genetic Algorithm'].append((end - start).total_seconds())

        start = datetime.now()
        _, best_fitness_mimic, fit_array_mimic = mlrose.mimic(problem, pop_size=prob_size_int * 3, curve=True,
                                                              keep_pct=.25, max_attempts=20, max_iters=1000)
        best_fit_dict['MIMIC'].append(best_fitness_mimic)
        end = datetime.now()
        times['MIMIC'].append((end - start).total_seconds())

    fits_per_iteration['Random Hill Climbing'] = fit_array_rhc
    fits_per_iteration['Simulated Annealing'] = fit_array_sa
    fits_per_iteration['Genetic Algorithm'] = fit_array_ga
    fits_per_iteration['MIMIC'] = fit_array_mimic

    fit_frame = pd.DataFrame.from_dict(best_fit_dict, orient='index').transpose()
    time_frame = pd.DataFrame.from_dict(times, orient='index').transpose()
    fit_iteration_frame = pd.DataFrame.from_dict(fits_per_iteration, orient='index').transpose()

    fit_frame.to_csv(destination + '/ContinuousPeaks/problem_size_fit.csv')
    time_frame.to_csv(destination + '/ContinuousPeaks/problem_size_time.csv')
    fit_iteration_frame.to_csv(destination + '/ContinuousPeaks/fit_per_iteration.csv')


if __name__ == "__main__":
    run_continuouspeaks()