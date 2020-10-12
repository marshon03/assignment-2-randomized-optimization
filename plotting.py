import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import os

sns.set_style("darkgrid")

destination = 'Add destination path'


def plot_problem_size_scores(problem_dataframe, problem_name, output_dir=destination):
    # We have a line for each of the algorithms. Use seaborn to draw them super easily.
    if 'Unnamed: 0' in problem_dataframe.keys():
        problem_dataframe.pop('Unnamed: 0')

    problem_dataframe = problem_dataframe.set_index('Problem Size')

    axis = sns.lineplot(data=problem_dataframe)
    axis.set_ylabel('Fit')
    axis.legend(loc='lower right')
    axis.set_xlabel('Problem size')
    axis.set_title('Fit score vs problem size: ' + problem_name)
    axis.figure.savefig(os.path.join(output_dir, problem_name, 'fit_vs_size.png'), dpi=300)
    plt.close()


def plot_problem_size_time(problem_dataframe, problem_name, output_dir=destination):
    if 'Unnamed: 0' in problem_dataframe.keys():
        problem_dataframe.pop('Unnamed: 0')

    problem_dataframe = problem_dataframe.set_index('Problem Size')

    axis = sns.lineplot(data=problem_dataframe)
    axis.legend(loc='lower right')
    axis.set_yscale('log')
    axis.set_ylabel('Time (log seconds)')
    axis.set_xlabel('Problem size')
    axis.set_title('Problem size vs optimization time: ' + problem_name)
    axis.figure.savefig(os.path.join(output_dir, problem_name, 'time_vs_size.png'), dpi=300)
    plt.close()


def plot_iteration_fit(problem_dataframe, problem_name, output_dir=destination):
    if 'Unnamed: 0' in problem_dataframe.keys():
        problem_dataframe.pop('Unnamed: 0')

    axis = sns.lineplot(data=problem_dataframe)
    axis.set_xscale('log')
    axis.legend(loc='lower right')
    axis.set_ylabel('Fit')
    axis.set_xlim(1, 400)
    axis.set_xlabel('Iteration')
    axis.set_title('Fit vs Iteration For All Optimizers: ' + problem_name)
    path = os.path.join(output_dir, problem_name, 'fit_vs_iter.png')
    axis.figure.savefig(path, dpi=300)
    plt.close()


def plot_neural_net_analysis(dataframe, dataset_name, output_dir=destination):
    axis = sns.lineplot(data=dataframe)
    axis.legend(loc='lower right')
    axis.set_xscale('log')
    axis.set_ylabel('Accuracy')
    axis.set_xlabel('Maximum Allowed iterations')
    axis.set_title('Accuracy vs Number of Allowed Iterations:' + dataset_name)
    path = os.path.join(output_dir, "images", 'accuracy_vs_iter.png')
    axis.figure.savefig(path, dpi=300)
    plt.close()