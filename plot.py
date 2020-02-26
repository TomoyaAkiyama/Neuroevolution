import os

import numpy as np
import matplotlib.pyplot as plt


def plot_test_scores(activation, layernorm, env_name, seed, x_min, x_max, y_min, y_max):
    parent_dir = os.path.abspath(os.path.dirname(__file__))
    save_dir = os.path.join(
        parent_dir,
        'results',
        env_name,
        activation + ('_LayerNorm' if layernorm else ''),
        'seed{}'.format(seed)
    )
    file_name = 'test_scores.txt'
    file_path = os.path.join(save_dir, file_name)
    test_data = np.loadtxt(file_path)
    time_steps = test_data[:, 0]
    test_scores = test_data[:, 1]

    plt.figure()
    plt.plot(time_steps, test_scores)
    plt.xlabel('Time Step')
    plt.ylabel('Average Episode Rewards')
    plt.grid()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    fig_name = 'test_scores.png'
    fig_path = os.path.join(save_dir, fig_name)
    plt.savefig(fig_path)


def plot_average_test_scores(activation, layernorm, env_name, seeds, x_min, x_max, y_min, y_max):
    parent_dir = os.path.abspath(os.path.dirname(__file__))
    all_test_scores = []
    time_steps = []
    for seed in seeds:
        save_dir = os.path.join(
            parent_dir,
            'results',
            env_name,
            activation + ('_LayerNorm' if layernorm else ''),
            'seed{}'.format(seed)
        )
        file_name = 'test_scores.txt'
        file_path = os.path.join(save_dir, file_name)
        test_data = np.loadtxt(file_path)
        time_steps = test_data[:, 0]
        all_test_scores.append(test_data[:, 1])

    average_test_scores = np.mean(all_test_scores, axis=0)
    std = np.std(all_test_scores, axis=0)

    plt.figure()
    plt.plot(time_steps, average_test_scores)
    plt.fill_between(time_steps, average_test_scores - std, average_test_scores + std, facecolor='c', alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('Average Episode Rewards')
    plt.grid()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    fig_name = 'average_test_scores.png'
    save_dir = os.path.join(parent_dir, 'results', env_name, activation + ('_LayerNorm' if layernorm else ''))
    fig_path = os.path.join(save_dir, fig_name)
    plt.savefig(fig_path)


if __name__ == '__main__':
    env_name = 'Swimmer-v2'
    activation = 'ELU'
    layernorm = True
    seeds = range(1, 11)

    x_min = 0
    x_max = 1000000
    y_min = -20
    y_max = 300

    for seed in seeds:
        plot_test_scores(activation, layernorm, env_name, seed, x_min, x_max, y_min, y_max)
    plot_average_test_scores(activation, layernorm, env_name, seeds, x_min, x_max, y_min, y_max)
