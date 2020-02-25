import os
import time

import torch
import torch.backends.cudnn
import numpy as np

from learner import EALearner
from logger import Logger

np.set_printoptions(precision=3)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def main(env_name, seed, args):
    parent_dir = os.path.abspath(os.path.dirname(__file__))
    save_dir = os.path.join(
        parent_dir,
        'results',
        env_name,
        'seed{}'.format(seed)
    )
    logger = Logger(save_dir)

    set_seed(seed)

    args['env_name'] = env_name
    agent = EALearner(**args)
    print('Running SSNE for', env_name)

    try:
        start_time = time.time()
        gen = 0
        while True:
            gen += 1
            pop_fitness, test_mean, test_std, champ_wwid = agent.train(gen, logger)

            message = 'Test Score of the Champ: {:.3f} ({:.3f})'.format(test_mean, test_std)
            print('=' * len(message))
            print(message)
            print('=' * len(message))
            print()

            print('Gen {}, Frames {},'.format(gen, agent.total_frames),
                  'Frames/sec: {:.2f}'.format(agent.total_frames / (time.time() - start_time)))
            print('\tPopulation Fitness', pop_fitness)
            print()
            print('\tBest Fitness ever: {:.3f}'.format(agent.best_score))
            print('\tBest Policy ever genealogy:', agent.genealogy.tree[int(agent.best_policy.wwid.item())].history)
            print('\tChamp Fitness: {:.3f}'.format(pop_fitness.max()))
            print('\tChamp genealogy:', agent.genealogy.tree[champ_wwid].history)
            print()

            if agent.total_frames >= 1000000:
                break
        logger.save()
        for i, individual in enumerate(agent.population):
            model_dir = os.path.join(save_dir, 'learned_model', 'Population')
            os.makedirs(model_dir, exist_ok=True)
            torch.save(individual.state_dict(), os.path.join(model_dir, 'individual{}.pth'.format(i)))

        print('Elapsed time: {}'.format(time.time() - start_time))
    finally:
        # Kill all processes
        for worker in agent.test_workers:
            worker.terminate()
        for worker in agent.evo_workers:
            worker.terminate()


if __name__ == '__main__':
    args = {
        'hidden_sizes': [400, 300],
        'pop_size': 10,
        'elite_fraction': 0.1,
        'cross_prob': 0.0,
        'cross_fraction': 0.3,
        'bias_cross_prob': 0.2,
        'mutation_prob': 0.9,
        'mut_strength': 0.1,
        'mut_fraction': 0.1,
        'super_mut_prob': 0.05,
        'reset_prob': 0.1,
    }

    env_name = 'Swimmer-v2'
    seed = 1
    main(env_name, seed, args)
