from copy import deepcopy

import numpy as np
from torch.multiprocessing import Manager, Pipe, Process
import gym

from ssne import SSNE
from ssne2 import SSNE2
from genealogy import Genealogy
from models.deterministic_policy import DeterministicPolicy
from rollout_worker import rollout_worker


def env_parse(env_name):
    dummy_env = gym.make(env_name)
    state_dim = sum(list(dummy_env.observation_space.shape))
    action_dim = sum(list(dummy_env.action_space.shape))
    return state_dim, action_dim


class EALearner:
    def __init__(
            self,
            env_name,
            hidden_sizes=None,
            pop_size=10,
            elite_fraction=0.2,
            cross_prob=0.01,
            cross_fraction=0.3,
            bias_cross_prob=0.2,
            mutation_prob=0.2,
            mut_strength=0.02,
            mut_fraction=0.03,
            super_mut_prob=0.1,
            reset_prob=0.2,
    ):
        if hidden_sizes is None:
            hidden_sizes = [400, 300]

        ea_kwargs = {
            'elite_fraction': elite_fraction,
            'cross_prob': cross_prob,
            'cross_fraction': cross_fraction,
            'bias_cross_prob': bias_cross_prob,
            'mutation_prob': mutation_prob,
            'mut_strength': mut_strength,
            'mut_fraction': mut_fraction,
            'super_mut_prob': super_mut_prob,
            'reset_prob': reset_prob,
        }
        self.EA = SSNE(**ea_kwargs)
        # self.EA = SSNE2(None)

        self.manager = Manager()
        self.genealogy = Genealogy()

        self.pop_size = pop_size

        state_dim, action_dim = env_parse(env_name)

        # policies for EA's rollout
        self.population = self.manager.list()
        for i in range(pop_size):
            wwid = self.genealogy.new_id('EA_{}'.format(i))
            policy = DeterministicPolicy(state_dim, action_dim, hidden_sizes, wwid).eval()
            self.population.append(policy)
        self.best_policy = DeterministicPolicy(state_dim, action_dim, hidden_sizes, -1).eval()

        # Evolutionary population Rollout workers
        self.evo_task_pipes = []
        self.evo_result_pipes = []
        self.evo_workers = []
        for index in range(pop_size):
            task_pipe = Pipe()
            result_pipe = Pipe()
            worker_args = (index, task_pipe[1], result_pipe[0], self.population, env_name)
            worker = Process(target=rollout_worker, args=worker_args)
            self.evo_task_pipes.append(task_pipe)
            self.evo_result_pipes.append(result_pipe)
            self.evo_workers.append(worker)
            worker.start()

        # test bucket
        self.test_bucket = self.manager.list()
        policy = DeterministicPolicy(state_dim, action_dim, hidden_sizes, -1).eval()
        self.test_bucket.append(policy)
        self.test_task_pipes = []
        self.test_result_pipes = []
        self.test_workers = []
        for index in range(10):
            task_pipe = Pipe()
            result_pipe = Pipe()
            worker_args = (index, task_pipe[1], result_pipe[0], self.test_bucket, env_name)
            worker = Process(target=rollout_worker, args=worker_args)
            self.test_task_pipes.append(task_pipe)
            self.test_result_pipes.append(result_pipe)
            self.test_workers.append(worker)
            worker.start()

        self.best_score = - float('inf')
        self.total_frames = 0

    # receive EA's rollout
    def receive_ea_rollout(self):
        pop_fitness = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            entry = self.evo_result_pipes[i][1].recv()
            net_index = entry[0]
            fitness = entry[1]
            transitions = entry[2]
            pop_fitness[i] = fitness

            self.total_frames += len(transitions)
            if fitness > self.best_score:
                self.best_score = fitness
                self.best_policy = deepcopy(self.population[net_index])
        return pop_fitness

    def train(self, gen, logger):
        # EA's rollout
        for index in range(self.pop_size):
            self.evo_task_pipes[index][0].send(index)

        # receive all transitions
        pop_fitness = self.receive_ea_rollout()

        # logging population fitness and learners fitness
        logger.add_fitness(self.total_frames, pop_fitness.tolist())

        # test champ policy in the population
        champ_index = pop_fitness.argmax()
        self.test_bucket[0] = deepcopy(self.population[champ_index])
        for pipe in self.test_task_pipes:
            pipe[0].send(0)

        test_scores = []
        for pipe in self.test_result_pipes:
            entry = pipe[1].recv()
            test_scores.append(entry[1])
        test_mean = np.mean(test_scores)
        test_std = np.std(test_scores)
        logger.add_test_score(self.total_frames, test_mean.item())

        # EA step
        self.EA.epoch(gen, self.genealogy, self.population, pop_fitness.tolist())

        # allocate learners' rollout according to ucb score

        champ_wwid = int(self.population[champ_index].wwid.item())

        return pop_fitness, test_mean, test_std, champ_wwid
