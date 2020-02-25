import os

import numpy as np


class Logger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # population's info
        self.population_fitnesses = []
        self.test_scores = []

    def add_fitness(self, total_frame, population_fitness):
        item = [total_frame] + population_fitness
        self.population_fitnesses.append(item)

    def add_test_score(self, total_frame, test_score):
        item = [total_frame, test_score]
        self.test_scores.append(item)

    def save(self):
        file_name = 'population_fitnesses.txt'
        file_path = os.path.join(self.save_dir, file_name)
        np.savetxt(file_path, self.population_fitnesses)

        file_name = 'test_scores.txt'
        file_path = os.path.join(self.save_dir, file_name)
        np.savetxt(file_path, self.test_scores)
