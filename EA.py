"""
author: Diego Di Benedetto, Julia Hindel
"""

import random
from copy import copy
from NN import NeuralNetwork
from NN2 import NeuralNetwork2
from world import World
import matplotlib.pyplot as plt
import time
import concurrent.futures as futures
import numpy as np

parallel=True


def flip_bit(x):
    if x == '0':
        return '1'
    else:
        return '0'


def hamming_distance(s1, s2):
    return sum(s1 != s2 for s1, s2 in zip(s1, s2))


def run(phenotype):
    # print(f"{phenotype} is being executed")
    return phenotype.fitness(), phenotype


class EA:

    def __init__(self, population_size=10, bit_length=64, mutation_p=0.03,
                 selection_factor=2, type_NN=1, time_interval_RNN=0, loaded_weights=None, two_point_crossover_bool=None, test_name=None, cluster=True):
        self.test_name = test_name
        self.two_point_crossover_bool = two_point_crossover_bool
        self.finished = False
        room_config = 0
        double_trapezoid_config = 5
        if loaded_weights is None or (loaded_weights is not None and population_size != len(loaded_weights)):
            if loaded_weights is not None:
                print(f'{len(loaded_weights)} weights loaded but the population size is: {population_size}. '
                      f'Continuing with random initialization..')
                loaded_weights = None
            self.current_population = [Phenotype(bit_length, type_NN, time_interval_RNN, room_config, double_trapezoid_config,
                                                 loaded_weights) for _ in range(population_size)]
        else:
            self.current_population = [Phenotype(bit_length, type_NN, time_interval_RNN, room_config, double_trapezoid_config, weight)
                                       for _, weight in zip(range(population_size), loaded_weights)]
        self.prev_populations = [self.current_population]
        self.mutation_p = mutation_p
        self.selection_factor = selection_factor
        self.best_list = []
        self.pool = futures.ProcessPoolExecutor(8)
        self.tasks = []
        self.cluster=cluster

    def run_generation(self, count):
        # for i in self.current_population:
        #    print("before", i, i.get_bit_string(), i.binary_decoder())
        self.evaluate(count)
        self.reproduction()
        if self.two_point_crossover_bool:
            self.two_points_crossover()
        else:
            self.crossover()
        self.mutation()
        self.prev_populations.append(self.current_population)  # data collection

    def reproduction(self):
        # truncated rank based selection with generational replacement
        # the population is sorted and divided by "selection_factor" (usually 1/5) and reproduced
        repetition_factor = int(len(self.current_population) / self.selection_factor)
        sorted_fitness_list = self.fitness_list[:self.selection_factor]
        self.current_population = [copy(phenotype[1]) for phenotype in sorted_fitness_list for _ in
                                   range(repetition_factor)]
        # for i in self.current_population:
        #    print("reproduction", i, i.get_bit_string(), i.binary_decoder())

    def crossover(self):
        # one point crossover, index is chosen randomly
        population = self.current_population.copy()
        random.shuffle(population)
        self.current_population = []
        while len(population) > 1:
            phenotype1 = population.pop()
            phenotype2 = population.pop()
            index = random.randint(1, len(phenotype1.get_bit_string()) - 1)
            head1, tail1 = phenotype1.get_bit_string()[:index], phenotype1.get_bit_string()[index:]
            head2, tail2 = phenotype2.get_bit_string()[:index], phenotype2.get_bit_string()[index:]
            phenotype1.set_bit_string(head1 + tail2)
            phenotype2.set_bit_string(head2 + tail1)
            self.current_population.append(phenotype1)
            self.current_population.append(phenotype2)

        if len(population) == 1:
            self.current_population.append(population.pop())

        # for i in self.current_population:
        #    print("crossover", i, i.get_bit_string(), i.binary_decoder())

    def uniform_crossover(self):
        # uniform crossover, is not applied since we have bitstrings
        population = self.current_population.copy()
        random.shuffle(population)
        self.current_population = []
        while len(population) > 1:
            phenotype1 = population.pop()
            phenotype2 = population.pop()
            bitlist1 = list(phenotype1.get_bit_string())
            bitlist2 = list(phenotype1.get_bit_string())
            for i in range(len(bitlist1)):
                if random.random() < 0.5 and bitlist1[i] != bitlist2[i]:
                    print(i, bitlist1)
                    print(i, bitlist2)
                    temp = bitlist1[i]
                    bitlist1[i] = bitlist2[i]
                    bitlist2[i] = temp
                    print(bitlist1)
                    print(bitlist2)
                print(i)

            bitstring1 = ''.join(bitlist1)
            bitstring2 = ''.join(bitlist2)
            phenotype1.set_bit_string(bitstring1)
            phenotype2.set_bit_string(bitstring2)
            self.current_population.append(phenotype1)
            self.current_population.append(phenotype2)

        if len(population) == 1:
            self.current_population.append(population.pop())

        # for i in self.current_population:
        #   print("crossover", i, i.get_bit_string(), i.binary_decoder())

    def two_points_crossover(self):
        # two points crossover, two index chosen randomly between the first and second half of the lenght of bitstring
        population = self.current_population.copy()
        random.shuffle(population)
        self.current_population = []
        while len(population) > 1:
            phenotype1 = population.pop()
            phenotype2 = population.pop()
            index1 = random.randint(1, int(len(phenotype1.get_bit_string()) / 2))
            index2 = random.randint(int(len(phenotype1.get_bit_string()) / 2 + 1), len(phenotype1.get_bit_string()) - 1)
            head1, body1, tail1 = phenotype1.get_bit_string()[:index1], phenotype1.get_bit_string()[index1:index2], \
                                  phenotype1.get_bit_string()[index2:]
            head2, body2, tail2 = phenotype2.get_bit_string()[:index1], phenotype2.get_bit_string()[index1:index2], \
                                  phenotype2.get_bit_string()[index2:]
            phenotype1.set_bit_string(head1 + body2 + tail1)
            phenotype2.set_bit_string(head2 + body1 + tail2)
            self.current_population.append(phenotype1)
            self.current_population.append(phenotype2)

        if len(population) == 1:
            self.current_population.append(population.pop())

        # for i in self.current_population:
        #   print("crossover", i, i.get_bit_string(), i.binary_decoder())

    def mutation(self):
        # mutation is applied for each bit with "mutation_p" probability
        for i in range(len(self.current_population)):
            new_string = (flip_bit(bit) if random.random() < self.mutation_p else bit
                          for bit in self.current_population[i].get_bit_string())
            new_string = ''.join(new_string)
            self.current_population[i].set_bit_string(new_string)
        # for i in self.current_population:
        #    print("mutation", i, i.get_bit_string(), i.binary_decoder())

    def evaluate(self, count):
        # evaluation of the generation that can be done in parallel is "parallel" is set to True
        # the method calculates the maximum/average/minimum fitness of the generation
        # it displays the results if if generation number % 10 is 0
        if parallel:
            self.calc_fitness_parallel()
        else:
            self.calc_fitness()
        sum_hamming_dist = self.calc_hamming_dist()
        max_fitness = max(self.fitness_list, key=lambda x: x[0])[0]
        avg_fitness = sum([i[0] for i in self.fitness_list]) / len([i[0] for i in self.fitness_list])
        min_fitness = min(self.fitness_list, key=lambda x: x[0])[0]
        if max_fitness > 100000:
            print("max_fitness", max_fitness)
            self.finished = True
        best_phenotype = self.fitness_list[0]
        weights = best_phenotype[1].binary_decoder()
        self.best_list.append([count, min_fitness, avg_fitness, max_fitness, weights, sum_hamming_dist])
        if count % 10 == 0:
            np.save(f'output/log_best_list_{self.test_name}.npy', np.array(self.best_list, dtype=object))
            if not self.cluster:
                best_phenotype[1].fitness(True)
        print(f"generation {count}, best fitness {best_phenotype[0]}")

    def calc_fitness(self):
        time_start = time.time()
        self.fitness_list = sorted([(i.fitness(False), i) for i in self.current_population], key=lambda x: x[0],
                                   reverse=True)
        print("time", time.time() - time_start)

    def calc_fitness_parallel(self):
        time_start = time.time()
        for i in self.current_population:
            self.tasks.append(self.pool.submit(run, i))
        # print(self.tasks)
        done, not_done = futures.wait(self.tasks, timeout=None, return_when=futures.ALL_COMPLETED)
        results = [it.result(timeout=None) for it in done]
        self.tasks = []
        self.fitness_list = sorted(results, key=lambda x: x[0], reverse=True)
        print("time", time.time() - time_start)

    def calc_hamming_dist(self):
        sum = 0
        for p1 in self.current_population:
            for p2 in self.current_population:
                if p1 != p2:
                    sum += hamming_distance(p1.get_bit_string(), p2.get_bit_string())
        return sum

    def draw_fitness_landscape(self):
        # drawing plot of fitness landscape with maximum and average fitness
        plt.plot([i[0] for i in self.best_list], [i[3] for i in self.best_list], label='max')
        plt.plot([i[0] for i in self.best_list], [i[2] for i in self.best_list], label='avg')
        plt.title("Fitness landscape")
        plt.legend()
        plt.savefig(f"output/fitness_landscape_{self.test_name}")
        plt.show()
        plt.clf()
        print([i[0] for i in self.best_list], [i[5] for i in self.best_list])
        plt.plot([i[0] for i in self.best_list], [i[5] for i in self.best_list], label='hamming distance')
        plt.title("Hamming distance")
        plt.savefig(f"output/hamming distance_{self.test_name}")
        plt.show()

    def get_weights(self):
        return [genotype.get_bit_string() for genotype in self.current_population]


class Phenotype:

    def __init__(self, bit_length, type_NN, time_interval_RNN, world_config1, world_config2, loaded_weight):
        '''
        instance of phenotype.
        two worlds are create to make the training more robust. the world configuration parameters are set in EA
        '''
        self.world_config1 = world_config1
        self.world_config2 = world_config2
        self.world1 = World(False, world_config1)
        self.world2 = World(False, world_config2)
        self.type_NN = type_NN
        self.time_interval_RNN = time_interval_RNN
        self.split = 28
        if type_NN == 2:
            self.split = 72
        elif type_NN != 1:
            print("Unsupported network type")
        if loaded_weight is not None:
            self.bit_string = loaded_weight
        else:
            self.bit_string = ''.join('0' if random.random() < 0.5 else '1' for _ in range(bit_length * self.split))

    def __copy__(self):
        new_phenotype = Phenotype(len(self.get_bit_string()) / self.split, self.type_NN, self.time_interval_RNN, self.world_config1,
                                  self.world_config2, self.get_bit_string())
        return new_phenotype

    def get_bit_string(self):
        return self.bit_string

    def set_bit_string(self, new_string):
        self.bit_string = new_string

    def fitness(self, GUI=False):
        # fitness is calculated in both worlds and averaged
        return (self.get_fitness_from_world(self.world1, GUI) + self.get_fitness_from_world(self.world2, GUI)) / 2

    def get_fitness_from_world(self, world, GUI):
        # the neural network is created and run into the environment to get the fitness
        world.change_display(GUI)
        if self.type_NN == 1:
            nn = NeuralNetwork(self.binary_decoder(), world, self.time_interval_RNN)
        else:
            nn = NeuralNetwork2(self.binary_decoder(), world, self.time_interval_RNN)
        fitness = nn.run_robot()
        if GUI:
            world.display_dust()
            print("total fitness", fitness)
        return fitness

    def split_string(self, n):
        return [self.bit_string[i:i + n] for i in range(0, len(self.bit_string), n)]

    def binary_decoder(self):
        # decode bitstring in values(weights) for the neural network
        list_bits = self.split_string(int(len(self.bit_string) / self.split))
        normalization = (10 - (-10)) / (2 ** (len(self.bit_string) / self.split) - 1)
        weights = [-10 + (int(bits, 2) * normalization) for bits in list_bits]
        return weights
