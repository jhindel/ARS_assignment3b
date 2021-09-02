"""
author: Diego Di Benedetto, Julia Hindel
"""

from EA import EA
import os


cluster = True
'''
The program create a new evolutionary algorithm instance and run for number of generation passed as parameter.
It is possible to load the weights to further train them setting "file name" and "load weights" to true.
At the end of the simulation the program write a file with the weights. If the "file name" already exists the 
program ask the user if overwrite it or not (console input: y/n) 
'''


def input_checking():
    input_is_correct = False
    while not input_is_correct:
        try:
            val = input('y/n ')
            if val != 'y' and val != 'n':
                raise ValueError
            input_is_correct = True
        except ValueError:
            print('Invalid input, please insert again.')
    if val == 'y':
        return True
    elif val == 'n':
        return False


def run(max, no_simulations, type_NN, time_interval_RNN, two_point_crossover, bit_length, mutation_p, selection_factor, population_size, test_name):
    file_name = f'output/log_weights_{test_name}.txt'
    load_weights = False  # set to true if you want to train saved weights
    loaded_weights = None

    if load_weights and os.path.exists(file_name):
        print("file doesn't exist")
        with open(file_name, 'r') as f:
            loaded_weights = f.read()
        loaded_weights = loaded_weights.replace("[", "").replace("]", "").replace("'", "").replace(" ", "").split(
            ',')
    for i in range(no_simulations):
        print(f'simulation : {i}')
        ea = EA(population_size=population_size, bit_length=bit_length, mutation_p=mutation_p,
                selection_factor=selection_factor, type_NN=type_NN, time_interval_RNN=time_interval_RNN, loaded_weights=loaded_weights,
                two_point_crossover_bool=two_point_crossover, test_name=test_name, cluster=cluster)

        count = 0
        while count < max and not ea.finished:
            # print(f"generation run {count}")
            ea.run_generation(count)
            count += 1
        ea.draw_fitness_landscape()
        weights = ea.get_weights()

    flag = True
    if os.path.exists(file_name):
        print('log file already exists, do you want to overwrite it?')
        flag = input_checking()
    if flag:
        with open(file_name, 'w') as f:
            f.write(str(weights))
