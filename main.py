"""
author: Diego Di Benedetto, Julia Hindel
"""
import run_simulation

# parameters to run
generations = 2  # number of generations
no_simulations = 1
type_NN = 2  # set the neural network type.
    # type 1=(input nodes:12 + 2, output nodes:2)
    # type 2=(input nodes:12 + 4, hiddel nodes:4, output nodes:2)
time_interval_RNN = 3  # set time interval for recursion of the neural network
two_point_crossover = True  # set whether to use one point crossover(False) or two points(True)
bit_length = 64  # bit length for each weight in the neural network
mutation_p = 0.03  # probability of flipping the value for each bit
selection_factor = 2  # selection factor, usually 1/5 of population size
population_size = 10
test_name = 'test_function'

if __name__ == "__main__":
    run_simulation.run(generations, no_simulations, type_NN, time_interval_RNN, two_point_crossover, bit_length, mutation_p,
                       selection_factor,
                       population_size, test_name)
