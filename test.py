"""
author: Julia Hindel
"""


from world import World
from NN import NeuralNetwork
from NN2 import NeuralNetwork2
import numpy as np

# TODO adjust
type_NN = 2
time_interval_RNN = 8
robot_steps = 1200
generations = [62]
test_name = 'test_final'
test_world_no = [1, 2, 3, 4]

'''Testing in different room'''
best_list = np.load(f'output/log_best_list_{test_name}.npy', allow_pickle=True)
# print(best_list)

for i in generations:
    weights = best_list[i][4]
    for w in test_world_no:
        test_world = World(True, w)
        if type_NN == 1:
            test_nn = NeuralNetwork(weights, test_world, robot_steps=robot_steps, time_interval=time_interval_RNN)
        else:
            test_nn = NeuralNetwork2(weights, test_world, robot_steps=robot_steps, time_interval_RNN=time_interval_RNN)
        fitness = test_nn.run_robot()
        print(f"gen {i}, world {w}, fitness {fitness}")
