"""
author: Julia Hindel
"""

import numpy as np
from robot import Robot


def tanh(x):
    return (2 / (1 + np.exp(-2 * x))) - 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork2:

    def __init__(self, weights, world, time_interval_RNN=None, robot_steps=600):
        self.time_interval = time_interval_RNN
        self.weights1 = np.array(weights[:64]).reshape(4, -1)
        self.weights2 = np.array(weights[64:]).reshape(2, -1)
        self.robot = Robot(world)
        self.max_robot_steps = robot_steps
        self.time = 0
        self.dic = [(0, [0, 0, 0, 0])]
        self.fitness = 0

    # print("NN created")

    def forward(self):
        past_activity = self.time - self.time_interval if self.time > self.time_interval else 0
        recurrent = dict(self.dic)[past_activity]
        input = np.array([*(self.robot.get_sensors()[:, 2] / 200), *recurrent])
        intermediate_layer = tanh(np.dot(input, self.weights1.T))
        output = tanh(np.dot(intermediate_layer, self.weights2.T))
        self.fitness += self.robot.update_velocity(*output)
        self.dic.append((self.time, intermediate_layer))
        self.time += 1

    def run_robot(self):
        count = 0
        while count < self.max_robot_steps:
            self.forward()
            count += 1
        # print("dust_value", self.robot.world.get_dust_value())
        # doing * 5 here as dust is important
        self.fitness /= self.max_robot_steps
        if self.robot.world.display:
            print("fitness before dust", self.fitness)
        self.fitness += self.robot.world.get_dust_value() * 5
        # print("fitness", self.fitness)
        return self.fitness
