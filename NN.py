"""
author: Julia Hindel
"""

import numpy as np
from robot import Robot


def tanh(x):
    return (2 / (1 + np.exp(-2 * x))) - 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:

    def __init__(self, weights, world, robot_steps=600, time_interval=3):
        self.weights = np.array(weights).reshape(2, -1)
        self.robot = Robot(world)
        self.max_robot_steps = robot_steps
        self.time_interval_RNN = time_interval
        self.time = 0
        self.dic = [(0, [0, 0])]
        self.fitness = 0

    def forward(self):
        past_activity = self.time - self.time_interval if self.time > self.time_interval else 0
        recurrent = dict(self.dic)[past_activity]
        input = np.array([*(self.robot.get_sensors()[:, 2]/200), *recurrent])
        output = tanh(np.dot(input, self.weights.T))
        self.fitness += self.robot.update_velocity(*output)
        self.dic.append((self.time, output))
        self.time += 1

    def run_robot(self):
        count = 0
        while count < self.max_robot_steps:
            self.forward()
            count += 1
        # print("dust_value", self.robot.world.get_dust_value())
        self.fitness += self.robot.world.get_dust_value()
        # print("fitness", self.fitness)
        return self.fitness
