import math
import numpy as np
import copy

FRAMES = 20
POPULATION_SIZE = 10
MAX_SPEED = 3


class NeuralNetwork:
    def __init__(self, p_pos: tuple, f_pos: tuple):
        self.player_pos = p_pos
        self.fruit_pos = f_pos
        self.layer_1_weights = self.generate_random_weights(3, 3)  # 3 neurons, 3 inputs
        self.layer_1_bias = self.generate_random_biases(3)
        self.layer_2_weights = self.generate_random_weights(2, 3)  # 2 neurons, 3 inputs
        self.layer_2_bias = self.generate_random_biases(2)

    def generate_random_weights(self, rows, cols):
        return np.random.uniform(-1, 1, (rows, cols))  # generate a matrix of random weights between -1 and 1

    def generate_random_biases(self, size):
        return np.random.uniform(-1, 1, size)  # generate a list of random bias between -1 and 1

    def calculate_A(self, inputs, weights, biases):
        raw_Z = []
        for neuron_weights in weights:  # for each row of weights
            raw_Z.append(sum(n * w for n, w in zip(inputs, neuron_weights)))  # multiply the values by the weights and sum everything

        Z = [v + b for v, b in zip(raw_Z, biases)]  # add bias

        A = [math.tanh(v) for v in Z]  # normalize
        return A

    def forward(self):

        dx = (self.fruit_pos[0] - self.player_pos[0]) / 800  # get x distance and normalize it to -1 and 1
        dy = (self.fruit_pos[1] - self.player_pos[1]) / 600  # same for y
        dist = math.sqrt(dx**2 + dy**2)  # get the distance also normalized

        raw_values = [dx / dist, dy/dist, 1 / (1 + dist)]

        # LAYER 1
        A_layer_1 = self.calculate_A(raw_values, self.layer_1_weights, self.layer_1_bias)

        # LAYER 2
        A_layer_2 = self.calculate_A(A_layer_1, self.layer_2_weights, self.layer_2_bias)

        # OUTPUT
        angle = (A_layer_2[0] + 1) / 2 * 360  # convert normilizaed value to angle
        speed = (A_layer_2[1] + 1) / 2 * MAX_SPEED  # convert normalized value to speed

        # get the postion of the player based on angle and speed
        angle_rad = math.radians(angle)
        self.player_pos = (
            self.player_pos[0] + math.cos(angle_rad) * speed,
            self.player_pos[1] + math.sin(angle_rad) * speed
        )

        return self.player_pos

    def get_fitness(self):
        p1 = self.player_pos
        p2 = self.fruit_pos
        distance = (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2
        fitness = 1 / (distance / 100)
        return fitness

    def mutate(self, rate=0.1):
        # add a small random noise to the weights and biases
        self.layer_1_weights += np.random.uniform(-rate, rate, self.layer_1_weights.shape)
        self.layer_1_bias += np.random.uniform(-rate, rate, self.layer_1_bias.shape)
        self.layer_2_weights += np.random.uniform(-rate, rate, self.layer_2_weights.shape)
        self.layer_2_bias += np.random.uniform(-rate, rate, self.layer_2_bias.shape)
