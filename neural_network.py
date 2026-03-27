import math
import numpy as np
import copy

FRAMES = 20
POPULATION_SIZE = 10
MAX_SPEED = 5


class NeuralNetwork:
    def __init__(self, player_pos: tuple, fruit_pos: tuple):
        self.player_pos = player_pos
        self.fruit_pos = fruit_pos
        self.parameters = self.set_parameters([[3, 3], [2, 3]])

    def generate_random_weights(self, rows, cols):
        return np.random.uniform(-1, 1, (rows, cols))  # generate a matrix of random weights between -1 and 1

    def generate_random_biases(self, size):
        return np.random.uniform(-1, 1, size)  # generate a list of random bias between -1 and 1

    def set_parameters(self, layers):
        params = []
        for values in layers:
            weights = self.generate_random_weights(values[0], values[1])  # x neurons, x inputs
            bias = self.generate_random_biases(values[0]) # x neurons
            params.append([weights, bias])

        return params

    def calculate_activation(self, inputs, weights, biases):
        raw_sum = []
        for neuron_weights in weights:  # for each row of weights
            raw_sum.append(sum(n * w for n, w in zip(inputs, neuron_weights)))  # multiply the values by the weights and sum everything

        sum = [v + b for v, b in zip(raw_sum, biases)]  # add bias

        activation = [math.tanh(v) for v in sum]  # normalize
        
        return activation

    def forward(self):

        dx = (self.fruit_pos[0] - self.player_pos[0]) / 800  # get x distance and normalize it to -1 and 1
        dy = (self.fruit_pos[1] - self.player_pos[1]) / 600  # same for y
        dist = math.sqrt(dx**2 + dy**2)  # get the distance also normalized

        raw_values = [dx / dist, dy/dist, 1 / (1 + dist)]

        # LAYER 1
        activation_layer_1 = self.calculate_activation(raw_values, self.parameters[0][0], self.parameters[0][1])

        # LAYER 2
        activation_layer_2 = self.calculate_activation(activation_layer_1, self.parameters[1][0], self.parameters[1][1])

        # OUTPUT
        angle = (activation_layer_2[0] + 1) / 2 * 360  # convert normilizaed value to angle
        speed = (activation_layer_2[1] + 1) / 2 * MAX_SPEED  # convert normalized value to speed

        # get the postion of the player based on angle and speed
        angle_rad = math.radians(angle)
        self.player_pos = (
            self.player_pos[0] + math.cos(angle_rad) * speed,
            self.player_pos[1] + math.sin(angle_rad) * speed
        )

        return self.player_pos

    def fitness(self):
        p1 = self.player_pos
        p2 = self.fruit_pos
        distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        fitness = 1 / (distance / 100)
        return fitness

    def mutate(self, rate=0.1):
        # add a small random noise to the weights and biases
        self.layer_1_weights += np.random.uniform(-rate, rate, self.layer_1_weights.shape)
        self.layer_1_bias += np.random.uniform(-rate, rate, self.layer_1_bias.shape)
        self.layer_2_weights += np.random.uniform(-rate, rate, self.layer_2_weights.shape)
        self.layer_2_bias += np.random.uniform(-rate, rate, self.layer_2_bias.shape)
