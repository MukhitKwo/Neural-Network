import math
import numpy as np
import copy
from neural_network_config import NeuralNetworkConfig


class NeuralNetwork:
    def __init__(self, position: tuple, goals: list, config: NeuralNetworkConfig):
        self.position = position
        self.goals = goals
        self.hidden_layer_parameters = self.set_parameters(config.hidden_layer_dimensions)
        self.mutation_rate = config.mutation_rate

    def generate_random_weights(self, rows, cols):
        return np.random.uniform(-1, 1, (rows, cols))  # generate a matrix of random weights between -1 and 1

    def generate_random_biases(self, size):
        return np.random.uniform(-1, 1, size)  # generate a list of random bias between -1 and 1

    def set_parameters(self, layer_dimensions):
        params = []
        for layer, dimension in enumerate(layer_dimensions):
            weights = self.generate_random_weights(dimension, layer_dimensions[layer - 1] if layer > 0 else (len(self.goals) * 3))  # x neurons, x inputs
            bias = self.generate_random_biases(dimension)  # x neurons
            params.append([weights, bias])

        return params

    def calculate_activation(self, inputs, weights, biases):
        raw_sum = []
        for neuron_weights in weights:  # for each row of weights
            raw_sum.append(sum(n * w for n, w in zip(inputs, neuron_weights)))  # multiply the values by the weights and sum everything

        final_sum = [v + b for v, b in zip(raw_sum, biases)]  # add bias

        activation = [math.tanh(v) for v in final_sum]  # normalize

        return activation

    def forward(self, input_layer):

        # todo turn layers dynamic
        # LAYER 1
        activation_layer_1 = self.calculate_activation(input_layer, self.hidden_layer_parameters[0][0], self.hidden_layer_parameters[0][1])

        # LAYER 2
        activation_layer_2 = self.calculate_activation(activation_layer_1, self.hidden_layer_parameters[1][0], self.hidden_layer_parameters[1][1])

        return activation_layer_2

    def mutate(self):
        # todo use Gaussian distribution instant of uniform
        # add a small random noise to the weights and biases
        rate = self.mutation_rate
        for layer in self.hidden_layer_parameters:
            layer[0] += np.random.uniform(-rate, rate, layer[0].shape)
            layer[1] += np.random.uniform(-rate, rate, layer[1].shape)
