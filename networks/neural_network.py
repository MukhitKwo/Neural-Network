import math
import numpy as np
import copy
from neural_network_config import NeuralNetworkConfig


class NeuralNetwork:
    def __init__(self, config: NeuralNetworkConfig):
        self.number_inputs = config.number_inputs
        self.hidden_layer_parameters = self.set_initial_parameters(config.hidden_layer_dimensions)
        self.mutation_rate = config.mutation_rate
        self.mutation_prob = config.mutation_prob

    def set_initial_parameters(self, layer_dimensions):
        # generates random weights and biases
        def generate_random_weights(rows, cols):
            return np.random.uniform(-1, 1, (rows, cols))  # generate a matrix of random weights between -1 and 1

        def generate_random_biases(size):
            return np.random.uniform(-1, 1, size)  # generate a list of random bias between -1 and 1

        params = []
        for layer, dimension in enumerate(layer_dimensions):
            weights = generate_random_weights(dimension, layer_dimensions[layer - 1] if layer > 0 else self.number_inputs)  # x neurons, x inputs
            bias = generate_random_biases(dimension)  # x neurons
            params.append([weights, bias])

        return params

    def forward(self, input_layer):
        # calculates each layer based on the previous layer
        previous_activation_layer = self.calculate_activation(input_layer, self.hidden_layer_parameters[0][0], self.hidden_layer_parameters[0][1])
        for l in range(1, len(self.hidden_layer_parameters)):
            previous_activation_layer = self.calculate_activation(
                previous_activation_layer, self.hidden_layer_parameters[l][0], self.hidden_layer_parameters[l][1])

        return previous_activation_layer

    def calculate_activation(self, inputs, weights, biases):
        # sums the weights and biases and normalizes them betwen -1 and 1
        raw_sum = np.dot(weights, inputs)  # * need to understand exactly how dot product works
        final_sum = raw_sum + biases

        activation = [math.tanh(v) for v in final_sum]  # normalize

        return activation

    def mutate(self):
        # has a probability to add a small random noise to the weights and biases
        rate = self.mutation_rate
        for layer in self.hidden_layer_parameters:
            w_mask = np.random.random(layer[0].shape) < self.mutation_prob
            b_mask = np.random.random(layer[1].shape) < self.mutation_prob
            layer[0] += w_mask * np.random.normal(0, scale=rate, size=layer[0].shape)
            layer[1] += b_mask * np.random.normal(0, scale=rate, size=layer[1].shape)
