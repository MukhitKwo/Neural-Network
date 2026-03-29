import math
import numpy as np
import copy
from neural_network_config import NeuralNetworkConfig


class NeuralNetwork:
    def __init__(self, goals: list, config: NeuralNetworkConfig):
        self.goals = goals
        self.hidden_layer_parameters = self.set_initial_parameters(config.hidden_layer_dimensions)
        self.mutation_rate = config.mutation_rate

    def set_initial_parameters(self, layer_dimensions):

        def generate_random_weights(rows, cols):
            return np.random.uniform(-1, 1, (rows, cols))  # generate a matrix of random weights between -1 and 1

        def generate_random_biases(size):
            return np.random.uniform(-1, 1, size)  # generate a list of random bias between -1 and 1

        params = []
        for layer, dimension in enumerate(layer_dimensions):
            weights = generate_random_weights(dimension, layer_dimensions[layer - 1] if layer > 0 else (len(self.goals) * 3))  # x neurons, x inputs
            bias = generate_random_biases(dimension)  # x neurons
            params.append([weights, bias])

        return params

    def forward(self, input_layer):

        previous_activation_layer = self.calculate_activation(input_layer, self.hidden_layer_parameters[0][0], self.hidden_layer_parameters[0][1])
        for l in range(1, len(self.hidden_layer_parameters)):
            previous_activation_layer = self.calculate_activation(
                previous_activation_layer, self.hidden_layer_parameters[l][0], self.hidden_layer_parameters[l][1])

        return previous_activation_layer

    def calculate_activation(self, inputs, weights, biases):
        # raw_sum = []
        # for neuron_weights in weights:  # for each row of weights
        #     raw_sum.append(sum(n * w for n, w in zip(inputs, neuron_weights)))  # multiply the values by the weights and sum everything

        # final_sum = [v + b for v, b in zip(raw_sum, biases)]  # add bias

        raw_sum = np.dot(weights, inputs)  # * need to understand exactly how dot product works
        final_sum = raw_sum + biases

        activation = [math.tanh(v) for v in final_sum]  # normalize

        return activation

    def mutate(self):
        # add a small random noise to the weights and biases
        # todo use Gaussian distribution instant of uniform
        rate = self.mutation_rate
        for layer in self.hidden_layer_parameters:
            layer[0] += np.random.normal(0, scale=rate, size=layer[0].shape)  # mutate weights
            layer[1] += np.random.normal(0, scale=rate, size=layer[1].shape)  # mutate biases
