import numpy as np
from configs import load_config

config = load_config()


class NeuralNetwork:
    def __init__(self, generate_hidden_layers):
        self.hidden_layers = self.set_initial_parameters(config["network"]["hidden_layer_dimensions"]) if generate_hidden_layers else None
        self.mutation_rate = config["network"]["mutation_rate"]
        self.mutation_prob = config["network"]["mutation_probability"]

    def set_initial_parameters(self, layer_dimensions):
        # generates random weights and biases
        def generate_random_weights(rows, cols):
            return np.random.uniform(-1, 1, (rows, cols))  # generate a matrix of random weights between -1 and 1

        def generate_random_biases(size):
            return np.random.uniform(-1, 1, size)  # generate a list of random bias between -1 and 1

        hidden_layers = []
        for layer, dimension in enumerate(layer_dimensions):
            weights = generate_random_weights(dimension, layer_dimensions[layer - 1] if layer > 0 else 5)  # x neurons, x inputs
            bias = generate_random_biases(dimension)  # x neurons
            hidden_layers.append([weights, bias])

        return hidden_layers

    def forward(self, input_vector):
        # calculates each layer based on the previous layer
        previous_activation_vector = self.calculate_activation(input_vector, self.hidden_layers[0][0], self.hidden_layers[0][1])

        for l in range(1, len(self.hidden_layers)):
            previous_activation_vector = self.calculate_activation(previous_activation_vector, self.hidden_layers[l][0], self.hidden_layers[l][1])

        return previous_activation_vector

    def calculate_activation(self, inputs, weights, biases):
        # sums the weights and biases and "activates" them betwen -1 and 1
        weights_sum = np.dot(weights, inputs)  # * need to understand exactly how dot product works
        total_sum = weights_sum + biases

        activated_vector = np.tanh(total_sum)

        return activated_vector

    def mutate(self):
        # has a probability to add a small random noise to the weights and biases
        for layer in self.hidden_layers:
            weights_prob_mask = np.random.random(layer[0].shape) < self.mutation_prob
            biases_prob_mask = np.random.random(layer[1].shape) < self.mutation_prob

            layer[0] += weights_prob_mask * np.random.normal(0, scale=self.mutation_rate, size=layer[0].shape)
            layer[1] += biases_prob_mask * np.random.normal(0, scale=self.mutation_rate, size=layer[1].shape)
