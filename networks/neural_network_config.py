import numpy as np


class NeuralNetworkConfig:

    def __init__(self, parameters_shape, mutation_rate, population_size, max_speed, max_degrees):
        self.parameters_shape = parameters_shape
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.max_speed = max_speed
        self.max_degrees = max_degrees
