


class NeuralNetworkConfig:

    def __init__(self, number_inputs: int, hidden_layer_dimensions: list, mutation_rate: float, mutation_prob: float, max_speed: float, max_degrees: int):
        self.number_inputs = number_inputs
        self.hidden_layer_dimensions = hidden_layer_dimensions
        self.mutation_rate = mutation_rate
        self.mutation_prob = mutation_prob
        self.max_speed = max_speed
        self.max_degrees = max_degrees
