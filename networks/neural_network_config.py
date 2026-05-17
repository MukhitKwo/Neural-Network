class NeuralNetworkConfig:

    def __init__(self, number_inputs, hidden_layer_dimensions: list, mutation_rate: float, population_size: int, max_speed: float, max_degrees: int):
        self.number_inputs = number_inputs
        self.hidden_layer_dimensions = hidden_layer_dimensions
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.max_speed = max_speed
        self.max_degrees = max_degrees
