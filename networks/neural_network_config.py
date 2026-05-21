class NeuralNetworkConfig:
    def __init__(self, number_inputs: int, hidden_layer_dimensions: list, mutation_rate: float, mutation_prob: float):
        self.number_inputs = number_inputs
        self.hidden_layer_dimensions = hidden_layer_dimensions
        self.mutation_rate = mutation_rate
        self.mutation_prob = mutation_prob


class FitnessConfig:
    def __init__(self, closeness_multiplier: float, time_bonus_multiplier: int, goals_reached_multiplier: int):
        self.closeness_multiplier = closeness_multiplier
        self.time_bonus_multiplier = time_bonus_multiplier
        self.goals_reached_multiplier = goals_reached_multiplier


class AgentConfig:
    def __init__(self, max_speed: float, max_degrees: int, fitness: FitnessConfig):
        self.start_position = (600, 450)
        self.max_speed = max_speed
        self.max_degrees = max_degrees
        self.fitness = fitness

class SimulationConfig:  # TODO: convert to json
    def __init__(self):
        self.neuralNetwork = NeuralNetworkConfig(
            number_inputs=3,
            hidden_layer_dimensions=[8, 6, 4],
            mutation_rate=0.02,
            mutation_prob=0.2
        )
        self.agent = AgentConfig(
            max_speed=10,
            max_degrees=360,
            fitness=FitnessConfig(
                closeness_multiplier=0.5,
                time_bonus_multiplier=20,
                goals_reached_multiplier=100),
        )
