class NeuralNetworkConfig:
    def __init__(self, hidden_layer_dimensions: list, mutation_rate: float, mutation_prob: float):
        self.hidden_layer_dimensions = hidden_layer_dimensions
        self.mutation_rate = mutation_rate
        self.mutation_probability = mutation_prob


class FitnessConfig:
    def __init__(self, proximity_multiplier: float, time_bonus_multiplier: int, goals_reached_multiplier: int):
        self.proximity_multiplier = proximity_multiplier
        self.time_bonus_multiplier = time_bonus_multiplier
        self.goals_reached_multiplier = goals_reached_multiplier


class AgentConfig:
    def __init__(self, max_speed: float, max_degrees: int, fitness: FitnessConfig):
        self.max_speed = max_speed
        self.max_degrees = max_degrees
        self.fitness = fitness

class SimulationConfig:  # TODO: convert to json
    def __init__(self):
        self.network = NeuralNetworkConfig(
            hidden_layer_dimensions=[8, 6, 4],
            mutation_rate=0.1,
            mutation_prob=0.1
        )
        self.agent = AgentConfig(
            max_speed=10,
            max_degrees=360,
            fitness=FitnessConfig(
                proximity_multiplier=0.1,
                time_bonus_multiplier=10,
                goals_reached_multiplier=100),
        )

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def xy(self):
        return (self.x, self.y)
    
    def set_xy(self, x, y):
        self.x += x
        self.y += y