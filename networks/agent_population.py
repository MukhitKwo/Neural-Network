from agent import Agent
import copy

from configs import Position


class AgentPopulation:
    def __init__(self, screen, goals, population_size, start_position: Position, config):
        self.screen = screen
        self.config = config
        self.start_position = start_position
        self.population_size = population_size
        self.population = [Agent(screen, self.start_position, goals, config, True) for _ in range(self.population_size)]
        self.goals = goals

    def forward(self, step):
        for player in self.population[:]:
            player.update(step)
            player.draw()

    def reproduce(self):  # TODO: refactor ts

        best_agent = None

        for agent in self.population:
            if agent.fitness > (best_agent.fitness if best_agent else 0):
                best_agent = agent

        print("Best fitness:", best_agent.fitness)

        best_hidden_layer = copy.deepcopy(best_agent.hidden_layers)
        self.population = []

        for a in range(self.population_size):
            agent = Agent(self.screen, self.start_position, self.goals, self.config, False)
            agent.inherit_hidden_layers(best_hidden_layer)

            self.population.append(agent)

            if a != 0:
                agent.mutate()
