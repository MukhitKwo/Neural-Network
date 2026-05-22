import random

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
        for player in self.population:
            player.update(step)
            player.draw()

    def reproduce(self):

        new_population = []

        for _ in range(self.population_size - 5):
            sample = random.sample(self.population, 10)

            best_agent = max(sample, key=(lambda agent: agent.fitness))

            new_agent = Agent(self.screen, self.start_position, self.goals, self.config, False)
            new_agent.inherit_hidden_layers(best_agent.hidden_layers)

            new_agent.mutate()

            new_population.append(new_agent)

        top_agents = sorted(self.population, key=lambda agent: agent.fitness, reverse=True)[:5]
        
        print("Best Fitness:", top_agents[0].fitness)

        for agent in top_agents:
            new_agent = Agent(self.screen, self.start_position, self.goals, self.config, False)
            new_agent.inherit_hidden_layers(agent.hidden_layers)

            new_population.append(new_agent)

        self.population = new_population
