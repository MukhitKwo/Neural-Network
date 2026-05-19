from math import log
import random

from agent import Agent
import copy


class AgentPopulation:
    def __init__(self, screen, start_position, goals, population_size, config):
        self.start_position = start_position
        self.config = config
        self.population_size = population_size
        self.population = [Agent(screen, self.start_position, goals, config, True) for _ in range(self.population_size)]
        self.goals = goals

    def forward(self, frame):
        for player in self.population[:]:
            player.update(frame)
            player.draw()

    def reproduce(self, screen):  # TODO: refactor ts

        best_agent = None

        for agent in self.population:
            agent.fitness_proximity_to_goal()

            if agent.fitness > (best_agent.fitness if best_agent else 0):
                best_agent = agent

        print("Best fitness:", best_agent.fitness)

        best_hidden_layer = copy.deepcopy(best_agent.hidden_layers)
        self.population = []

        for a in range(self.population_size):
            agent = Agent(screen, self.start_position, self.goals, self.config, False)
            agent.inherit_hidden_layers(best_hidden_layer)

            self.population.append(agent)

            if a != 0:
                agent.mutate()
