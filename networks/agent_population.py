from math import log
import random

from agent import Agent
import copy


class AgentPopulation:
    def __init__(self, screen, start_position, goals, config):
        self.start_position = start_position
        self.config = config
        self.population = [Agent(screen, self.start_position, goals, config) for _ in range(self.config.population_size)]
        self.goals = goals

    def forward(self, frame):

        # if (len(self.population)):
        #     return

        for player in self.population[:]:
            player.update(frame)

            if player.closest_goal is None:
                return False

            if player.outside_of_window:
                self.population.remove(player)

            player.draw()

        return True

    def reproduce(self, screen):  # todo: refactor ts

        if (len(self.population)) == 0:

            for a in range(self.config.population_size):

                agent = Agent(screen, self.start_position, self.goals, self.config)

                self.population.append(agent)

                if a != 0:
                    agent.mutate()

            return

        best_agent = None

        for agent in self.population:

            agent.fitness_proximity_to_goal()

            if agent.fitness > (best_agent.fitness if best_agent else 0):
                best_agent = agent

        print("Best fitness:", best_agent.fitness)

        best_hidden_layer_parameters = copy.deepcopy(best_agent.hidden_layer_parameters)
        self.population = []

        for a in range(self.config.population_size):

            agent = Agent(screen, self.start_position, self.goals, self.config)
            agent.hidden_layer_parameters = copy.deepcopy(best_hidden_layer_parameters)

            self.population.append(agent)

            if a != 0:
                agent.mutate()
