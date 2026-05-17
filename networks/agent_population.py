import random

from agent import Agent
import copy


class AgentPopulation:
    def __init__(self, screen, start_position, goals, config):
        self.start_position = start_position
        self.population = [Agent(screen, self.start_position, goals, config) for _ in range(config.population_size)]
        self.goals = goals

    def forward(self, frame):
        for player in self.population:
            player.update(frame)
            player.draw()

    def reproduce(self):
        best_player = None

        for player in self.population:
            player.fitness_proximity_to_goal()
            if player.fitness > (best_player.fitness if best_player else 0):
                best_player = player
                
        print("Best fitness:",best_player.fitness)

        for i, player in enumerate(self.population):
            player.inherit_best_player(self.start_position, copy.deepcopy(best_player.hidden_layer_parameters), self.goals)

            if i != 0:
                player.mutate()

        for goal in self.goals:
            goal.position = (random.randint(100, 1100), random.randint(100, 800))
            
