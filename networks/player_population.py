import random

from player import Player
import copy


class PlayerPopulation:
    def __init__(self, screen, start_position, goals, config):
        self.start_position = start_position
        self.population = [Player(screen, self.start_position, goals, config) for _ in range(config.population_size)]
        self.goals = goals

    def forward(self, frame):
        for player in self.population:
            player.update(frame)
            player.draw()

    def reproduce(self):
        best_player = None

        for player in self.population:
            player.fitness_proximity_to_fruit()
            if player.fitness > (best_player.fitness if best_player else 0):
                best_player = player
                
        print("Best fitness:",best_player.fitness)

        for i, player in enumerate(self.population):
            player.inherit_best_player(self.start_position, best_player.hidden_layer_parameters, self.goals)

            if i != 0:
                player.mutate()

        self.goals[0].position = (random.randint(100, 1100), random.randint(100, 800))
        self.goals[1].position = (random.randint(100, 1100), random.randint(100, 800))
        self.goals[2].position = (random.randint(100, 1100), random.randint(100, 800))
