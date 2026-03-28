from player import Player
import copy


class PlayerPopulation:
    def __init__(self, screen, initial_player_pos, fruit, config):
        self.initial_player_pos = initial_player_pos
        self.population = [Player(screen, self.initial_player_pos, fruit.position, config) for _ in range(config.population_size)]

    def forward(self):
        for player in self.population:
            player.forward()
            player.draw()

    def reproduce(self):
        best_player = None

        for player in self.population:
            if player.fitness() > (best_player.fitness() if best_player else 0):
                best_player = player

        for i, player in enumerate(self.population):
            player.player_pos = self.initial_player_pos[:]
            player.parameters = copy.deepcopy(best_player.parameters)

            if i != 0:
                player.mutate()
