from player import Player
import copy


class PlayerPopulation:
    def __init__(self, screen, start_position, goals, config):
        self.start_position = start_position
        self.population = [Player(screen, self.start_position, goals, config) for _ in range(config.population_size)]

    def forward(self):
        for player in self.population:
            input_layer = player.input_layer()
            last_hidden_layer = player.forward(input_layer)
            output_layer = player.output_layer(last_hidden_layer)
            player.set_player_position(output_layer)
            player.draw()

    def reproduce(self):
        best_player = None

        for player in self.population:
            if player.fitness() > (best_player.fitness() if best_player else 0):
                best_player = player

        for i, player in enumerate(self.population):
            player.position = self.start_position
            player.hidden_layer_parameters = copy.deepcopy(best_player.hidden_layer_parameters)

            if i != 0:
                player.mutate()
