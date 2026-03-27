import pygame
from neural_network import NeuralNetwork


class Player(NeuralNetwork):
    def __init__(self, screen, player_pos, fruit_pos):
        super().__init__(player_pos, fruit_pos)
        self.screen = screen
        self.color = (0, 0, 255)
        self.radius = 25

    def reset_pos(self, new_pos):
        self.player_pos = new_pos

    def draw(self):
        pygame.draw.circle(self.screen, self.color, self.player_pos, self.radius)
