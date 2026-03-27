import pygame
from neural_network import NeuralNetwork
import math
import copy


class Player:
    def __init__(self, screen, player_pos, fruit_pos):
        self.screen = screen
        self.position = player_pos
        self.fruit_pos = fruit_pos
        self.color = (0, 0, 255)
        self.radius = 25
        self.neural_network = NeuralNetwork(self.position[:], fruit_pos[:])

    def forward(self):
        new_pos = self.neural_network.forward()
        self.position = new_pos

    def reset_pos(self, new_pos):
        self.position = new_pos
        self.neural_network.player_pos = self.position

    def draw(self):
        pygame.draw.circle(self.screen, self.color, self.position, self.radius)
