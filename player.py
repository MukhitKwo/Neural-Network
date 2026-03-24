import pygame
from neural_network import NeuralNetwork
import math
import copy


class Player:
    def __init__(self, screen, player_pos, fruit_pos):
        self.screen = screen
        self.player_pos = player_pos
        self.fruit_pos = fruit_pos
        self.color = (0, 0, 255)
        self.radius = 25
        self.neural_network = NeuralNetwork(self.player_pos[:], fruit_pos[:])

    def forward(self):
        new_pos = self.neural_network.forward()
        self.player_pos = new_pos
        

    def draw(self):
        pygame.draw.circle(self.screen, self.color, self.player_pos, self.radius)
