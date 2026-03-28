import pygame
from neural_network import NeuralNetwork
import random

class Player(NeuralNetwork):
    def __init__(self, screen, position, goals, config):
        super().__init__(position, goals, config)
        self.screen = screen
        self.color = (0, 0, random.randint(200, 255))
        self.radius = 25

    def draw(self):
        pygame.draw.circle(self.screen, self.color, self.position, self.radius)
