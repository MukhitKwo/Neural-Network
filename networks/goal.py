import pygame

from configs import Position


class Goal:
    def __init__(self, screen, position: Position):
        self.screen = screen
        self.position = position
        self.color = (128, 0, 128)
        self.radius = 15

    def draw(self):
        pygame.draw.circle(self.screen, self.color, self.position.xy, self.radius)