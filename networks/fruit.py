import pygame


class Fruit:
    def __init__(self, screen, position):
        self.screen = screen
        self.position = position
        self.color = (128, 0, 128)
        self.radius = 25

    def draw(self):
        pygame.draw.circle(self.screen, self.color, (self.position[0], self.position[1]), self.radius)


