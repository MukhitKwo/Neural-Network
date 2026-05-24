import random
import pygame
from configs import load_config
from utils import Position

config = load_config()

class Goal:
    def __init__(self, screen, position: Position):
        self.screen = screen
        self.position = position
        self.color = (128, 0, 128)
        self.radius = 15

    def draw(self):
        pygame.draw.circle(self.screen, self.color, self.position.xy, self.radius)


class GoalPopulation():
    def __init__(self, screen):
        self.screen = screen
        self.population_size = config["simulation"]["goal_population_size"]
        self.population = [Goal(screen, Position(random.randint(100, 1100), random.randint(100, 800))) for _ in range(self.population_size)]

    def randomize_position(self):
        for goal in self.population:
            goal.position = Position(random.randint(100, 1100), random.randint(100, 800))

    def draw(self):
        for goal in self.population:
            goal.draw()