import pygame
from configs import load_config
from utils import Position, get_random_position

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
        self.population = [Goal(screen, get_random_position(100)) for _ in range(self.population_size)]

    def randomize_position(self):
        for goal in self.population:
            goal.position = get_random_position(100)

    def draw(self):
        for goal in self.population:
            goal.draw()