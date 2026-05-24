import random

from configs import load_config


config = load_config()

WIDTH = config["simulation"]["window_width"]
HEIGHT = config["simulation"]["window_height"]

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def xy(self):
        return (self.x, self.y)


def get_random_position(margin):   
    return Position(random.randint(margin, WIDTH - margin), random.randint(margin, HEIGHT - margin))


class OutputValues:
    def __init__(self, angle, speed):
        self.angle = angle
        self.speed = speed
