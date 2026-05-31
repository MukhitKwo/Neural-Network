import random
import tomllib


def load_config() -> dict:
    with open("config.toml", "rb") as f:
        return tomllib.load(f)


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


class AccelerationValues:
    def __init__(self, ax, ay):
        self.ax = ax
        self.ay = ay
