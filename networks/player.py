import pygame
from neural_network import NeuralNetwork
import random
import math


class OutputValues:
    def __init__(self, angle, speed):
        self.angle = angle
        self.speed = speed


class Player(NeuralNetwork):
    def __init__(self, screen, position, goals, config):
        self.screen = screen
        super().__init__(position, goals, config)
        self.max_degrees = config.max_degrees
        self.max_speed = config.max_speed
        self.color = (0, 0, random.randint(200, 255))
        self.radius = 25

    def draw(self):
        pygame.draw.circle(self.screen, self.color, self.position, self.radius)

    def input_layer(self):
        input_values = []
        for goal in self.goals:
            goal_pos = goal.position
            dx = (goal_pos[0] - self.position[0]) / 800  # get x distance and normalize it to -1 and 1
            dy = (goal_pos[1] - self.position[1]) / 600  # same for y
            dist = math.sqrt(dx**2 + dy**2)  # get the distance already normalized

            values = [dx / dist, dy/dist, 1 / (1 + dist)]
            input_values.extend(values)

        return input_values

    def output_layer(self, last_hidden_layer):
        # OUTPUT
        angle = (last_hidden_layer[0] + 1) / 2 * self.max_degrees  # convert normilizaed value to angle
        speed = (last_hidden_layer[1] + 1) / 2 * self.max_speed  # convert normalized value to speed

        return OutputValues(angle, speed)

    def set_player_position(self, output_values: OutputValues):
        # get the postion of the player based on angle and speed
        angle_rad = math.radians(output_values.angle)
        new_position = (
            self.position[0] + math.cos(angle_rad) * output_values.speed,
            self.position[1] + math.sin(angle_rad) * output_values.speed
        )

        self.position = new_position

    def fitness(self):
        p1 = self.position
        p2 = self.goals[0].position
        distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        fitness = 1 / (distance + 1)
        return fitness
