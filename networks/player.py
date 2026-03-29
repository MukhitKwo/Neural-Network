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
        super().__init__(goals, config)
        self.position = position
        self.max_degrees = config.max_degrees
        self.max_speed = config.max_speed
        self.color = (0, 0, random.randint(200, 255))
        self.radius = 25
        self.fitness = 0
        self.collided = False

    def draw(self):
        pygame.draw.circle(self.screen, self.color, self.position, self.radius)

    def update(self, frame):
        input_layer = self.input_layer()
        last_hidden_layer = self.forward(input_layer)
        output_layer = self.output_layer(last_hidden_layer)
        self.set_player_position(output_layer)
        self.did_collide_with_fruit(frame)

    def input_layer(self):

        closest_goal = None

        for goal in self.goals:
            goal_pos = goal.position
            dx = (goal_pos[0] - self.position[0]) / 800  # get x distance and normalize it to -1 and 1
            dy = (goal_pos[1] - self.position[1]) / 600  # same for y
            dist = math.sqrt(dx**2 + dy**2)  # get the distance already normalized

            if closest_goal is None or dist < closest_goal:
                closest_goal = goal
                values = [dx / dist, dy/dist, 1 / (1 + dist)]

        return values

    def output_layer(self, last_hidden_layer):
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

    def did_collide_with_fruit(self, frame):
        for goal in self.goals:
            p1 = self.position
            p2 = goal.position
            distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

            if distance < self.radius and not self.collided:
                self.fitness += (1/(frame+1) + 1)
                self.collided = True

    def distance_from_fruit(self):
        p1 = self.position
        p2 = self.goals[0].position  # todo update to be dynamic
        distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        self.fitness += 1 / (distance + 1)
