import copy

import pygame
from neural_network import NeuralNetwork
import random
import math


class OutputValues:
    def __init__(self, angle, speed):
        self.angle = angle
        self.speed = speed


class Agent(NeuralNetwork):
    def __init__(self, screen, position, goals, config, generate_hidden_layers=True):
        self.screen = screen
        super().__init__(config.neuralNetwork, generate_hidden_layers)
        self.agent_config = config.agent
        self.position = position
        self.remaining_goals = goals.copy()
        self.closest_goal = self.get_closest_goal()
        self.fitness = 0
        self.color = (0, 0, random.randint(155, 255))
        self.radius = 25

    def draw(self):
        pygame.draw.circle(self.screen, self.color, self.position, self.radius)

    def update(self, frame):
        if self.closest_goal is None:
            return

        input_vector = self.set_input_vector()
        last_hidden_vector = self.forward(input_vector)
        output_vector = self.get_output_vector(last_hidden_vector)
        self.set_player_position(output_vector)
        self.did_collide_with_goal(frame)
        self.check_proximity_to_goal()

    def set_input_vector(self):
        goal_pos = self.closest_goal.position
        dx = (goal_pos[0] - self.position[0]) / 800  # get x distance and normalize it to -1 and 1
        dy = (goal_pos[1] - self.position[1]) / 600  # same for y
        dist = math.sqrt(dx**2 + dy**2)  # get the distance already normalized due to dx and dy

        inputs = [dx / dist, dy/dist, 1 / (1 + dist)]

        return inputs

    def get_output_vector(self, last_hidden_layer):
        angle = (last_hidden_layer[0] + 1) / 2 * self.agent_config.max_degrees  # convert normilizaed value to angle
        speed = (last_hidden_layer[1] + 1) / 2 * self.agent_config.max_speed  # convert normalized value to speed

        return OutputValues(angle, speed)

    def set_player_position(self, output_values: OutputValues):
        # get the postion of the player based on angle and speed
        angle_rad = math.radians(output_values.angle)
        new_position = (
            self.position[0] + math.cos(angle_rad) * output_values.speed,
            self.position[1] + math.sin(angle_rad) * output_values.speed
        )

        self.position = new_position

    def get_closest_goal(self):
        if len(self.remaining_goals) == 0:
            return None

        closest_distance = None
        closest_goal = None

        for goal in self.remaining_goals:
            goal_pos = goal.position
            raw_dx = (goal_pos[0] - self.position[0])
            raw_dy = (goal_pos[1] - self.position[1])
            raw_dist = math.sqrt(raw_dx**2 + raw_dy**2)

            if closest_distance is None or raw_dist < closest_distance:
                closest_distance = raw_dist
                closest_goal = goal

        return closest_goal

    def did_collide_with_goal(self, step):
        for goal in self.remaining_goals:
            p1 = self.position
            p2 = goal.position
            distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

            if distance < self.radius:
                T = 360
                self.fitness += self.agent_config.fitness.goals_reached_multiplier + ((T - step)/360) * self.agent_config.fitness.time_bonus_multiplier
                self.remaining_goals.remove(goal)
                self.closest_goal = self.get_closest_goal()

    def check_proximity_to_goal(self):
        if self.closest_goal is None:
            return

        p1 = self.position
        p2 = self.closest_goal.position
        distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        self.fitness += (1 / (distance + 1e-3)) * self.agent_config.fitness.closeness_multiplier

    def inherit_hidden_layers(self, best_player_hidden_layer_parameters):
        self.hidden_layers = copy.deepcopy(best_player_hidden_layer_parameters)
