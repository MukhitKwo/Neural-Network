import pygame
from agent.neural_network import NeuralNetwork
from configs import load_config
import random
import math
import copy
from utils import AccelerationValues, OutputValues, Position

config = load_config()


class Agent(NeuralNetwork):
    def __init__(self, screen, position: Position, goals, generate_hidden_layers=True):
        self.screen = screen
        super().__init__(generate_hidden_layers)
        self.position = position
        self.remaining_goals = goals.copy()
        self.closest_goal = self.get_closest_goal()
        self.vx = 0
        self.vy = 0
        self.fitness = 0
        self.color = (0, 0, random.randint(155, 255))
        self.radius = 25

    def draw(self):
        pygame.draw.circle(self.screen, self.color, self.position.xy, self.radius)

    def update(self, frame):
        if self.closest_goal is None:
            return

        input_vector = self.get_input_vector()
        last_hidden_vector = self.forward(input_vector)
        output_vector = self.get_output_vector(last_hidden_vector)
        self.set_player_position(output_vector)
        self.did_collide_with_goal(frame)
        self.check_proximity_to_goal()

    def get_input_vector(self):
        goal_pos = self.closest_goal.position
        dx = (goal_pos.x - self.position.x) / 800  # get x distance and normalize it to -1 and 1
        dy = (goal_pos.y - self.position.y) / 600  # same for y
        dist = math.sqrt(dx**2 + dy**2)  # get the distance already normalized due to dx and dy

        inputs = [dx / dist, dy/dist, 1 / (1 + dist), self.vx, self.vy]

        return inputs

    def get_output_vector(self, last_hidden_layer):
        
        ax = last_hidden_layer[0] * config["agent"]["max_acceleration"]
        ay = last_hidden_layer[1] * config["agent"]["max_acceleration"]
        
        return AccelerationValues(ax, ay)

    def set_player_position(self, output_values: AccelerationValues):
        self.vx += output_values.ax
        self.vy += output_values.ay
        
        new_position = Position(
            self.position.x + self.vx,
            self.position.y + self.vy
        )

        self.position = new_position

    def get_distance(self, pos_1, pos_2):
        return math.sqrt((pos_2.x - pos_1.x)**2 + (pos_2.y - pos_1.y)**2)

    def get_closest_goal(self):
        if len(self.remaining_goals) == 0:
            return None

        return min(self.remaining_goals, key=(lambda goal: self.get_distance(self.position, goal.position)))

    def did_collide_with_goal(self, step):
        for goal in self.remaining_goals:
            distance = self.get_distance(self.position, goal.position)

            if distance < self.radius:
                T = config["simulation"]["steps_per_generation"]
                self.fitness += config["fitness"]["goals_reached_multiplier"] + ((T - step)/T) * config["fitness"]["time_bonus_multiplier"]
                self.remaining_goals.remove(goal)
                self.closest_goal = self.get_closest_goal()

        if self.closest_goal is None:
            self.color = (0, random.randint(100, 155), 0)

    def check_proximity_to_goal(self):
        if self.closest_goal is None:
            return

        distance = self.get_distance(self.position, self.closest_goal.position)

        self.fitness += (1 / (distance + 1e-3)) * config["fitness"]["proximity_multiplier"]

    def inherit_hidden_layers(self, best_player_hidden_layer_parameters):
        self.hidden_layers = copy.deepcopy(best_player_hidden_layer_parameters)
