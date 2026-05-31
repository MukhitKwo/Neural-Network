import pygame
from agent.neural_network import NeuralNetwork
import math
import copy
from utils import AccelerationValues, Position, load_config

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
        self.color = (0, 0, 180)
        self.radius = 25

    def draw(self):
        pygame.draw.circle(self.screen, (90, 0, 90), self.position.xy, self.radius + 1)
        pygame.draw.circle(self.screen, self.color, self.position.xy, self.radius)

    def update_color(self):
        start_color = (0, 0, 180)
        end_color = (180, 0, 0)

        factor = 1 - (1 / (config["population"]["goal_population_size"] / (len(self.remaining_goals) + 1e-5)))

        r = int(start_color[0] + (end_color[0] - start_color[0]) * factor)
        g = int(start_color[1] + (end_color[1] - start_color[1]) * factor)
        b = int(start_color[2] + (end_color[2] - start_color[2]) * factor)

        self.color = (r, g, b)

    def update(self, frame):
        if self.closest_goal is None:
            return

        input_vector = self.get_input_vector()
        last_hidden_vector = self.forward(input_vector)
        output_vector = self.get_output_vector(last_hidden_vector)
        self.set_agent_position(output_vector)
        self.did_collide_with_goal(frame)
        self.check_proximity_to_goal()

    def get_input_vector(self):
        goal_pos = self.closest_goal.position
        dx = (goal_pos.x - self.position.x) / config["simulation"]["window_width"]
        dy = (goal_pos.y - self.position.y) / config["simulation"]["window_height"]
        dist = math.sqrt(dx**2 + dy**2)

        max_v = config["agent"]["max_velocity"]

        inputs = [dx / dist, dy/dist, 1 / (1 + dist), self.vx / max_v, self.vy / max_v]

        return inputs

    def get_output_vector(self, last_hidden_layer):

        ax = last_hidden_layer[0] * config["agent"]["max_acceleration"]
        ay = last_hidden_layer[1] * config["agent"]["max_acceleration"]

        return AccelerationValues(ax, ay)

    def set_agent_position(self, output_values: AccelerationValues):

        max_v = config["agent"]["max_velocity"]

        self.vx = min(self.vx + output_values.ax, max_v) if output_values.ax > 0 else max(self.vx + output_values.ax, -max_v)
        self.vy = min(self.vy + output_values.ay, max_v) if output_values.ay > 0 else max(self.vy + output_values.ay, -max_v)

        new_position = Position(
            self.position.x + self.vx,
            self.position.y + self.vy
        )

        self.position = new_position

    def get_true_distance(self, pos_1, pos_2):
        return math.sqrt((pos_2.x - pos_1.x)**2 + (pos_2.y - pos_1.y)**2)

    def get_squared_distances(self, pos_1, pos_2):
        return (pos_2.x - pos_1.x)**2 + (pos_2.y - pos_1.y)**2

    def get_closest_goal(self):
        if len(self.remaining_goals) == 0:
            return None

        return min(self.remaining_goals, key=(lambda goal: self.get_squared_distances(self.position, goal.position)))

    def did_collide_with_goal(self, step):
        for goal in self.remaining_goals:
            distance = self.get_true_distance(self.position, goal.position)

            if distance < self.radius:
                S = config["simulation"]["steps_per_generation"]
                self.fitness += config["fitness"]["goals_reached_bonus"] + ((S - step)/S) * config["fitness"]["time_multiplier"]
                self.remaining_goals.remove(goal)
                self.closest_goal = self.get_closest_goal()
                self.update_color()

    def check_proximity_to_goal(self):
        if self.closest_goal is None:
            return

        distance = self.get_true_distance(self.position, self.closest_goal.position)

        self.fitness += (1 / (distance + 1e-3)) * config["fitness"]["proximity_multiplier"]

    def set_hidden_layers(self, best_player_hidden_layer_parameters):
        self.hidden_layers = copy.deepcopy(best_player_hidden_layer_parameters)
