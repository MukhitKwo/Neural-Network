import random

import pygame
import sys
from fruit import Fruit
from agent_population import AgentPopulation
from neural_network_config import NeuralNetworkConfig

pygame.init()

WIDTH, HEIGHT = 1200, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network")

clock = pygame.time.Clock()
FPS = 60

# todo: convert to population class
fruits = [Fruit(screen, (random.randint(100, 1100), random.randint(100, 800))) for _ in range(3)]

config = NeuralNetworkConfig(
    number_inputs=3,
    hidden_layer_dimensions=[3, 6, 4, 2],
    max_speed=5,
    max_degrees=360,
    mutation_rate=0.03,
    population_size=10
)
agent_population = AgentPopulation(screen, (600, 450), fruits, config)

forwards_per_generation = 5 * FPS
forwards_this_generation = 0
generation = 0
# speed_multiplier: int = 1 # todo: use speed multiplier

running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    if forwards_this_generation >= forwards_per_generation:
        for fruit in fruits:
            fruit.position = (random.randint(100, 1100), random.randint(100, 800))

        agent_population.reproduce()

        generation += 1
        forwards_this_generation = 0
        print("Current generation:", generation)

    screen.fill((26, 26, 26))

    for fruit in fruits:
        fruit.draw()

    agent_population.forward(forwards_this_generation)
    forwards_this_generation += 1

    pygame.display.flip()

    clock.tick(FPS)
    fps = clock.get_fps()
    if forwards_this_generation % 60 == 0:
        print(f"FPS: {fps:.2f}")

pygame.quit()
sys.exit()
