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
fruits = [Fruit(screen, (random.randint(100, 1100), random.randint(100, 800))) for _ in range(20)]

config = NeuralNetworkConfig(
    number_inputs=3,
    hidden_layer_dimensions=[8, 6, 4],
    mutation_rate=0.02,
    mutation_prob=0.2,
    max_speed=5,
    max_degrees=360
)

agent_population = AgentPopulation(screen, (600, 450), fruits, 100, config)

forwards_per_generation: int = 6 * FPS
forwards_this_generation: int = 0
generation: int = 0
speed_multiplier: int = 1

running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_UP:
                speed_multiplier += 1 if speed_multiplier < 10 else 0
            if event.key == pygame.K_DOWN:
                speed_multiplier -= 1 if speed_multiplier > 1 else 0

    for s in range(speed_multiplier):

        if forwards_this_generation >= forwards_per_generation:
            for fruit in fruits:
                fruit.position = (random.randint(100, 1100), random.randint(100, 800))

            agent_population.reproduce(screen)

            generation += 1
            forwards_this_generation = 0
            print(f"{31 * '='}")
            print("Generation:", generation)
            print("Speed Mult:", speed_multiplier)

        screen.fill((26, 26, 26))

        for fruit in fruits:
            fruit.draw()

        agent_population.forward(forwards_this_generation)

        forwards_this_generation += 1

    pygame.display.flip()

    clock.tick(FPS)
    fps = clock.get_fps()
    if forwards_this_generation % 10 == 0:
        print(f"FPS: {fps:.2f}", end="\r")

pygame.quit()
sys.exit()
