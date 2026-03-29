from copy import deepcopy

import pygame
import sys
from player import Player
from fruit import Fruit
from player_population import PlayerPopulation
from neural_network_config import NeuralNetworkConfig

pygame.init()

# Window settings
WIDTH, HEIGHT = 1200, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network")

clock = pygame.time.Clock()
FPS = 240

fruit = Fruit(screen, (900, 200))

config = NeuralNetworkConfig(
    hidden_layer_dimensions=[3, 6, 4, 2],  # 1st hidden layer has 3 inputs, 2nd has 4 inputs and 3rd has 2
    mutation_rate=0.02,
    population_size=10,
    max_speed=5,
    max_degrees=360
)
player_population = PlayerPopulation(screen, (600, 450), [fruit], config)

frames_per_generation = 240
frame = 0
generation = 0

running = True
while running:

    screen.fill((26, 26, 26))

    # 1. Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    if frame >= frames_per_generation:
        frame = 0
        generation += 1
        player_population.reproduce()
        print("Current generation:", generation)

    fruit.draw()
    player_population.forward(frame)

    pygame.display.flip()

    clock.tick(FPS)
    frame += 1

    fps = clock.get_fps()
    if frame % 60 == 0:
        print(f"FPS: {fps:.2f}")

pygame.quit()
sys.exit()
