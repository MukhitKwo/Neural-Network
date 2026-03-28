from copy import deepcopy

import pygame
import sys
from player import Player
from fruit import Fruit
from player_population import PlayerPopulation
from neural_network_config import NeuralNetworkConfig

pygame.init()

# Window settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network")

clock = pygame.time.Clock()
FPS = 60

fruit = Fruit(screen, (700, 500))

config = NeuralNetworkConfig(
    hidden_layer_dimensions=[3, 2],  # 1st hidden layer with 3 neurons and 2nd hidden layer with 2 neurons
    mutation_rate=0.1,
    population_size=10,
    max_speed=5,
    max_degrees=360
)
player_population = PlayerPopulation(screen, (200, 200), [fruit], config)

ticks_per_generation = 240
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

    if frame >= ticks_per_generation:
        frame = 0
        generation += 1
        player_population.reproduce()
        print("Current generation:", generation)

    fruit.draw()
    player_population.forward()

    pygame.display.flip()

    clock.tick(FPS)
    frame += 1

pygame.quit()
sys.exit()
