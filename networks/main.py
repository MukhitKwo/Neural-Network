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

fruit1 = Fruit(screen, (700, 500))
fruit2 = Fruit(screen, (600, 100))

config = NeuralNetworkConfig(
    hidden_layer_dimensions=[3, 4, 2],  # 1st has 3 inputs, 2nd has 4 inputs and 3rd has 2
    mutation_rate=0.05,
    population_size=10,
    max_speed=5,
    max_degrees=360
)
player_population = PlayerPopulation(screen, (200, 300), [fruit2], config)

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

    fruit1.draw()
    fruit2.draw()
    player_population.forward()

    pygame.display.flip()

    clock.tick(FPS)
    frame += 1

pygame.quit()
sys.exit()
