from copy import deepcopy

import pygame
import sys
from networks.agent import Agent
from fruit import Fruit
from player_population import AgentPopulation
from neural_network_config import NeuralNetworkConfig

pygame.init()

# Window settings
WIDTH, HEIGHT = 1200, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network")

clock = pygame.time.Clock()
FPS = 60

fruit1 = Fruit(screen, (900, 200))
fruit2 = Fruit(screen, (300, 400))

config = NeuralNetworkConfig(
    number_inputs=3,
    hidden_layer_dimensions=[3, 6, 4, 2],
    mutation_rate=0.03,
    population_size=10,
    max_speed=5,
    max_degrees=360
)
player_population = AgentPopulation(screen, (600, 450), [fruit1, fruit2], config)

max_seconds_per_generation = 5
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

    if frame >= (max_seconds_per_generation * FPS):
        frame = 0
        generation += 1
        player_population.reproduce()
        print("Current generation:", generation)

    fruit1.draw()
    fruit2.draw()
    
    player_population.forward(frame)
    

    pygame.display.flip()

    clock.tick(FPS)
    frame += 1

    fps = clock.get_fps()
    if frame % 60 == 0:
        print(f"FPS: {fps:.2f}")

pygame.quit()
sys.exit()
