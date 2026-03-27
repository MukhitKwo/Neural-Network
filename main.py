from copy import deepcopy

import pygame
import sys
from neural_network import NeuralNetwork
from player import Player
from fruit import Fruit

pygame.init()

# Window settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network")

clock = pygame.time.Clock()
FPS = 60

fruit = Fruit(screen, (700, 500))

players = []
for i in range(5):
    players.append(Player(screen, (200, 200), fruit.position))

frame = 0

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

    for player in players:
        player.forward()
        player.draw()

    if frame >= 100:
        frame = 0
        best_player = None

        for player in players:

            print(player.fitness())
            if player.fitness() > (best_player.fitness() if best_player else 0):
                best_player = player

        best_player.player_pos = (200, 200)

        for player in players:

            player.player_pos = best_player.player_pos[:]
            player.parameters = deepcopy(best_player.parameters)

        for player in players[1:]:
            player.mutate(0.1)

    fruit.draw()

    pygame.display.flip()       # Push frame to display

    clock.tick(FPS)             # Cap to 60 FPS
    frame += 1
    print(frame)

pygame.quit()
sys.exit()
