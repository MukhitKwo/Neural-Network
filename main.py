import pygame
import sys
from player import Player
from fruit import Fruit

pygame.init()

# Window settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network")

clock = pygame.time.Clock()
FPS = 60

fruit = Fruit(screen, (WIDTH // 2, HEIGHT // 4))

players = []
for i in range(5):
    players.append(Player(screen, (WIDTH // 2, HEIGHT // 2), fruit.position))


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

    fruit.draw()

    pygame.display.flip()       # Push frame to display
    clock.tick(FPS)             # Cap to 60 FPS

pygame.quit()
sys.exit()
