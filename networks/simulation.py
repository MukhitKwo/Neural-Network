import random

import pygame
import sys
from goal import Goal
from agent_population import AgentPopulation
from configs import Position, SimulationConfig

pygame.init()

WIDTH, HEIGHT = 1200, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network")

clock = pygame.time.Clock()
FPS = 60

config = SimulationConfig()

# todo: convert to population class
goals = [Goal(screen, Position(random.randint(100, 1100), random.randint(100, 800))) for _ in range(20)]
agents = AgentPopulation(screen, goals, 100, Position(600, 450), config)

forwards_per_generation: int = 6 * FPS
steps_this_generation: int = 0
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

        if steps_this_generation >= forwards_per_generation:
            for fruit in goals:
                fruit.position = Position(random.randint(100, 1100), random.randint(100, 800))

            agents.reproduce(screen)

            generation += 1
            steps_this_generation = 0

            print(f"{31 * '='}")
            print("Generation:", generation)
            print("Speed Mult:", speed_multiplier)

        screen.fill((26, 26, 26))

        for fruit in goals:
            fruit.draw()

        agents.forward(steps_this_generation)

        steps_this_generation += 1

        fps = clock.get_fps()
        print(f"Forwards: {steps_this_generation}/{forwards_per_generation} | FPS: {fps:.2f}", end="\r")

    pygame.display.flip()

    clock.tick(FPS)

pygame.quit()
sys.exit()
