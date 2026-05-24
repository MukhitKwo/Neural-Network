import pygame
import sys
from goal.goal import GoalPopulation
from agent.agent_population import AgentPopulation
from configs import load_config

config = load_config()

pygame.init()

WIDTH = config["simulation"]["window_width"]
HEIGHT = config["simulation"]["window_height"]

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network")

clock = pygame.time.Clock()
FPS = 60


def draw_text(text, font_size, pos):
    text_font = pygame.font.SysFont("Arial", font_size)
    img = text_font.render(text, True, (255, 255, 255))
    screen.blit(img, pos)


goals = GoalPopulation(screen)
agents = AgentPopulation(screen, goals.population)

steps_per_generation = config["simulation"]["steps_per_generation"]
steps_this_generation: int = 0
generation: int = 0
speed_multiplier: int = 1
previous_best_fitness = 0

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

        if steps_this_generation >= steps_per_generation:

            previous_best_fitness = agents.reproduce()

            goals.randomize_position()

            generation += 1
            steps_this_generation = 0

        screen.fill((26, 26, 26))

        goals.draw()

        agents.forward(steps_this_generation)

        steps_this_generation += 1

        fps = clock.get_fps()

    draw_text(f"Generation: {generation}", 12, (20, HEIGHT - (6 * 14)))
    draw_text(f"Steps: {steps_this_generation}/{steps_per_generation}", 12, (20, HEIGHT - (5 * 14)))
    draw_text(f"Speed Multiplier: {speed_multiplier}", 12, (20, HEIGHT - (4 * 14)))
    draw_text(f"Previous Best Fitness: {previous_best_fitness}", 12, (20, HEIGHT - (3 * 14)))
    draw_text(f"FPS: {int(fps)}", 12, (20, HEIGHT - (2 * 14)))

    pygame.display.flip()

    clock.tick(FPS)

pygame.quit()
sys.exit()
