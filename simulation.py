import pygame
import sys
from goal.goal import GoalPopulation
from agent.agent_population import AgentPopulation
from utils import load_config

config = load_config()

pygame.init()

WIDTH = config["simulation"]["window_width"]
HEIGHT = config["simulation"]["window_height"]

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network")

clock = pygame.time.Clock()
FPS = 60

text_font = pygame.font.SysFont("Arial", 12)


def render_text(text, pos, center=False):
    img = text_font.render(text, True, (255, 255, 255))
    if center:
        rect = img.get_rect(center=pos)
        screen.blit(img, rect)
    else:
        screen.blit(img, pos)


goals = GoalPopulation(screen)
agents = AgentPopulation(screen, goals.population)

steps_per_generation = config["simulation"]["steps_per_generation"]
steps_this_generation: int = 0
generation: int = 0
previous_best_fitness = 0
speed_multiplier: int = 1
show_proximity_lines = False
show_fitness_value = False
paused = False

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
            if event.key == pygame.K_RIGHT:
                agents.forward(steps_per_generation)
                steps_this_generation += 1
            if event.key == pygame.K_l:
                show_proximity_lines = not show_proximity_lines
            if event.key == pygame.K_f:
                show_fitness_value = not show_fitness_value
            if event.key == pygame.K_SPACE:
                paused = not paused

    if not paused:
        for s in range(speed_multiplier):

            if steps_this_generation >= steps_per_generation:

                goals.randomize_position()

                previous_best_fitness = agents.reproduce()

                generation += 1
                steps_this_generation = 0

            steps_this_generation += 1
            fps = clock.get_fps()

            agents.forward(steps_this_generation)

    screen.fill((26, 26, 26))

    if show_proximity_lines:
        for agent in agents.population:
            if agent.closest_goal:
                pygame.draw.line(screen, (64, 64, 64), agent.position.xy, agent.closest_goal.position.xy)

    goals.draw()
    agents.draw()

    if show_fitness_value:
        for agent in agents.population:
            render_text(f"{agent.fitness:.4f}", agent.position.xy, center=True)

    if paused:
        render_text("PAUSED!", (20, HEIGHT - (8 * 14)))

    seconds_left = ((steps_per_generation / FPS) - (steps_this_generation / FPS)) / speed_multiplier

    render_text(f"Generation: {generation}", (20, HEIGHT - (6 * 14)))
    render_text(f"Steps: {steps_this_generation}/{steps_per_generation} ({seconds_left:.1f}s left)", (20, HEIGHT - (5 * 14)))
    render_text(f"Speed Multiplier: {speed_multiplier}", (20, HEIGHT - (4 * 14)))
    render_text(f"Previous Best Fitness: {previous_best_fitness}", (20, HEIGHT - (3 * 14)))
    render_text(f"FPS: {int(fps)}", (20, HEIGHT - (2 * 14)))


    pygame.display.flip()

    clock.tick(FPS)

pygame.quit()
sys.exit()
