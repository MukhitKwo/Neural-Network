import pygame
import sys

# Initialize Pygame
pygame.init()

# Window settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("My Game")

# Clock for controlling FPS
clock = pygame.time.Clock()
FPS = 60

# Colors (R, G, B)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# --- Game loop ---
running = True
while running:

    # 1. Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # 2. Update game state (logic goes here)

    # 3. Draw everything
    screen.fill(BLACK)          # Clear screen
    # ... draw your stuff here ...

    pygame.draw.circle(screen, (0, 0, 255), (WIDTH // 2, HEIGHT // 2), 25)
    pygame.draw.circle(screen, (255, 0, 0), (WIDTH // 2, 60), 25)

    pygame.display.flip()       # Push frame to display

    clock.tick(FPS)             # Cap to 60 FPS

pygame.quit()
sys.exit()
