<<<<<<< HEAD
import pygame
import math

pygame.init()

# Set up the simulation field
L = 800
r = 60
field_size = (L, L)
screen = pygame.display.set_mode(field_size)
clock = pygame.time.Clock()

# Set up colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Set up agent and target positions
T1 = (int(L * 0.75), int(L * 0.75))
T2 = (int(L * 0.75), int(L * 0.25))
A1 = (int(L * 0.25), int(L * 0.25))
A2 = (int(L * 0.25), int(L * 0.75))

Va = 5

# Main simulation loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    # Draw circular repulsion field
    pygame.draw.circle(screen, RED, (L // 2, L // 2), 40)

    # Draw targets
    pygame.draw.circle(screen, GREEN, T1, 5)
    pygame.draw.circle(screen, GREEN, T2, 5)

    # Calculate movement directions for agents
    direction_A1 = math.atan2(T1[1] - A1[1], T1[0] - A1[0])
    direction_A2 = math.atan2(T2[1] - A2[1], T2[0] - A2[0])

    # Calculate new positions for agents
    if math.dist(A1, (L // 2, L // 2)) <= r:
        direction_A1 += math.pi / 2  # Rotate 90 degrees to avoid repulsion field
    if math.dist(A2, (L // 2, L // 2)) <= r:
        direction_A2 += math.pi / 2  # Rotate 90 degrees to avoid repulsion field

    A1 = (A1[0] + int(Va * math.cos(direction_A1)), A1[1] + int(Va * math.sin(direction_A1)))
    A2 = (A2[0] + int(Va * math.cos(direction_A2)), A2[1] + int(Va * math.sin(direction_A2)))

    # Draw agents as arrows with clear orientation
    pygame.draw.line(screen, YELLOW, A1, (A1[0] + int(10 * math.cos(direction_A1)), A1[1] + int(10 * math.sin(direction_A1))), 2)
    pygame.draw.polygon(screen, YELLOW, [(A1[0] + int(10 * math.cos(direction_A1)), A1[1] + int(10 * math.sin(direction_A1))),
                                          (A1[0] + int(20 * math.cos(direction_A1 + math.pi / 6)),
                                           A1[1] + int(20 * math.sin(direction_A1 + math.pi / 6))),
                                          (A1[0] + int(20 * math.cos(direction_A1 - math.pi / 6)),
                                           A1[1] + int(20 * math.sin(direction_A1 - math.pi / 6)))])
    
    pygame.draw.line(screen, YELLOW, A2, (A2[0] + int(10 * math.cos(direction_A2)), A2[1] + int(10 * math.sin(direction_A2))), 2)
    pygame.draw.polygon(screen, YELLOW, [(A2[0] + int(10 * math.cos(direction_A2)), A2[1] + int(10 * math.sin(direction_A2))),
                                          (A2[0] + int(20 * math.cos(direction_A2 + math.pi / 6)),
                                           A2[1] + int(20 * math.sin(direction_A2 + math.pi / 6))),
                                          (A2[0] + int(20 * math.cos(direction_A2 - math.pi / 6)),
                                           A2[1] + int(20 * math.sin(direction_A2 - math.pi / 6)))])

    # Check if agents reached their targets
    if A1 == T1:
        Va = 0
    if A2 == T2:
        Va = 0

    pygame.display.flip()
    clock.tick(60)

# Quit the simulation
=======
import pygame
import math

pygame.init()

# Set up the simulation field
L = 800
r = 60
field_size = (L, L)
screen = pygame.display.set_mode(field_size)
clock = pygame.time.Clock()

# Set up colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Set up agent and target positions
T1 = (int(L * 0.75), int(L * 0.75))
T2 = (int(L * 0.75), int(L * 0.25))
A1 = (int(L * 0.25), int(L * 0.25))
A2 = (int(L * 0.25), int(L * 0.75))

Va = 5

# Main simulation loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    # Draw circular repulsion field
    pygame.draw.circle(screen, RED, (L // 2, L // 2), 40)

    # Draw targets
    pygame.draw.circle(screen, GREEN, T1, 5)
    pygame.draw.circle(screen, GREEN, T2, 5)

    # Calculate movement directions for agents
    direction_A1 = math.atan2(T1[1] - A1[1], T1[0] - A1[0])
    direction_A2 = math.atan2(T2[1] - A2[1], T2[0] - A2[0])

    # Calculate new positions for agents
    if math.dist(A1, (L // 2, L // 2)) <= r:
        direction_A1 += math.pi / 2  # Rotate 90 degrees to avoid repulsion field
    if math.dist(A2, (L // 2, L // 2)) <= r:
        direction_A2 += math.pi / 2  # Rotate 90 degrees to avoid repulsion field

    A1 = (A1[0] + int(Va * math.cos(direction_A1)), A1[1] + int(Va * math.sin(direction_A1)))
    A2 = (A2[0] + int(Va * math.cos(direction_A2)), A2[1] + int(Va * math.sin(direction_A2)))

    # Draw agents as arrows with clear orientation
    pygame.draw.line(screen, YELLOW, A1, (A1[0] + int(10 * math.cos(direction_A1)), A1[1] + int(10 * math.sin(direction_A1))), 2)
    pygame.draw.polygon(screen, YELLOW, [(A1[0] + int(10 * math.cos(direction_A1)), A1[1] + int(10 * math.sin(direction_A1))),
                                          (A1[0] + int(20 * math.cos(direction_A1 + math.pi / 6)),
                                           A1[1] + int(20 * math.sin(direction_A1 + math.pi / 6))),
                                          (A1[0] + int(20 * math.cos(direction_A1 - math.pi / 6)),
                                           A1[1] + int(20 * math.sin(direction_A1 - math.pi / 6)))])
    
    pygame.draw.line(screen, YELLOW, A2, (A2[0] + int(10 * math.cos(direction_A2)), A2[1] + int(10 * math.sin(direction_A2))), 2)
    pygame.draw.polygon(screen, YELLOW, [(A2[0] + int(10 * math.cos(direction_A2)), A2[1] + int(10 * math.sin(direction_A2))),
                                          (A2[0] + int(20 * math.cos(direction_A2 + math.pi / 6)),
                                           A2[1] + int(20 * math.sin(direction_A2 + math.pi / 6))),
                                          (A2[0] + int(20 * math.cos(direction_A2 - math.pi / 6)),
                                           A2[1] + int(20 * math.sin(direction_A2 - math.pi / 6)))])

    # Check if agents reached their targets
    if A1 == T1:
        Va = 0
    if A2 == T2:
        Va = 0

    pygame.display.flip()
    clock.tick(60)

# Quit the simulation
>>>>>>> 6f17868cc3d8c2f6228cb2a444cf1237d2f75967
pygame.quit()