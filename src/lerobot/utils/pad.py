import pygame
import sys

pygame.init()

# Set up display
WIDTH, HEIGHT = 600, 400
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gamepad Button Mapping")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)

# Initialize joystick
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    print("No joystick/gamepad detected.")
    pygame.quit()
    sys.exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()
num_buttons = joystick.get_numbuttons()

font = pygame.font.SysFont(None, 32)

def draw_mapping():
    window.fill(BLACK)
    text = font.render("Gamepad Button Mapping", True, WHITE)
    window.blit(text, (20, 20))

    for i in range(num_buttons):
        btn_state = joystick.get_button(i)
        btn_color = GREEN if btn_state else RED
        btn_label = f"Button {i}: {'Pressed' if btn_state else 'Released'}"
        btn_text = font.render(btn_label, True, btn_color)
        window.blit(btn_text, (50, 60 + 30 * i))

    pygame.display.flip()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    draw_mapping()
    pygame.time.wait(50) # adjust as needed for responsiveness

pygame.quit()
