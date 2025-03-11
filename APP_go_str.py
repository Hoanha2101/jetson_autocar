import pygame
import cv2
import time
import serial
import sys
from AI_brain_TRT_go_str import AI_TRT, Process_No_lane
from setting_AI import *
from utils_func_go_str import CLEAN_DATA_CSV_DIRECTION, CLEAN_DATA_CSV_DIRECTION_STRAIGHT

serial_p = False
if serial_p:
    serial_port = serial.Serial("COM8", 9600, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)
    
# Initialize camera
cap = cv2.VideoCapture("videos/g.mp4")
if serial_p:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

pygame.init()

# Screen settings
screen_width, screen_height = 1600, 900
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("AI Camera Control")

# Colors
WHITE = (240, 240, 240)
GREEN = (34, 177, 76)
RED = (200, 50, 50)
BLACK = (20, 20, 20)
GRAY = (180, 180, 180)
DARK_GRAY = (100, 100, 100)

# Fonts
font = pygame.font.Font(None, 50)
small_font = pygame.font.Font(None, 36)

# Buttons
start_button = pygame.Rect(100, 820, 220, 70)
end_button = pygame.Rect(400, 820, 220, 70)

# Slider settings
ROTATION_SPEED = 10
slider_rect = pygame.Rect(750, 850, 300, 10)
slider_knob_rect = pygame.Rect(750 + int((ROTATION_SPEED / 30) * 300) - 10, 840, 20, 30)
slider_dragging = False



time.sleep(1)

time_stop = sys.maxsize
sleep_time = sys.maxsize
running = True
active = False
clear = True
push_results = []

while running:
    screen.fill(WHITE)
    pygame.draw.rect(screen, DARK_GRAY, (0, 800, screen_width, 100))
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if start_button.collidepoint(event.pos):
                active = True
                clear = True
            elif end_button.collidepoint(event.pos):
                active = False
                print("Stopped pushing")
            elif slider_knob_rect.collidepoint(event.pos):
                slider_dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            slider_dragging = False
        elif event.type == pygame.MOUSEMOTION and slider_dragging:
            slider_knob_rect.x = max(slider_rect.x, min(event.pos[0] - 10, slider_rect.x + 300 - 20))
            ROTATION_SPEED = int(((slider_knob_rect.x - slider_rect.x) / 300) * 50)
    
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    visualization_img = frame
    
    
    
    if active:
        if clear:
            CLEAN_DATA_CSV_DIRECTION()
            CLEAN_DATA_CSV_DIRECTION_STRAIGHT()
            clear = False
        visualization_img, PUSH_RETURN, Have_lane = AI_TRT(frame, paint=True, resize_img=True)
        
        if Have_lane:
            if PUSH_RETURN:
                if serial_p:
                    serial_port.write(PUSH_RETURN.encode())
                push_results.append(PUSH_RETURN)
                if len(push_results) > 5:
                    push_results.pop(0)
                angle = min(30, int(PUSH_RETURN.split(":")[1]))
                sleep_time = angle / ROTATION_SPEED
                time_stop = time.time()
            
            if time.time() - time_stop >= sleep_time:
                if serial_p:
                    serial_port.write("x:000".encode())
                push_results.append("x:000")
                if len(push_results) > 5:
                    push_results.pop(0)
                time_stop = sys.maxsize
        else:
            
            visualization_img, Direction_mask = Process_No_lane(frame)
            
            
    
    pygame_frame = pygame.surfarray.make_surface(cv2.rotate(cv2.flip(visualization_img, 1), cv2.ROTATE_90_COUNTERCLOCKWISE))
    screen.blit(pygame_frame, (10, 10))

    # Buttons
    pygame.draw.rect(screen, GREEN if active else GRAY, start_button, border_radius=15)
    pygame.draw.rect(screen, RED, end_button, border_radius=15)
    screen.blit(font.render("Start", True, WHITE), (start_button.x + 70, start_button.y + 20))
    screen.blit(font.render("End", True, WHITE), (end_button.x + 80, end_button.y + 20))

    # Slider
    pygame.draw.rect(screen, GRAY, slider_rect, border_radius=5)
    pygame.draw.ellipse(screen, BLACK, slider_knob_rect)
    screen.blit(font.render(f"Speed: {ROTATION_SPEED}", True, WHITE), (1080, 820))

    # Display push results
    for i, result in enumerate(reversed(push_results)):
        screen.blit(small_font.render(result, True, BLACK), (1200, 600 - i * 40))
    
    pygame.display.flip()

cap.release()
cv2.destroyAllWindows()
pygame.quit()
