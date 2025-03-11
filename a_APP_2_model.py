import pygame
import cv2
import time
import serial
import sys
from a_AI_brain_2_model import AI_TRT, INFER_CLASSIFICATION
from setting_AI import *
from a_utils_func_2_model import CLEAN_DATA_CSV_DIRECTION, CLEAN_DATA_CSV_DIRECTION_STRAIGHT
from a_control_classification import USE_CLASSIFICATION

serial_p = False
if serial_p:
    serial_port = serial.Serial("COM8", 9600, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)
    
# Initialize camera
cap = cv2.VideoCapture("videos/a.mp4")
cap_ = cv2.VideoCapture("videos/a.mp4")
if serial_p:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap_ = cv2.VideoCapture(1, cv2.CAP_DSHOW)

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
    start_time = time.time()
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
    _, frame_ = cap_.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    visualization_img = frame
    
    INFER_CLASSIFICATION(frame_)

    if active:
        if USE_CLASSIFICATION.check() == "STRAIGHT":
            if clear:
                CLEAN_DATA_CSV_DIRECTION()
                CLEAN_DATA_CSV_DIRECTION_STRAIGHT()
                clear = False
            visualization_img, PUSH_RETURN, Have_lane = AI_TRT(frame, paint=True, resize_img=True)
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
                    serial_port.write(PUSH_STOP.encode())
                push_results.append(PUSH_STOP)
                if len(push_results) > 5:
                    push_results.pop(0)
                time_stop = sys.maxsize

        else:
            
            hard_left = "X:000"
            hard_right = "Y:000"

            hard_time_1 = 0.5
            hard_time_2 = 1.5
            hard_time_3 = 0.5
            
            
            if USE_CLASSIFICATION.check() == "LEFT":
                serial_port.write(hard_right.encode())
                print(hard_right)
                time.sleep(0.5)
                serial_port.write(PUSH_STOP.encode())
                print(PUSH_STOP)
                
                
                serial_port.write(hard_left.encode())
                print(hard_left)
                time.sleep(1.5)
                serial_port.write(PUSH_STOP.encode())
                print(PUSH_STOP)
                
                
                serial_port.write(hard_right.encode())
                print(hard_right)
                time.sleep(0.5)
                serial_port.write(PUSH_STOP.encode())
                print(PUSH_STOP)
                
                USE_CLASSIFICATION.change("STRAIGHT")
                
            if USE_CLASSIFICATION.check() == "LEFT":
                serial_port.write(hard_left.encode())
                print(hard_left)
                time.sleep(0.5)
                serial_port.write(PUSH_STOP.encode())
                print(PUSH_STOP)
                
                serial_port.write(hard_right.encode())
                print(hard_right)
                time.sleep(1.5)
                serial_port.write(PUSH_STOP.encode())
                print(PUSH_STOP)
                
                serial_port.write(hard_left.encode())
                print(hard_left)
                time.sleep(0.5)
                serial_port.write(PUSH_STOP.encode())
                print(PUSH_STOP)
                
                USE_CLASSIFICATION.change("STRAIGHT")
            
    text_cls = font.render(USE_CLASSIFICATION.check(), True, (0, 0, 255)) 
    text_rect = text_cls.get_rect(center=(900, 200)) 
    screen.blit(text_cls, text_rect)
    
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    
    text_fps = font.render(f"FPS: {fps:.2f}", True, (0, 0, 255)) 
    text_rect = text_fps.get_rect(center=(900, 250)) 
    screen.blit(text_fps, text_rect)
    
    visualization_img = cv2.resize(visualization_img, (visualization_img.shape[1]//2, visualization_img.shape[0]//2))
    
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
