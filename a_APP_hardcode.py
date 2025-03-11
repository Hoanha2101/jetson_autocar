
import pygame
import cv2
import time
import serial
import sys
from AI_brain_TRT_go_str import AI_TRT
from setting_AI import *
from utils_func_go_str import CLEAN_DATA_CSV_DIRECTION, ADD_DATA_CSV_MASK_DIRECTION, ADD_DATA_CSV_DIRECTION_STRAIGHT, CLEAN_DATA_CSV_DIRECTION_STRAIGHT, CHECK_PUSH, CLEAN_DATA_CSV_BACK_CONTROL

import json

data = json.loads("data.json")

serial_p = False
if serial_p:
    serial_port = serial.Serial("COM8", 9600, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)

# Initialize pygame
pygame.init()

# Screen settings
screen_width = 1536
screen_height = 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("AI Camera Control")

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Fonts
font = pygame.font.Font(None, 50)
small_font = pygame.font.Font(None, 36)  # Font nhỏ hơn để hiển thị kết quả push

# Buttons
start_button = pygame.Rect(100, 750, 200, 50)
end_button = pygame.Rect(400, 750, 200, 50)

# Initialize camera
cap = cv2.VideoCapture("videos/a.mp4")
if serial_p:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Wait for serial port to initialize
time.sleep(0.5)

time_stop = sys.maxsize
sleep_time = sys.maxsize
running = True
active = False  # Flag to track whether the AI processing is active
clear = True
push_results = []  # List lưu 5 kết quả push gần nhất

count_json = 0

def json_control(tuple_data):
    if serial_p: 
        serial_port.write(f"{tuple_data[0]}:000".encode())
    print(tuple_data)
    time.sleep(tuple_data[1])
    
while running:
    screen.fill(WHITE)
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

    _, frame = cap.read()
    
    data_item_json = json_data[f"state_{count_json}"]
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    visualization_img = frame

    if active:
        if clear:
            CLEAN_DATA_CSV_DIRECTION()
            CLEAN_DATA_CSV_DIRECTION_STRAIGHT()
            # CLEAN_DATA_CSV_BACK_CONTROL()
            clear = False
        for data_tuple in data_item_json:
            if data_tuple[0] != "S":
                json_control(data_tuple)
            else:
            
                visualization_img, PUSH_RETURN = AI_TRT(frame, paint=True, resize_img=True)
                
                if PUSH_RETURN:
                    serial_port.write(PUSH_RETURN.encode())
                    
                    push_results.append(PUSH_RETURN)  # Thêm kết quả mới vào danh sách
                    angle = int(PUSH_RETURN.split(":")[1])
                    angle = min(30, angle)
                    sleep_time = angle / ROTATION_SPEED
                    time.sleep(sleep_time)
                    push_results.append("x:000")
                    
                    if len(push_results) > 5:
                        push_results.pop(0) 


    pygame_frame = pygame.surfarray.make_surface(cv2.rotate(cv2.flip(visualization_img, 1), cv2.ROTATE_90_COUNTERCLOCKWISE))
    screen.blit(pygame_frame, (10, 10))

    pygame.draw.rect(screen, GREEN if active else BLACK, start_button)
    pygame.draw.rect(screen, RED, end_button)

    start_text = font.render("Start", True, WHITE)
    end_text = font.render("End", True, WHITE)
    screen.blit(start_text, (start_button.x + 60, start_button.y + 10))
    screen.blit(end_text, (end_button.x + 70, end_button.y + 10))

    # Hiển thị kết quả push gần nhất
    for i, result in enumerate(reversed(push_results)):
        result_text = small_font.render(result, True, BLACK)
        screen.blit(result_text, (1400, 600 - i * 40))  # Hiển thị từ dưới lên trên

    pygame.display.flip()

cap.release()
cv2.destroyAllWindows()
pygame.quit()