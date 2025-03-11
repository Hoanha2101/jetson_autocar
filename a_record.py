import numpy as np
import os
import pygame
import cv2

pygame.init()

folder_path = "collect_data" 


if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Định nghĩa kích thước màn hình
screen_width = 1000
screen_height = 600
camera_width = 640  # Mặc định 640
camera_height = 480  # Mặc định 480

# Hàm tìm tên file video mới không trùng lặp
def get_next_filename():
    index = 1
    while os.path.exists(f"{folder_path}/{index}.mp4"):
        index += 1
    return f"{folder_path}/{index}.mp4"

# Khởi tạo camera
# camera = cv2.VideoCapture(1)   # thầy thay đổi 
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)


camera.set(3, camera_width)
camera.set(4, camera_height)

# Font chữ
font = pygame.font.Font(None, 36)

# Khởi tạo màn hình pygame
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Camera App")

# Trạng thái quay video
recording = False
out = None
blink = False  # Hiệu ứng nhấp nháy khi quay
frame_count = 0  # Đếm số frame để tạo hiệu ứng nhấp nháy

# Hàm vẽ nút với hiệu ứng bấm
def draw_button(text, pos, color, active=False):
    rect = pygame.Rect(pos[0], pos[1], 150, 50)
    pygame.draw.rect(screen, color, rect, border_radius=10)
    
    if active:
        pygame.draw.rect(screen, (255, 255, 255), rect, 3, border_radius=10)  # Viền sáng khi đang quay
    
    text_surf = font.render(text, True, (255, 255, 255))
    screen.blit(text_surf, (pos[0] + 30, pos[1] + 10))
    
    return rect

running = True
while running:
    screen.fill((192, 192, 192))
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            if start_button.collidepoint(x, y) and not recording:
                filename = get_next_filename()
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(filename, fourcc, 25, (camera_width, camera_height))
                recording = True
            elif stop_button.collidepoint(x, y) and recording:
                recording = False
                out.release()
                out = None
    
    # Đọc hình ảnh từ camera
    ret, frame = camera.read()
    if ret:
        frame = cv2.flip(frame, 1)
        if recording:
            out.write(frame)
        
        frame = cv2.resize(frame, (350, 250))
        frame = pygame.surfarray.make_surface(cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE))
        screen.blit(frame, (10, 10))
    
    # Vẽ nút bấm với hiệu ứng
    start_button = draw_button("RECORD", (800, 100), (0, 200, 0), recording)
    stop_button = draw_button("STOP", (800, 200), (200, 0, 0))
    
    # Hiển thị trạng thái quay
    if recording:
        frame_count += 1
        if frame_count % 30 < 15:  # Hiệu ứng nhấp nháy
            blink = not blink
        
        status_color = (255, 0, 0) if blink else (200, 0, 0)
        pygame.draw.circle(screen, status_color, (screen_width - 50, 50), 15)
        status_text = font.render("Recording...", True, (255, 0, 0))
        screen.blit(status_text, (screen_width - 200, 40))
    
    pygame.display.flip()

# Giải phóng tài nguyên
if recording:
    out.release()
camera.release()
pygame.quit()
