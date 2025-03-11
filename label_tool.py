import cv2
import os
import tkinter as tk
from PIL import Image, ImageTk

root_path = "data_label"

# Create folders for labeled images
os.makedirs(f"{root_path}/left", exist_ok=True)
os.makedirs(f"{root_path}/right", exist_ok=True)
os.makedirs(f"{root_path}/straight", exist_ok=True)

# Load video
video_path = "collect_data/1.mp4"  # Change this to your video file
cap = cv2.VideoCapture(video_path)

frame_count = 0
current_frame = None
highlight_label = None  # Biến để lưu nhãn đang được chọn

# Read first frame
def read_frame():
    global current_frame, frame_count
    ret, frame = cap.read()
    if ret:
        current_frame = frame
        frame_count += 1
        show_frame()
    else:
        print("Video ended.")
        cap.release()
        root.quit()

# Save frame to respective folder
def save_frame(label):
    global highlight_label
    if current_frame is not None:
        frame_resized = cv2.resize(current_frame, (640, 480))
        filename = f"{label}/{frame_count}.jpg"
        cv2.imwrite(filename, frame_resized)
        highlight_label = label  # Cập nhật nhãn đang chọn
        update_label_highlight()
        read_frame()

# Xử lý sự kiện nhấn phím
def key_press(event):
    if event.char == '1':
        save_frame("left")
    elif event.char == '2':
        save_frame("right")
    elif event.char == '3':
        save_frame("straight")

# Hiển thị hình ảnh từ video lên GUI
def show_frame():
    frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (640, 480))
    img = Image.fromarray(frame_resized)
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img

# Cập nhật giao diện khi nhấn nút
def update_label_highlight():
    global highlight_label
    if highlight_label == "left":
        label_status.config(text="Selected: LEFT", bg="red")
    elif highlight_label == "right":
        label_status.config(text="Selected: RIGHT", bg="blue")
    elif highlight_label == "straight":
        label_status.config(text="Selected: STRAIGHT", bg="green")

    # Tạo hiệu ứng biến mất sau 500ms
    root.after(500, reset_label_highlight)

# Reset màu nền sau khi chọn label
def reset_label_highlight():
    label_status.config(text="Press 1, 2, 3 to label", bg="white")

# Setup GUI
root = tk.Tk()
root.title("Autonomous Car Label Tool")

# Set window size to match frame size
root.geometry("640x580")  # Tăng chiều cao để thêm thông báo trạng thái

# Add instructions label
instructions = tk.Label(root, text="Press: 1 for Left | 2 for Right | 3 for Straight", font=("Arial", 12))
instructions.pack()

# Add label status indicator
label_status = tk.Label(root, text="Press 1, 2, 3 to label", font=("Arial", 14, "bold"), bg="white", width=30)
label_status.pack(pady=5)

# Panel để hiển thị hình ảnh
panel = tk.Label(root)
panel.pack()

# Bind keyboard events
root.bind('<Key>', key_press)

# Start processing
read_frame()
root.mainloop()
