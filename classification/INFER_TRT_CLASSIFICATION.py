import torch
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from __init__ import TensorrtBase
import cv2
import time


plan = "classification/model/classification_FP16.trt"
input_names = ['input']
output_names = ['output']
batch = 1

net = TensorrtBase(plan,
                   input_names=input_names,
                   output_names=output_names,
                   max_batch_size=batch)

classes = ['LEFT', 'RIGHT', 'STRAIGHT']

def infer_with_trt(image):
    images = np.ascontiguousarray(image).astype(np.float16)
    net.cuda_ctx.push()
    inputs, outputs, bindings, stream = net.buffers
    
    # Set optimization profile and input shape
    net.context.set_optimization_profile_async(0, stream.handle)
    net.context.set_input_shape(input_names[0], images.shape)
    
    # Transfer input data to the GPU
    cuda.memcpy_htod_async(inputs[0].device, images, stream)
    
    # Execute inference
    net.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)      
    
    # Transfer predictions back to the host
    cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream)
    stream.synchronize()
    
    # Copy outputs
    trt_outputs = [out.host.copy() for out in outputs]
    net.cuda_ctx.pop()
    
    return trt_outputs[0]

def pre_processing(image):
    img = cv2.resize(image, (224, 224)).astype(np.float16)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    img = (img / 255.0 - mean) / std
    img = np.transpose(img, (2, 0, 1))  # Convert to CHW format
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def softmax(x):
    """Apply softmax function to numpy array."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()


def run(frame):
    global classes
    
    # height = frame.shape[0]
    # width = frame.shape[1]
    # frame = frame[height // 2:, :]
    
    processed_frame = pre_processing(frame)
    output = infer_with_trt(processed_frame)
    probabilities = softmax(output)
    predicted_class = classes[np.argmax(probabilities)]
    confidence = np.max(probabilities)
    
    return predicted_class, confidence



# cap = cv2.VideoCapture("videos/c.mp4")
# while True:
#     start_time = time.time()
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     height = frame.shape[0]
#     width = frame.shape[1]
#     frame = frame[height // 2:, :]
    
#     # Preprocess frame
#     processed_frame = pre_processing(frame)
    
#     # Perform inference
#     output = infer_with_trt(processed_frame)
    
#     # Apply softmax
#     probabilities = softmax(output)
    
#     # Get predicted class
#     predicted_class = classes[np.argmax(probabilities)]
#     confidence = np.max(probabilities)
    
#     # Hiển thị FPS lên hình
#     elapsed_time = time.time() - start_time
#     fps = 1 / elapsed_time if elapsed_time > 0 else 0
    
#     cv2.line(frame, (0, height // 2), (width, height // 2), (255,255,0), 2)
    
#     # Display result on frame
#     cv2.putText(frame, f"{predicted_class}: {confidence:.2f}", (50, 50), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.putText(frame, f"FPS: {fps:.2f}", (50, 90), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
#     cv2.imshow("Inference", frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
