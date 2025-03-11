import onnxruntime as ort
import cv2
import numpy as np
from numpy.typing import NDArray
import sys
sys.path.append("../")


def load_model(model_path: str):
    """Load ONNX model for inference."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session


def prepare_input(image: NDArray[np.uint8]) -> NDArray[np.float16]:
    """
    Prepare image input for model inference.

    Args:
        image: Input image in BGR format with shape (H, W, 3)

    Returns:
        Preprocessed image as float16 array with shape (1, 3, H, W)
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype(np.float16)

    # Normalize pixel values to range [-1, 1]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    img = (img / 255.0 - mean) / std

    # Convert to (1, 3, H, W) format
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    return img.astype(np.float16)


def softmax(x):
    """Apply softmax function to numpy array."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()


def inference(
    session, image: NDArray[np.uint8]
) -> tuple[int, float, NDArray[np.float16]]:
    """
    Run inference on an image and return class prediction with probabilities.

    Args:
        session: ONNX runtime session
        image: Input image in BGR format

    Returns:
        tuple containing:
        - predicted class index (int)
        - confidence score (float)
        - probability distribution (numpy array)
    """
    input_tensor = prepare_input(image)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: input_tensor})[0]

    # Apply softmax to get probabilities
    probabilities = softmax(output[0])
    max_index = np.argmax(probabilities)
    max_value = probabilities[max_index]

    return max_index, max_value, probabilities

def process_video(video_path: str, session, output_path: str = None, display: bool = True):
    """
    Process video file and perform inference on each frame.
    
    Args:
        video_path: Path to input video file
        session: ONNX runtime session
        output_path: Path to save output video (optional)
        display: Whether to display video while processing
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer if output path is specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    classes = ['left', 'right', 'straight']
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference
            max_index, confidence, probs = inference(session, frame)
            
            # Draw prediction on frame
            text = f"{classes[max_index]}: {confidence:.2f}"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2, cv2.LINE_AA)

            if display:
                cv2.imshow('Video Processing', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if writer:
                writer.write(frame)

    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
if __name__ == "__main__":
    model_path = "model/direction_model_float16.onnx"

    # Load model
    session = load_model(model_path)

    # # Load and preprocess image
    # image = cv2.imread(image_path)

    # # Perform inference
    # max_index, max_value, probabilities = inference(session, image)

    # print(f"Predicted Class: {max_index}, Confidence: {max_value}")


    video_path = "videos/a.mp4"  # Replace with your video path
    # output_path = "./output_video.mp4"  # Optional output path
    
    try:
        process_video(video_path, session,)
    except Exception as e:
        print(f"Error processing video: {str(e)}")

# 0 left
# 1 right
# 2 straight
