import cv2
import torch
from PIL import Image
import numpy as np
from model import SRGAN
import logging
from tqdm import tqdm

import warnings


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def enhance_video(input_video_path, output_video_path, scale=4, weights_path='weights/SRGAN.pth'):
    """
    Enhance a video using SRGAN.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the enhanced video.
        scale (int): Scaling factor (2 or 4).
        weights_path (str): Path to the model weights.
    """
    try:
        # Set up device and initialize the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        model = SRGAN(device, scale=scale)
        model.load_weights(weights_path)
        logging.info("Model loaded successfully.")

        # Open the input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {input_video_path}")

        # Retrieve video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate output dimensions
        output_width = frame_width * scale
        output_height = frame_height * scale

        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

        logging.info(f"Processing {frame_count} frames...")

        # Process each frame
        for _ in tqdm(range(frame_count), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Convert the frame (BGR) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Enhance the frame using the model
            sr_pil_image = model.predict(pil_image)

            # Convert the enhanced PIL image back to a NumPy array (RGB) then to BGR for OpenCV
            sr_frame = np.array(sr_pil_image)
            sr_frame_bgr = cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)

            # Write the enhanced frame to the output video
            out.write(sr_frame_bgr)

        logging.info(f"Enhanced video saved as: {output_video_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Release resources
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals():
            out.release()
        logging.info("Resources released.")

