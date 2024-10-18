import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import time
from io import BytesIO
from PIL import Image

# Initialize FastAPI app
app = FastAPI()

# Load custom YOLOv8 model
model = YOLO('yolov8m.pt')

# Define the eel class
eel_class_id = 0

# Initialize dictionaries to track eel positions and speeds
eel_positions = {}
eel_speeds = {}

# Constants for speed thresholds (in m/s)
ACTIVE_MIN_SPEED = 0.60
ACTIVE_MAX_SPEED = 0.90

# Function to calculate the Euclidean distance between two points
def calculate_distance(p1, p2):
    return np.sqrt(([0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to classify eel as Active or Inactive based on speed
def classify_eel_speed(eel_id, new_position, old_position, time_elapsed):
    if time_elapsed == 0:
        return 'Inactive'
    
    # Calculate the distance moved (in pixels)
    distance = calculate_distance(new_position, old_position)

    # Convert pixels to meters
    distance_meters = distance / 100

    # Calculate speed in meters per second
    speed = distance_meters / time_elapsed
    eel_speeds[eel_id] = speed # Store the speed for potential future use

    # Classify based on speed thresholds
    if ACTIVE_MIN_SPEED <= speed <= ACTIVE_MAX_SPEED:
        return 'Active'
    else:
        return 'Inactive'
    
# Function to check if eel crossed the virtual line
def check_eel_crossing(eel_id, new_position, old_position, line_position):
    eel_in_count = 0
    eel_out_count = 0

    # If eel moved from left to right 
    if old_position[0] < line_position and new_position[0] > line_position:
        eel_in_count += 1
    # If eel moved from right to left 
    elif old_position[0] > line_position and new_position[0] < line_position:
        eel_out_count += 1
    
    return eel_in_count, eel_out_count

# API endpoint to receive frames and process them
@app.post("/detect_eels/")
async def detect_eels(file: UploadFile = File(...)):
    # Read image bytes and convert to a frame
    image = Image.open(BytesIO(await file.read()))
    frame = np.array(image)

    # Current frame properties
    frame_height, frame_width, _ = frame.shape
    line_position = frame_width // 2 # Vertical line
    
    # Get the current time
    prev_time = time.time()

    # Run the YOLOv8 model on the current frame
    results = model(frame)

    # Reset counts for this frame
    frame_active_eel_count = 0
    frame_inactive_eel_count = 0
    total_eel_count = 0
    active_eel_count = 0
    inactive_eel_count = 0
    eel_in_count = 0
    eel_out_count = 0

    # Process detections
    for result in results:
        boxes = result.boxes  # Bounding boxes
        for box in boxes:
            # Get class ID, confidence, and bounding box coordinates
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Check if the detected object is an eel (by class ID)
            if class_id == eel_class_id:
                eel_id = len(eel_positions)  # Temporary eel ID for each detection
                if eel_id in eel_positions:
                    # Classify eel based on speed
                    old_position = eel_positions[eel_id]
                    time_elapsed = time.time() - prev_time
                    state = classify_eel_speed(eel_id, (center_x, center_y), old_position, time_elapsed)

                    # Check for line crossing
                    in_count, out_count = check_eel_crossing(eel_id, (center_x, center_y), old_position, line_position)
                    eel_in_count += in_count
                    eel_out_count += out_count
                else:
                    state = 'Inactive'  # Assume new eels are inactive initially

                # Update eel position for the next frame
                eel_positions[eel_id] = (center_x, center_y)

                # Count eels based on state
                if state == 'Active':
                    frame_active_eel_count += 1
                else:
                    frame_inactive_eel_count += 1

    # Update total counts
    total_eel_count = frame_active_eel_count + frame_inactive_eel_count
    active_eel_count += frame_active_eel_count
    inactive_eel_count += frame_inactive_eel_count

    # Return results as JSON response
    return {
        "total_eel_count": total_eel_count,
        "active_eels": frame_active_eel_count,
        "inactive_eels": frame_inactive_eel_count,
        "eel_in_count": eel_in_count,
        "eel_out_count": eel_out_count
    }
