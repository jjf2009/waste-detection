import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import time
from ultralytics import YOLO
import settings
import os

# Mediapipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
global stage, counter
# Output directory for recorded videos
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)
def get_unique_filename():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"recorded_video_{timestamp}.mp4")

# Initialize video capture and writer
cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files

# Global variables
counter = 0
stage = None

# Load YOLO model
model_path = Path(settings.DETECTION_MODEL)

def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model(model_path)

# Generate unique filename for recordings
def get_unique_filename():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"captured_image_{timestamp}.jpg")

# Calculate angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

# Categorize detected objects
def classify_waste_type(detected_items):
    recyclable_items = set(detected_items) & set(settings.RECYCLABLE)
    non_recyclable_items = set(detected_items) & set(settings.NON_RECYCLABLE)
    hazardous_items = set(detected_items) & set(settings.HAZARDOUS)
    return recyclable_items, non_recyclable_items, hazardous_items

# Format class names for display
def remove_dash_from_class_name(class_name):
    return class_name.replace("_", " ")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to retrieve frame. Exiting...")
            break

        # Resize frame for consistent display
        frame = cv2.resize(frame, (640, 480))

        # Convert to RGB for Mediapipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # Convert back to BGR for OpenCV display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=3, circle_radius=1),
        )

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Extract coordinates for left arm
                shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)

                # Display angle on frame
                cv2.putText(
                    image, f"{int(angle_left)}°",
                    tuple(np.multiply(elbow_left, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                )

                # Curl counter logic
                
                if angle_left > 160:
                    stage = "down"
                if angle_left < 30 and stage == "down":
                    stage = "up"
                    counter += 1
                    file_path = os.path.join(output_dir, "captured_image.jpg")
                    cv2.imwrite(file_path, frame)
        except Exception as e:
            print(f"Error processing landmarks: {e}")

        # Display curl counter
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
        cv2.putText(image, 'CURL COUNTER', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # YOLO object detection
        res = model.predict(image, conf=0.6)
        names = model.names
        detected_items = set()

        for result in res:
            if hasattr(result, 'boxes') and hasattr(result.boxes, 'cls'):
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()

                # Draw bounding boxes and labels
                for box, cls, conf in zip(boxes, classes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{names[cls]}: {conf:.2f}"
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Collect detected items
                detected_items.update([names[cls] for cls in classes])

        # Classify detected items
        recyclable_items, non_recyclable_items, hazardous_items = classify_waste_type(detected_items)

        # Display waste categorization
        y_offset = 20
        for category, color in zip(
                [recyclable_items, non_recyclable_items, hazardous_items],
                [(0, 255, 0), (0, 0, 255), (255, 0, 0)]):  # Green, Red, Blue
            for item in category:
                cv2.putText(image, f"{remove_dash_from_class_name(item)}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 20

        # Show the frame
        cv2.imshow('Pose and Object Detection', image)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
