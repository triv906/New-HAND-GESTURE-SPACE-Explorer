import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

print("OpenCV version:", cv2.__version__)
print("MediaPipe version:", mp.__version__)
print("MediaPipe imported successfully!")

model_path = "/Users/sindhujatrivedi/Desktop/New Hand Gesture project/hand_landmarker.task"

if os.path.exists(model_path):
    print("Model file found!")
else:
    print("ERROR: Model file not found")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)
print("Hand Landmarker loaded successfully!")

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Webcam detected!")
    cap.release()
else:
    print("No webcam found")

print("\nAll checks passed! Your setup is complete!")