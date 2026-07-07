import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "/Users/sindhujatrivedi/Desktop/New Hand Gesture project/hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

def count_fingers(lms):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    count = 0
    for tip, pip in zip(tips, pips):
        if lms[tip].y < lms[pip].y:
            count += 1
    if lms[4].x > lms[3].x:
        count += 1
    return count

def draw_landmarks(frame, lms):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 200, 255), 2)
    for i, pt in enumerate(points):
        cv2.circle(frame, pt, 5, (255, 255, 255), -1)
        cv2.circle(frame, pt, 5, (0, 150, 255), 1)

cap = cv2.VideoCapture(0)
print("Camera started! Show your hand. Press Q to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand_lms in result.hand_landmarks:
            draw_landmarks(frame, hand_lms)
            fingers = count_fingers(hand_lms)

            if fingers == 0:
                label = "FIST - Deep Space"
                color = (100, 100, 255)
            elif fingers <= 3:
                label = f"{fingers} FINGERS - Planet View"
                color = (100, 200, 255)
            else:
                label = "OPEN HAND - Milky Way!"
                color = (100, 255, 200)

            cv2.putText(frame, label, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"Fingers open: {fingers}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Show your hand...", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)

    cv2.imshow("Hand Gesture Space Explorer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera closed.")