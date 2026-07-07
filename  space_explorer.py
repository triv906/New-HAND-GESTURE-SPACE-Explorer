import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import time

model_path = "/Users/sindhujatrivedi/Desktop/New Hand Gesture project/hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

W, H = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

# ── stars ──────────────────────────────────────────────
np.random.seed(42)
NUM_STARS = 300
star_x = np.random.randint(0, W, NUM_STARS)
star_y = np.random.randint(0, H, NUM_STARS)
star_r = np.random.choice([1, 1, 1, 2, 2, 3], NUM_STARS)
star_bright = np.random.uniform(0.4, 1.0, NUM_STARS)

# ── galaxy stars ───────────────────────────────────────
NUM_GAL = 2000
gal_x, gal_y, gal_color = [], [], []
for _ in range(NUM_GAL):
    angle = np.random.uniform(0, 2 * math.pi)
    r = np.random.power(0.5) * 0.45
    arm = np.random.randint(0, 3)
    spiral = angle + r * 8 + arm * (2 * math.pi / 3)
    cx = int(0.5 * W + math.cos(spiral) * r * W + np.random.uniform(-20, 20))
    cy = int(0.5 * H + math.sin(spiral) * r * H * 0.4 + np.random.uniform(-10, 10))
    cx = max(0, min(W - 1, cx))
    cy = max(0, min(H - 1, cy))
    hue = np.random.randint(140, 180)
    gal_x.append(cx)
    gal_y.append(cy)
    gal_color.append(hue)

PLANETS = [
    {"name": "SATURN",  "color": (100, 200, 230), "size": 90,  "ring": True,  "ring_color": (80, 170, 210)},
    {"name": "NEPTUNE", "color": (200, 100, 60),  "size": 70,  "ring": False, "ring_color": None},
    {"name": "MARS",    "color": (50,  80,  200), "size": 65,  "ring": False, "ring_color": None},
    {"name": "JUPITER", "color": (60,  140, 210), "size": 110, "ring": False, "ring_color": None},
    {"name": "URANUS",  "color": (180, 200, 80),  "size": 75,  "ring": True,  "ring_color": (160, 180, 60)},
]

planet_index = 0
planet_timer = time.time()
scene_blend = 0.0
finger_history = []
start_time = time.time()

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
    (15,16),(13,17),(17,18),(18,19),(19,20),(0,17)
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

def draw_hand(frame, lms):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (180, 180, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (255, 255, 255), -1)

def draw_stars(canvas, t):
    for i in range(NUM_STARS):
        twinkle = star_bright[i] * (0.7 + 0.3 * math.sin(t * 2 + i))
        val = int(255 * twinkle)
        cv2.circle(canvas, (star_x[i], star_y[i]), star_r[i], (val, val, val), -1)

def draw_planet(canvas, planet, t):
    cx, cy = W // 2, H // 2
    p = planet
    size = p["size"]
    col = p["color"]

    if p["ring"]:
        axes = (int(size * 1.9), int(size * 0.45))
        cv2.ellipse(canvas, (cx, cy), axes, 20, 0, 360, p["ring_color"], int(size * 0.28))

    for r in range(size, 0, -1):
        ratio = r / size
        b = int(col[0] * ratio)
        g = int(col[1] * ratio)
        rv = int(col[2] * ratio)
        cv2.circle(canvas, (cx - int(size * 0.2), cy - int(size * 0.2)), r, (b, g, rv), -1)

    cv2.circle(canvas, (cx, cy), size, col, 2)

    if p["name"] == "JUPITER":
        for band in range(-3, 4):
            by = cy + int(band * size * 0.25)
            band_col = (40, 100, 160) if band % 2 == 0 else (80, 160, 220)
            cv2.line(canvas, (cx - size, by), (cx + size, by), band_col, int(size * 0.08))

    moon_dist = size + 40
    moon_angle = t * 60
    mx = int(cx + moon_dist * math.cos(math.radians(moon_angle)))
    my = int(cy + moon_dist * math.sin(math.radians(moon_angle)) * 0.4)
    cv2.circle(canvas, (mx, my), 8, (180, 180, 180), -1)

    cv2.putText(canvas, p["name"], (cx - 60, cy + size + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (220, 220, 255), 2)

def draw_galaxy(canvas, t):
    for i in range(NUM_GAL):
        bright = int(np.random.uniform(80, 220))
        hue = gal_color[i]
        cv2.circle(canvas, (gal_x[i], gal_y[i]), 1, (bright, bright, min(255, bright + hue - 150)), -1)

    cx, cy = W // 2, H // 2
    for r in range(80, 0, -1):
        alpha = int(255 * (1 - r / 80))
        cv2.circle(canvas, (cx, cy), r, (alpha, alpha, min(255, alpha + 60)), -1)

    cv2.putText(canvas, "MILKY WAY GALAXY", (cx - 200, int(H * 0.15)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (220, 220, 255), 3)
    cv2.putText(canvas, "~100,000 Light Years Across", (cx - 190, int(H * 0.22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 220), 2)
    cv2.putText(canvas, "200-400 Billion Stars", (cx - 150, int(H * 0.28)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (140, 140, 200), 2)

print("Space Explorer started! Press Q to quit.")
print("FIST = Deep Space | 1-3 fingers = Planet | Open hand = Milky Way")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    t = time.time() - start_time

    if time.time() - planet_timer > 8:
        planet_index = (planet_index + 1) % len(PLANETS)
        planet_timer = time.time()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    fingers = 0
    hand_found = False
    if result.hand_landmarks:
        hand_found = True
        for hand_lms in result.hand_landmarks:
            fingers = count_fingers(hand_lms)

    finger_history.append(fingers)
    if len(finger_history) > 6:
        finger_history.pop(0)
    smooth_fingers = round(sum(finger_history) / len(finger_history))

    if smooth_fingers == 0:
        target_blend = 0.0
    elif smooth_fingers <= 3:
        target_blend = 0.5
    else:
        target_blend = 1.0

    scene_blend += (target_blend - scene_blend) * 0.06

    # ── build scene canvas ─────────────────────────────
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    if scene_blend < 0.5:
        draw_stars(canvas, t)
        planet_alpha = scene_blend / 0.5
        if planet_alpha > 0.05:
            planet_canvas = np.zeros((H, W, 3), dtype=np.uint8)
            draw_stars(planet_canvas, t)
            draw_planet(planet_canvas, PLANETS[planet_index], t)
            canvas = cv2.addWeighted(canvas, 1 - planet_alpha, planet_canvas, planet_alpha, 0)
    else:
        planet_alpha = max(0.0, 1.0 - (scene_blend - 0.5) / 0.5)
        galaxy_alpha = min(1.0, (scene_blend - 0.5) / 0.5)

        planet_canvas = np.zeros((H, W, 3), dtype=np.uint8)
        draw_stars(planet_canvas, t)
        draw_planet(planet_canvas, PLANETS[planet_index], t)

        galaxy_canvas = np.zeros((H, W, 3), dtype=np.uint8)
        draw_galaxy(galaxy_canvas, t)

        canvas = cv2.addWeighted(planet_canvas, planet_alpha, galaxy_canvas, galaxy_alpha, 0)

    # ── webcam overlay ─────────────────────────────────
    small_cam = cv2.resize(frame, (320, 180))
    canvas[H - 190:H - 10, W - 330:W - 10] = small_cam
    cv2.rectangle(canvas, (W - 330, H - 190), (W - 10, H - 10), (100, 100, 255), 2)

    if result.hand_landmarks:
        for hand_lms in result.hand_landmarks:
            draw_hand(canvas, hand_lms)

    # ── gesture label ──────────────────────────────────
    if not hand_found:
        gesture_text = "Show your hand to the camera"
        color = (150, 150, 150)
    elif smooth_fingers == 0:
        gesture_text = "FIST - Deep Space"
        color = (100, 150, 255)
    elif smooth_fingers <= 3:
        gesture_text = f"{smooth_fingers} FINGERS - {PLANETS[planet_index]['name']}"
        color = (100, 220, 255)
    else:
        gesture_text = "OPEN HAND - Milky Way Galaxy!"
        color = (100, 255, 200)

    cv2.rectangle(canvas, (0, 0), (W, 60), (0, 0, 0), -1)
    cv2.putText(canvas, gesture_text, (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(canvas, f"Fingers: {smooth_fingers}", (W - 200, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

    cv2.imshow("Hand Gesture Space Explorer", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Closed.")