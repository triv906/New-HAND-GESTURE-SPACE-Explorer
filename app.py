import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer
import av
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import time
import random
import os
import urllib.request
import ssl

st.set_page_config(
    page_title="Hand Gesture Space Explorer",
    page_icon="🚀",
    layout="wide"
)

st.markdown("""
<style>
body { background-color: #0a0a0f; color: white; }
.main { background-color: #0a0a0f; }
.stApp { background-color: #0a0a0f; }
h1 { color: white; text-align: center; }
p { color: rgba(255,255,255,0.7); text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Hand Gesture Space Explorer")
st.markdown("**Show your hand to explore the solar system!**")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("✊ **Fist** → Deep Space")
with col2:
    st.info("☝️ **1-8 Fingers** → Planets")
with col3:
    st.info("🖐🖐 **10 Fingers** → Milky Way")

st.markdown("---")

MODEL_PATH = "hand_landmarker.task"

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading hand tracking model... (25MB, one time only)"):
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ctx)
            )
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            data = opener.open(url).read()
            with open(MODEL_PATH, "wb") as f:
                f.write(data)
    return MODEL_PATH

@st.cache_resource
def load_detector():
    model_path = download_model()
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)

detector = load_detector()
st.success("Hand tracking model loaded!")

W, H = 1280, 720

random.seed(42)
np.random.seed(42)
# 1. Define a function that processes each video frame
def video_frame_callback(frame):
    # Convert the incoming web frame to an OpenCV BGR image
    img = frame.to_ndarray(format="bgr24")
    
    # ----------------------------------------------------
    # INSERT YOUR HAND TRACKING / MEDIAPIPE CODE HERE
    # e.g., results = hands.process(img)
    # ----------------------------------------------------
    
    # Return the processed frame back to the browser screen
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 2. Add the component where you want the video to display
webrtc_streamer(key="hand-gesture", video_frame_callback=video_frame_callback)

NUM_STARS = 400
stars = [(random.randint(0,W), random.randint(0,H),
          random.choice([1,1,1,2,2,3]),
          random.uniform(0.3,1.0),
          random.uniform(0,2*math.pi)) for _ in range(NUM_STARS)]

PLANETS = {
    1: {"name":"MERCURY","radius":55,"body":(105,105,112),"atm":(130,130,138),
        "ring":False,"bands":False,"moons":0,"moon_col":(150,150,150),
        "ring_col":(0,0,0),"color_name":"Slate Gray",
        "fact1":"Airless rocky surface","fact2":"Closest to the Sun","band_cols":[]},
    2: {"name":"VENUS","radius":75,"body":(160,210,240),"atm":(180,230,255),
        "ring":False,"bands":True,"moons":0,"moon_col":(200,220,240),
        "ring_col":(0,0,0),"color_name":"Pearly White",
        "fact1":"Thick CO2 atmosphere","fact2":"Hottest planet",
        "band_cols":[(170,215,245),(150,200,235)]},
    3: {"name":"EARTH","radius":78,"body":(180,100,30),"atm":(210,140,60),
        "ring":False,"bands":True,"moons":1,"moon_col":(200,200,200),
        "ring_col":(0,0,0),"color_name":"Vibrant Blue",
        "fact1":"Oceans cover 71%","fact2":"Only planet with life",
        "band_cols":[(60,140,34),(180,100,30)]},
    4: {"name":"MARS","radius":65,"body":(50,80,200),"atm":(80,110,220),
        "ring":False,"bands":False,"moons":2,"moon_col":(160,140,130),
        "ring_col":(0,0,0),"color_name":"Dusky Red",
        "fact1":"Iron rich red dust","fact2":"Home of Olympus Mons","band_cols":[]},
    5: {"name":"JUPITER","radius":115,"body":(80,150,210),"atm":(100,175,230),
        "ring":False,"bands":True,"moons":4,"moon_col":(200,200,180),
        "ring_col":(0,0,0),"color_name":"Yellow Orange Brown",
        "fact1":"Great Red Spot storm","fact2":"Largest planet",
        "band_cols":[(60,120,180),(90,160,220),(50,100,160)]},
    6: {"name":"SATURN","radius":95,"body":(160,195,215),"atm":(180,215,230),
        "ring":True,"ring_col":(140,170,190),"bands":True,"moons":3,
        "moon_col":(180,190,200),"color_name":"Pale Yellow Gray",
        "fact1":"Ice and rock rings","fact2":"Least dense planet",
        "band_cols":[(150,185,205),(170,205,220)]},
    7: {"name":"URANUS","radius":80,"body":(210,210,100),"atm":(230,230,130),
        "ring":True,"ring_col":(170,175,80),"bands":False,"moons":2,
        "moon_col":(180,210,210),"color_name":"Cyan Pale Blue",
        "fact1":"Methane blue green hue","fact2":"Rotates on its side","band_cols":[]},
    8: {"name":"NEPTUNE","radius":72,"body":(200,80,30),"atm":(220,110,60),
        "ring":False,"bands":False,"moons":2,"moon_col":(150,160,200),
        "ring_col":(0,0,0),"color_name":"Deep Blue",
        "fact1":"Fastest winds","fact2":"165 years to orbit Sun","band_cols":[]},
}

CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),
               (9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
               (13,17),(17,18),(18,19),(19,20),(0,17)]

def count_fingers(lms, handedness):
    tips,pips,mcps = [8,12,16,20],[6,10,14,18],[5,9,13,17]
    count = 0
    for tip,pip,mcp in zip(tips,pips,mcps):
        if lms[tip].y < lms[pip].y - 0.02 and lms[tip].y < lms[mcp].y:
            count += 1
    if handedness.lower() == "right":
        if lms[4].x > lms[3].x + 0.02: count += 1
    else:
        if lms[4].x < lms[3].x - 0.02: count += 1
    return count

def draw_stars_bg(canvas, t):
    for (sx,sy,sr,sbright,sphase) in stars:
        twinkle = sbright*(0.6+0.4*math.sin(t*2.5+sphase))
        val = int(255*twinkle)
        cv2.circle(canvas,(sx,sy),sr,(val,val,min(255,val+30)),-1)

def draw_planet(canvas, p, t, key):
    cx,cy = W//2, H//2
    R = p["radius"]
    body = p["body"]
    atm = p["atm"]
    for gr in range(R+80,R,-5):
        fade = max(0,1-(gr-R)/80)
        cv2.circle(canvas,(cx,cy),gr,tuple(int(c*fade*0.3) for c in atm),-1)
    if p["ring"]:
        for rw in range(10,0,-1):
            col = tuple(int(c*(rw/10)) for c in p["ring_col"])
            cv2.ellipse(canvas,(cx,cy),(int(R*2.1),int(R*0.4)),18,180,360,col,max(1,int(R*0.25)))
    for r in range(R,0,-1):
        ratio = r/R
        hl = 1.25-ratio*0.5
        bx = min(255,int(body[0]*hl*(0.35+ratio*0.65)))
        gx = min(255,int(body[1]*hl*(0.35+ratio*0.65)))
        rx = min(255,int(body[2]*hl*(0.35+ratio*0.65)))
        cv2.circle(canvas,(cx-int(R*0.18*ratio),cy-int(R*0.18*ratio)),r,(bx,gx,rx),-1)
    if p["ring"]:
        for rw in range(10,0,-1):
            col = tuple(int(c*(rw/10)) for c in p["ring_col"])
            cv2.ellipse(canvas,(cx,cy),(int(R*2.1),int(R*0.4)),18,0,180,col,max(1,int(R*0.25)))
    for m in range(p["moons"]):
        ma = t*0.6+m*2.3
        mr = R+50+m*32
        mx = int(cx+mr*math.cos(ma))
        my = int(cy+mr*math.sin(ma)*0.4)
        cv2.circle(canvas,(mx,my),8,p["moon_col"],-1)
    name = p["name"]
    (tw,_),_ = cv2.getTextSize(name,cv2.FONT_HERSHEY_SIMPLEX,1.4,3)
    cv2.putText(canvas,name,(cx-tw//2,cy+R+55),cv2.FONT_HERSHEY_SIMPLEX,1.4,(255,255,255),3)
    (cw,_),_ = cv2.getTextSize(p["color_name"],cv2.FONT_HERSHEY_SIMPLEX,0.6,1)
    cv2.putText(canvas,p["color_name"],(cx-cw//2,cy+R+88),cv2.FONT_HERSHEY_SIMPLEX,0.6,tuple(int(c*0.9) for c in atm),1)
    for i,fk in enumerate(["fact1","fact2"]):
        (fw,_),_ = cv2.getTextSize(p[fk],cv2.FONT_HERSHEY_SIMPLEX,0.52,1)
        cv2.putText(canvas,p[fk],(cx-fw//2,cy+R+118+i*26),cv2.FONT_HERSHEY_SIMPLEX,0.52,(170,170,210),1)

def draw_galaxy(canvas, t):
    canvas[:] = (5,3,15)
    for _ in range(2000):
        sx = random.randint(0,W)
        sy = random.randint(0,H)
        bright = random.randint(60,220)
        cv2.circle(canvas,(sx,sy),1,(bright,bright,min(255,bright+20)),-1)
    for gr in range(120,0,-4):
        fade = (1-(gr/120))**1.8
        iv = int(fade*255)
        cv2.circle(canvas,(W//2,H//2),gr,(iv,int(iv*0.85),min(255,iv+60)),-1)
    title = "MILKY WAY GALAXY"
    (tw,_),_ = cv2.getTextSize(title,cv2.FONT_HERSHEY_SIMPLEX,1.5,3)
    cv2.putText(canvas,title,(W//2-tw//2,80),cv2.FONT_HERSHEY_SIMPLEX,1.5,(220,220,255),3)
    desc = [
        "A barred spiral galaxy",
        "200-400 Billion Stars",
        "Sagittarius A* at center"
    ]
    for i,d in enumerate(desc):
        (dw,_),_ = cv2.getTextSize(d,cv2.FONT_HERSHEY_SIMPLEX,0.65,1)
        cv2.putText(canvas,d,(W//2-dw//2,120+i*32),cv2.FONT_HERSHEY_SIMPLEX,0.65,(160,160,210),1)

def draw_hand(canvas, lms):
    fh,fw = canvas.shape[:2]
    pts = [(int(lm.x*fw),int(lm.y*fh)) for lm in lms]
    for a,b in CONNECTIONS:
        cv2.line(canvas,pts[a],pts[b],(120,180,255),2)
    for i,pt in enumerate(pts):
        is_tip = i in [4,8,12,16,20]
        col = (255,220,120) if is_tip else (200,220,255)
        cv2.circle(canvas,pt,6 if is_tip else 4,col,-1)

st.markdown("### Live Camera Feed")
run = st.checkbox("Start Camera", value=False)
FRAME_WINDOW = st.image([])
status_text = st.empty()

finger_history = []
current_planet = 1
t = 0

if run:
    cap = cv2.VideoCapture(0)
    while run:
        success, frame = cap.read()
        if not success:
            st.error("Cannot access camera")
            break

        t += 0.033
        frame = cv2.flip(frame, 1)
        frame_resized = cv2.resize(frame, (W, H))

        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        total_fingers = 0
        hand_counts = []
        hand_found = False

        if result.hand_landmarks:
            hand_found = True
            handedness_list = result.handedness if result.handedness else []
            for idx, hand_lms in enumerate(result.hand_landmarks):
                hlabel = handedness_list[idx][0].category_name if idx < len(handedness_list) else "right"
                fc = count_fingers(hand_lms, hlabel)
                hand_counts.append(fc)
                total_fingers += fc

        effective = min(total_fingers, 10)
        finger_history.append(effective)
        if len(finger_history) > 10:
            finger_history.pop(0)
        smooth = round(sum(finger_history)/len(finger_history))
        smooth = max(0, min(10, smooth))

        if 1 <= smooth <= 8:
            current_planet = smooth

        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        if smooth == 0:
            draw_stars_bg(canvas, t)
            msg = "DEEP SPACE"
            (mw,_),_ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
            cv2.putText(canvas, msg, (W//2-mw//2, H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (100,140,255), 3)
            status_text.info("FIST = DEEP SPACE")
        elif smooth == 10:
            draw_galaxy(canvas, t)
            status_text.success("10 FINGERS = MILKY WAY GALAXY")
        else:
            draw_stars_bg(canvas, t)
            draw_planet(canvas, PLANETS[current_planet], t, current_planet)
            status_text.info(f"{PLANETS[current_planet]['name']} — {smooth} finger(s)")

        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                draw_hand(canvas, hand_lms)

        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(canvas_rgb, use_column_width=True)

    cap.release()
else:
    status_text.warning("Tick the checkbox above to start the camera")
