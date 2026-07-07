import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import time
import random

model_path = "/Users/sindhujatrivedi/Desktop/New Hand Gesture project/hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

W, H = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

random.seed(42)
np.random.seed(42)

# ── stars ──────────────────────────────────────────────
NUM_STARS = 400
stars = [(random.randint(0,W), random.randint(0,H),
          random.choice([1,1,1,2,2,3]),
          random.uniform(0.3,1.0),
          random.uniform(0,2*math.pi)) for _ in range(NUM_STARS)]

# ── shooting stars ─────────────────────────────────────
class ShootingStar:
    def __init__(self):
        self.active = False
        self.timer  = random.uniform(0,6)
        self.x = self.y = self.alpha = 0
        self.length = self.speed = self.angle = 0
    def update(self, dt):
        self.timer -= dt
        if self.timer <= 0 and not self.active:
            self.active = True
            self.x      = random.randint(100,W)
            self.y      = random.randint(0,H//4)
            self.length = random.randint(80,180)
            self.speed  = random.uniform(12,22)
            self.angle  = math.radians(random.uniform(20,40))
            self.alpha  = 1.0
        if self.active:
            self.x += self.speed*math.cos(self.angle)
            self.y += self.speed*math.sin(self.angle)
            self.alpha -= 0.04
            if self.alpha <= 0 or self.x > W or self.y > H:
                self.active = False
                self.timer  = random.uniform(3,9)

shooting_stars = [ShootingStar() for _ in range(4)]

# ── planets ────────────────────────────────────────────
PLANETS = {
    1: {"name":"MERCURY","fingers":"1 Finger","radius":55,
        "body":(105,105,112),"atm":(130,130,138),
        "ring":False,"bands":False,"moons":0,"moon_col":(150,150,150),
        "ring_col":(0,0,0),"color_name":"Slate Gray",
        "fact1":"Airless rocky surface covered in thick dark gray dust.",
        "fact2":"Closest planet to the Sun.",
        "band_cols":[]},
    2: {"name":"VENUS","fingers":"2 Fingers","radius":75,
        "body":(160,210,240),"atm":(180,230,255),
        "ring":False,"bands":True,"moons":0,"moon_col":(200,220,240),
        "ring_col":(0,0,0),"color_name":"Pearly White / Pale Yellow",
        "fact1":"Thick CO2 atmosphere with sulfuric acid clouds.",
        "fact2":"Hottest planet in the solar system.",
        "band_cols":[(170,215,245),(150,200,235),(180,225,250),(145,195,230)]},
    3: {"name":"EARTH","fingers":"3 Fingers","radius":78,
        "body":(180,100,30),"atm":(210,140,60),
        "ring":False,"bands":True,"moons":1,"moon_col":(200,200,200),
        "ring_col":(0,0,0),"color_name":"Vibrant Blue",
        "fact1":"Oceans cover 71% of the surface.",
        "fact2":"Only known planet to harbor life.",
        "band_cols":[(60,140,34),(180,100,30),(60,140,34),(200,150,80)]},
    4: {"name":"MARS","fingers":"4 Fingers","radius":65,
        "body":(50,80,200),"atm":(80,110,220),
        "ring":False,"bands":False,"moons":2,"moon_col":(160,140,130),
        "ring_col":(0,0,0),"color_name":"Dusky Red",
        "fact1":"Iron-rich dust oxidized over billions of years.",
        "fact2":"Home of Olympus Mons, tallest volcano.",
        "band_cols":[]},
    5: {"name":"JUPITER","fingers":"5 Fingers","radius":115,
        "body":(80,150,210),"atm":(100,175,230),
        "ring":False,"bands":True,"moons":4,"moon_col":(200,200,180),
        "ring_col":(0,0,0),"color_name":"Yellow, Orange & Brown",
        "fact1":"Great Red Spot storm raging for centuries.",
        "fact2":"Largest planet in the solar system.",
        "band_cols":[(60,120,180),(90,160,220),(50,100,160),(100,170,230),(70,130,190)]},
    6: {"name":"SATURN","fingers":"6 Fingers","radius":95,
        "body":(160,195,215),"atm":(180,215,230),
        "ring":True,"ring_col":(140,170,190),"bands":True,"moons":3,
        "moon_col":(180,190,200),"color_name":"Pale Yellow & Gray",
        "fact1":"Ring system made of ice and rock particles.",
        "fact2":"Least dense planet would float on water.",
        "band_cols":[(150,185,205),(170,205,220),(145,180,200),(165,200,215)]},
    7: {"name":"URANUS","fingers":"7 Fingers","radius":80,
        "body":(210,210,100),"atm":(230,230,130),
        "ring":True,"ring_col":(170,175,80),"bands":False,"moons":2,
        "moon_col":(180,210,210),"color_name":"Cyan / Pale Blue",
        "fact1":"Methane absorbs red light, reflects blue-green.",
        "fact2":"Rotates on its side at 98 degrees.",
        "band_cols":[]},
    8: {"name":"NEPTUNE","fingers":"8 Fingers","radius":72,
        "body":(200,80,30),"atm":(220,110,60),
        "ring":False,"bands":False,"moons":2,"moon_col":(150,160,200),
        "ring_col":(0,0,0),"color_name":"Deep Blue",
        "fact1":"Fastest winds in the solar system.",
        "fact2":"Takes 165 Earth years to orbit the Sun.",
        "band_cols":[]},
}

# ── pre-render galaxy ──────────────────────────────────
galaxy_cache = None

def build_galaxy_cache():
    global galaxy_cache
    print("Building galaxy... please wait")
    canvas = np.zeros((H,W,3),dtype=np.uint8)
    cx,cy  = W//2, H//2

    # deep black base
    canvas[:] = (5,3,15)

    # background stars — dense field
    for _ in range(2500):
        sx     = random.randint(0,W)
        sy     = random.randint(0,H)
        bright = random.randint(60,220)
        size   = random.choice([1,1,1,1,2])
        cv2.circle(canvas,(sx,sy),size,(bright,bright,min(255,bright+20)),-1)

    # spiral arms — 3 arms with particles
    NUM_ARM_STARS = 4000
    for _ in range(NUM_ARM_STARS):
        arm   = random.randint(0,2)
        r     = random.betavariate(1,2)*0.46
        angle = r*8 + arm*(2*math.pi/3) + random.gauss(0,0.06)
        gx    = int(cx + math.cos(angle)*r*W*0.72)
        gy    = int(cy + math.sin(angle)*r*H*0.36)
        if 0<=gx<W and 0<=gy<H:
            bright = int(random.uniform(120,255)*(1-r*0.4))
            hue    = random.randint(-20,50)
            col    = (max(0,min(255,bright+hue)),
                      max(0,min(255,bright)),
                      max(0,min(255,bright+40)))
            size   = 1 if bright < 170 else 2
            cv2.circle(canvas,(gx,gy),size,col,-1)

    # soft milky white band across center
    band_overlay = canvas.copy()
    for bx in range(0,W,2):
        wave      = int(math.sin(bx*0.006)*22)
        band_cy   = cy + wave
        for by_off in range(-80,80):
            by = band_cy + by_off
            if 0<=by<H:
                fade = max(0,1-(abs(by_off)/80)**1.6)
                val  = int(fade*fade*140)
                if val > 0:
                    existing         = band_overlay[by,bx].tolist()
                    band_overlay[by,bx] = (min(255,existing[0]+val//4),
                                           min(255,existing[1]+val//4),
                                           min(255,existing[2]+val//2))
    cv2.addWeighted(band_overlay,0.7,canvas,0.3,0,canvas)

    # galactic bar through core
    for br in range(60,0,-2):
        fade = (br/60)**1.5
        iv   = int(fade*180)
        cv2.ellipse(canvas,(cx,cy),(int(W*0.18),br),0,0,360,
                    (int(iv*0.8),int(iv*0.85),min(255,iv+60)),-1)

    # bright glowing core
    for gr in range(120,0,-3):
        fade = (1-(gr/120))**1.6
        iv   = int(fade*255)
        cv2.circle(canvas,(cx,cy),gr,
                   (min(255,int(iv*0.8)),
                    min(255,int(iv*0.85)),
                    min(255,iv+40)),-1)

    # bright sparkle stars
    sparkles = [(150,100),(980,180),(350,560),(1120,380),
                (190,580),(720,60),(1060,580),(500,200),(800,500)]
    for (spx,spy) in sparkles:
        cv2.circle(canvas,(spx,spy),2,(255,255,255),-1)
        cv2.line(canvas,(spx-10,spy),(spx+10,spy),(180,180,220),1)
        cv2.line(canvas,(spx,spy-10),(spx,spy+10),(180,180,220),1)

    galaxy_cache = canvas.copy()
    print("Galaxy ready!")

# ── draw functions ─────────────────────────────────────
def draw_stars_bg(canvas, t):
    for (sx,sy,sr,sbright,sphase) in stars:
        twinkle = sbright*(0.6+0.4*math.sin(t*2.5+sphase))
        val     = int(255*twinkle)
        cv2.circle(canvas,(sx,sy),sr,(val,val,min(255,val+30)),-1)

def draw_shooting_stars(canvas, dt):
    for ss in shooting_stars:
        ss.update(dt)
        if ss.active:
            ex = int(ss.x - ss.length*math.cos(ss.angle))
            ey = int(ss.y - ss.length*math.sin(ss.angle))
            a  = max(0,int(ss.alpha*255))
            cv2.line(canvas,(int(ss.x),int(ss.y)),(ex,ey),(a,a,min(255,a+40)),2)

def draw_planet_scene(canvas, p, t):
    # ── planet on LEFT side ────────────────────────────
    cx,cy = W//3, H//2
    R     = p["radius"]
    body  = p["body"]
    atm   = p["atm"]

    # glow aura
    for gr in range(R+80,R,-5):
        fade = max(0,1-(gr-R)/80)
        cv2.circle(canvas,(cx,cy),gr,tuple(int(c*fade*0.3) for c in atm),-1)

    # back ring
    if p["ring"]:
        for rw in range(10,0,-1):
            col = tuple(int(c*(rw/10)) for c in p["ring_col"])
            cv2.ellipse(canvas,(cx,cy),(int(R*2.1),int(R*0.4)),18,180,360,col,max(1,int(R*0.25)))

    # body gradient
    for r in range(R,0,-1):
        ratio = r/R
        hl    = 1.25 - ratio*0.5
        bx2   = min(255,int(body[0]*hl*(0.35+ratio*0.65)))
        gx2   = min(255,int(body[1]*hl*(0.35+ratio*0.65)))
        rx2   = min(255,int(body[2]*hl*(0.35+ratio*0.65)))
        cv2.circle(canvas,(cx-int(R*0.18*ratio),cy-int(R*0.18*ratio)),r,(bx2,gx2,rx2),-1)

    # bands
    if p["bands"] and p["band_cols"]:
        for i,bc in enumerate(p["band_cols"]):
            by2 = cy - R + int(R*0.35*(i+1))
            if cy-R < by2 < cy+R:
                hw  = int(math.sqrt(max(0,R**2-(by2-cy)**2)))
                ov  = canvas.copy()
                cv2.ellipse(ov,(cx,by2),(hw,max(1,int(R*0.05))),0,0,360,bc,-1)
                cv2.addWeighted(ov,0.35,canvas,0.65,0,canvas)

    # Jupiter Great Red Spot
    if p["name"] == "JUPITER":
        sx2,sy2 = cx+int(R*0.3),cy+int(R*0.2)
        cv2.ellipse(canvas,(sx2,sy2),(int(R*0.22),int(R*0.13)),0,0,360,(40,50,180),-1)
        cv2.ellipse(canvas,(sx2,sy2),(int(R*0.22),int(R*0.13)),0,0,360,(20,30,150),2)

    # atmosphere rim
    cv2.circle(canvas,(cx,cy),R,tuple(int(c*0.7) for c in atm),3)
    cv2.circle(canvas,(cx,cy),R+3,tuple(int(c*0.35) for c in atm),2)

    # front ring
    if p["ring"]:
        for rw in range(10,0,-1):
            col = tuple(int(c*(rw/10)) for c in p["ring_col"])
            cv2.ellipse(canvas,(cx,cy),(int(R*2.1),int(R*0.4)),18,0,180,col,max(1,int(R*0.25)))

    # moons
    for m in range(p["moons"]):
        ma  = t*0.6 + m*2.3
        mr  = R + 50 + m*32
        mx2 = int(cx + mr*math.cos(ma))
        my2 = int(cy + mr*math.sin(ma)*0.4)
        mc  = p["moon_col"]
        cv2.circle(canvas,(mx2,my2),8,mc,-1)
        cv2.circle(canvas,(mx2,my2),8,tuple(int(c*0.5) for c in mc),1)

    # ── INFO CARD on RIGHT side ────────────────────────
    card_x  = W//2 + 40
    card_y  = 120
    card_w  = 560
    card_h  = 420

    # card background
    card_bg = canvas.copy()
    cv2.rectangle(card_bg,(card_x,card_y),(card_x+card_w,card_y+card_h),(10,12,35),-1)
    cv2.addWeighted(card_bg,0.85,canvas,0.15,0,canvas)

    # card border
    cv2.rectangle(canvas,(card_x,card_y),(card_x+card_w,card_y+card_h),
                  tuple(int(c*0.6) for c in atm),2)

    # colored top accent bar
    cv2.rectangle(canvas,(card_x,card_y),(card_x+card_w,card_y+6),atm,-1)

    # planet name
    cv2.putText(canvas,p["name"],
                (card_x+20,card_y+55),
                cv2.FONT_HERSHEY_SIMPLEX,1.8,(255,255,255),3)
    cv2.putText(canvas,p["name"],
                (card_x+20,card_y+55),
                cv2.FONT_HERSHEY_SIMPLEX,1.8,tuple(int(c) for c in atm),1)

    # divider line
    cv2.line(canvas,(card_x+20,card_y+70),(card_x+card_w-20,card_y+70),
             tuple(int(c*0.4) for c in atm),1)

    # gesture tag
    cv2.putText(canvas,f"Gesture: {p['fingers']}",
                (card_x+20,card_y+100),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(160,160,200),1)

    # color label row
    cv2.putText(canvas,"COLOR",
                (card_x+20,card_y+140),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(120,120,160),1)
    # color swatch
    cv2.rectangle(canvas,(card_x+110,card_y+122),(card_x+140,card_y+142),atm,-1)
    cv2.rectangle(canvas,(card_x+110,card_y+122),(card_x+140,card_y+142),
                  (200,200,200),1)
    cv2.putText(canvas,p["color_name"],
                (card_x+150,card_y+140),
                cv2.FONT_HERSHEY_SIMPLEX,0.58,(200,200,230),1)

    # divider
    cv2.line(canvas,(card_x+20,card_y+160),(card_x+card_w-20,card_y+160),
             (30,35,70),1)

    # facts header
    cv2.putText(canvas,"PLANET FACTS",
                (card_x+20,card_y+192),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(120,160,220),1)

    # fact 1 bullet
    cv2.circle(canvas,(card_x+28,card_y+220),4,atm,-1)
    cv2.putText(canvas,p["fact1"],
                (card_x+44,card_y+225),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(210,210,230),1)

    # fact 2 bullet
    cv2.circle(canvas,(card_x+28,card_y+262),4,atm,-1)
    cv2.putText(canvas,p["fact2"],
                (card_x+44,card_y+267),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(210,210,230),1)

    # divider
    cv2.line(canvas,(card_x+20,card_y+295),(card_x+card_w-20,card_y+295),
             (30,35,70),1)

    # moons count
    cv2.putText(canvas,"MOONS",
                (card_x+20,card_y+328),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(120,120,160),1)
    moon_str = str(p["moons"]) if p["moons"] > 0 else "None"
    cv2.putText(canvas,moon_str,
                (card_x+130,card_y+328),
                cv2.FONT_HERSHEY_SIMPLEX,0.58,(200,200,230),1)

    # rings
    cv2.putText(canvas,"RINGS",
                (card_x+20,card_y+362),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(120,120,160),1)
    ring_str = "Yes" if p["ring"] else "No"
    ring_col = (100,220,180) if p["ring"] else (160,160,190)
    cv2.putText(canvas,ring_str,
                (card_x+130,card_y+362),
                cv2.FONT_HERSHEY_SIMPLEX,0.58,ring_col,1)

    # position in solar system
    cv2.putText(canvas,"POSITION",
                (card_x+20,card_y+396),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(120,120,160),1)
    positions = {1:"1st",2:"2nd",3:"3rd",4:"4th",5:"5th",6:"6th",7:"7th",8:"8th"}
    pos_str   = f"{positions[list(PLANETS.keys())[list(PLANETS.values()).index(p)]]} from the Sun"
    cv2.putText(canvas,pos_str,
                (card_x+130,card_y+396),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,230),1)

def draw_galaxy_scene(canvas, t):
    global galaxy_cache
    cx,cy = W//2, H//2

    if galaxy_cache is not None:
        np.copyto(canvas,galaxy_cache)

    # twinkling stars on top (only dynamic part)
    for (sx,sy,sr,sbright,sphase) in stars:
        twinkle = sbright*(0.5+0.5*math.sin(t*2+sphase))
        val     = int(220*twinkle)
        cv2.circle(canvas,(sx,sy),sr,(val,val,min(255,val+40)),-1)

    # ── Milky Way info card ────────────────────────────
    card_x = 30
    card_y = 80
    card_w = 420
    card_h = 220

    card_bg = canvas.copy()
    cv2.rectangle(card_bg,(card_x,card_y),(card_x+card_w,card_y+card_h),(5,5,20),-1)
    cv2.addWeighted(card_bg,0.8,canvas,0.2,0,canvas)
    cv2.rectangle(canvas,(card_x,card_y),(card_x+card_w,card_y+card_h),(100,100,200),2)
    cv2.rectangle(canvas,(card_x,card_y),(card_x+card_w,card_y+6),(160,160,255),-1)

    cv2.putText(canvas,"MILKY WAY",(card_x+15,card_y+48),
                cv2.FONT_HERSHEY_SIMPLEX,1.3,(220,220,255),2)
    cv2.putText(canvas,"GALAXY",(card_x+15,card_y+82),
                cv2.FONT_HERSHEY_SIMPLEX,1.0,(180,180,230),1)
    cv2.line(canvas,(card_x+15,card_y+95),(card_x+card_w-15,card_y+95),(50,50,120),1)

    info_lines = [
        "Type: Barred Spiral Galaxy",
        "Diameter: ~100,000 Light Years",
        "Stars: 200-400 Billion",
        "Black Hole: Sagittarius A*",
    ]
    for i,line in enumerate(info_lines):
        cv2.circle(canvas,(card_x+22,card_y+118+i*28),3,(160,160,255),-1)
        cv2.putText(canvas,line,(card_x+36,card_y+123+i*28),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(190,190,230),1)

    # title centered on screen
    title = "MILKY WAY GALAXY"
    (tw,_),_ = cv2.getTextSize(title,cv2.FONT_HERSHEY_SIMPLEX,1.6,3)
    cv2.putText(canvas,title,(cx-tw//2,55),cv2.FONT_HERSHEY_SIMPLEX,1.6,(220,220,255),3)
    cv2.putText(canvas,title,(cx-tw//2,55),cv2.FONT_HERSHEY_SIMPLEX,1.6,(180,180,240),1)

CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),
               (9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
               (13,17),(17,18),(18,19),(19,20),(0,17)]

def count_fingers(lms, handedness):
    tips  = [8, 12, 16, 20]
    pips  = [6, 10, 14, 18]
    mcps  = [5,  9, 13, 17]
    count = 0
    for tip,pip,mcp in zip(tips,pips,mcps):
        tip_y  = lms[tip].y
        pip_y  = lms[pip].y
        mcp_y  = lms[mcp].y
        # finger is open only if tip is clearly above both pip and mcp
        if tip_y < pip_y - 0.02 and tip_y < mcp_y:
            count += 1
    # thumb — stricter check
    thumb_tip  = lms[4]
    thumb_ip   = lms[3]
    thumb_mcp  = lms[2]
    if handedness.lower() == "right":
        if thumb_tip.x > thumb_ip.x + 0.02 and thumb_tip.x > thumb_mcp.x:
            count += 1
    else:
        if thumb_tip.x < thumb_ip.x - 0.02 and thumb_tip.x < thumb_mcp.x:
            count += 1
    return count

def draw_hand(canvas, lms, tip_color=(255,220,120)):
    fh,fw = canvas.shape[:2]
    pts   = [(int(lm.x*fw),int(lm.y*fh)) for lm in lms]
    for a,b in CONNECTIONS:
        cv2.line(canvas,pts[a],pts[b],(120,180,255),2)
    for i,pt in enumerate(pts):
        is_tip = i in [4,8,12,16,20]
        col    = tip_color if is_tip else (200,220,255)
        cv2.circle(canvas,pt,6 if is_tip else 4,col,-1)
        cv2.circle(canvas,pt,6 if is_tip else 4,(80,120,200),1)

def draw_guide(canvas, smooth):
    labels = [(0,"FIST","Deep Space"),(1,"1","Mercury"),(2,"2","Venus"),
              (3,"3","Earth"),(4,"4","Mars"),(5,"5","Jupiter"),
              (6,"6","Saturn"),(7,"7","Uranus"),(8,"8","Neptune"),
              (10,"5+5","Milky Way")]
    box_w   = 108
    total_w = len(labels)*box_w + (len(labels)-1)*4
    start_x = (W-total_w)//2
    y       = H-52

    for i,(fcount,flabel,pname) in enumerate(labels):
        bx       = start_x + i*(box_w+4)
        is_active= (smooth==fcount) or (fcount==10 and smooth>=10)
        bg       = (60,80,160) if is_active else (15,15,28)
        border   = (120,160,255) if is_active else (40,40,70)
        cv2.rectangle(canvas,(bx,y),(bx+box_w,y+42),bg,-1)
        cv2.rectangle(canvas,(bx,y),(bx+box_w,y+42),border,1)
        (tw,_),_ = cv2.getTextSize(flabel,cv2.FONT_HERSHEY_SIMPLEX,0.42,1)
        cv2.putText(canvas,flabel,(bx+box_w//2-tw//2,y+16),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1)
        (nw,_),_ = cv2.getTextSize(pname,cv2.FONT_HERSHEY_SIMPLEX,0.34,1)
        col      = (120,220,255) if is_active else (100,100,140)
        cv2.putText(canvas,pname,(bx+box_w//2-nw//2,y+34),
                    cv2.FONT_HERSHEY_SIMPLEX,0.34,col,1)

# ── main ───────────────────────────────────────────────
scene_blend    = 0.0
finger_history = []
start_time     = time.time()
prev_time      = time.time()
current_planet = 1

build_galaxy_cache()

print("Space Explorer started! Press Q to quit.")

while True:
    success,frame = cap.read()
    if not success: break

    now = time.time()
    dt  = now - prev_time
    prev_time = now
    t   = now - start_time

    frame = cv2.flip(frame,1)

    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)
    result    = detector.detect(mp_image)

    total_fingers = 0
    hand_counts   = []
    hand_found    = False

    if result.hand_landmarks:
        hand_found = True
        handedness_list = result.handedness if result.handedness else []
        for idx,hand_lms in enumerate(result.hand_landmarks):
            hlabel = handedness_list[idx][0].category_name if idx < len(handedness_list) else "right"
            fc     = count_fingers(hand_lms,hlabel)
            hand_counts.append(fc)
            total_fingers += fc

    effective = min(total_fingers,10)

    finger_history.append(effective)
    if len(finger_history) > 10: finger_history.pop(0)
    smooth = round(sum(finger_history)/len(finger_history))
    smooth = max(0,min(10,smooth))

    if 1 <= smooth <= 8:
        current_planet = smooth

    if smooth == 0:   target = 0.0
    elif smooth == 10: target = 1.0
    else:              target = 0.5

    scene_blend += (target-scene_blend)*0.05

    # ── canvas ─────────────────────────────────────────
    canvas = np.zeros((H,W,3),dtype=np.uint8)
    draw_stars_bg(canvas,t)
    draw_shooting_stars(canvas,dt)
    # deep space label when fist shown
    if smooth == 0 and hand_found:
        msg = "DEEP SPACE"
        (mw,_),_ = cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,2.0,3)
        cv2.putText(canvas,msg,(W//2-mw//2,H//2),
                    cv2.FONT_HERSHEY_SIMPLEX,2.0,(60,80,180),3)
        cv2.putText(canvas,msg,(W//2-mw//2,H//2),
                    cv2.FONT_HERSHEY_SIMPLEX,2.0,(100,140,255),1)
        hint = "Open fingers to explore planets"
        (hw,_),_ = cv2.getTextSize(hint,cv2.FONT_HERSHEY_SIMPLEX,0.7,1)
        cv2.putText(canvas,hint,(W//2-hw//2,H//2+45),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(80,100,180),1)

    if scene_blend < 0.5:
        pa = min(1.0,scene_blend/0.5)
        if pa > 0.02:
            pc = np.zeros((H,W,3),dtype=np.uint8)
            draw_stars_bg(pc,t)
            draw_planet_scene(pc,PLANETS[current_planet],t)
            canvas = cv2.addWeighted(canvas,1-pa,pc,pa,0)
    else:
        pa = max(0.0,1.0-(scene_blend-0.5)/0.5)
        ga = min(1.0,(scene_blend-0.5)/0.5)
        pc = np.zeros((H,W,3),dtype=np.uint8)
        draw_stars_bg(pc,t)
        draw_planet_scene(pc,PLANETS[current_planet],t)
        gc = np.zeros((H,W,3),dtype=np.uint8)
        draw_galaxy_scene(gc,t)
        canvas = cv2.addWeighted(pc,pa,gc,ga,0)

    # hands
    if result.hand_landmarks:
        tip_colors = [(255,220,120),(120,255,200)]
        for idx,hand_lms in enumerate(result.hand_landmarks):
            draw_hand(canvas,hand_lms,tip_colors[idx%2])
            fh2,fw2 = canvas.shape[:2]
            wx = int(hand_lms[0].x*fw2)
            wy = int(hand_lms[0].y*fh2)
            fc = hand_counts[idx] if idx < len(hand_counts) else 0
            cv2.putText(canvas,f"{fc}f",(wx-15,wy+28),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,180),2)

    # webcam pip
    small = cv2.resize(frame,(220,124))
    canvas[H-134:H-10,W-230:W-10] = small
    cv2.rectangle(canvas,(W-230,H-134),(W-10,H-10),(80,120,255),2)

    # HUD top
    cv2.rectangle(canvas,(0,0),(W,68),(0,0,0),-1)
    cv2.rectangle(canvas,(0,66),(W,68),(60,80,160),-1)

    num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0

    if not hand_found:
        label = " Show your hand(s) to the camera"
        color = (140,140,140)
    elif smooth == 0:
        label = "  FIST = DEEP SPACE  |  Make a fist to explore the cosmos"
        color = (100,160,255)
    elif smooth == 10:
        label = " 10 FINGERS = MILKY WAY GALAXY  |  Both hands fully open"
        color = (120,255,190)
    elif smooth == 9:
        label = "Almost!  Open one more finger for Milky Way Galaxy"
        color = (100,220,180)
    else:
        p     = PLANETS[current_planet]
        label = f" {p['name']}  |  {p['fingers']}  |  Fist=Space  |  10 Fingers=Milky Way"
        color = tuple(min(255,int(c*1.1)) for c in p["atm"])

    cv2.putText(canvas,label,(20,46),cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)
    cv2.putText(canvas,f"Hands: {num_hands}",(W-150,46),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(160,160,200),1)

    # guide bar
    cv2.rectangle(canvas,(0,H-60),(W,H),(0,0,0),-1)
    cv2.rectangle(canvas,(0,H-60),(W,H-58),(60,80,160),-1)
    draw_guide(canvas,smooth)

    cv2.imshow("Hand Gesture Space Explorer",canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Closed.")