import cv2
import numpy as np
import math
from collections import deque
from ultralytics import YOLO

# ========================= CONFIG =========================
YOLO_MODEL_PATH = "yolo/best.pt"

PIXELS_PER_MM = 7.0
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0
RULER_HEIGHT_ADJUSTMENT = 1.0

CALIBRATION_MODE = 'CM'          # 'CM' or 'INCH'
MIN_LINES_CM = 5
DIVISOR_CM = 10.0
MIN_LINES_INCH = 4
DIVISOR_INCH = 25.4

DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 20000

# ======================= GLOBAL STATE =======================
yolo_model = None
is_calibrated = False
frozen_frame = None
is_frozen = False
calibration_on_frozen = False   # ← This was missing! Fixed

in_manual_calib_mode = False
manual_line_start = None
manual_line_end = None
manual_frozen_frame = None
is_dragging = False

# Manual panel
MANUAL_PANEL_X, MANUAL_PANEL_Y = 10, 80
MANUAL_PANEL_W, MANUAL_PANEL_H = 380, 280
RESET_BTN  = (MANUAL_PANEL_X + 20,  MANUAL_PANEL_Y + 200, 100, 40)
APPLY_BTN  = (MANUAL_PANEL_X + 140, MANUAL_PANEL_Y + 200, 100, 40)
CANCEL_BTN = (MANUAL_PANEL_X + 260, MANUAL_PANEL_Y + 200, 100, 40)
MANUAL_REFERENCE_LENGTH_MM = 76.2

pellet_stabilizers = {}

# ======================= RANGES =========================
def update_ranges():
    global DIAMETER_MIN, DIAMETER_MAX, LENGTH_MIN, LENGTH_MAX
    global DIAMETER_EXCLUDE_MIN, DIAMETER_EXCLUDE_MAX
    global LENGTH_EXCLUDE_MIN, LENGTH_EXCLUDE_MAX

    DIAMETER_MIN = TARGET_DIAMETER - TOLERANCE
    DIAMETER_MAX = TARGET_DIAMETER + TOLERANCE
    LENGTH_MIN    = TARGET_LENGTH - TOLERANCE
    LENGTH_MAX    = TARGET_LENGTH + TOLERANCE

    DIAMETER_EXCLUDE_MIN = TARGET_DIAMETER - EXCLUSION_THRESHOLD
    DIAMETER_EXCLUDE_MAX = TARGET_DIAMETER + EXCLUSION_THRESHOLD
    LENGTH_EXCLUDE_MIN    = TARGET_LENGTH - EXCLUSION_THRESHOLD
    LENGTH_EXCLUDE_MAX    = TARGET_LENGTH + EXCLUSION_THRESHOLD

update_ranges()

# ===================== YOLO LOAD ========================
def load_yolo_model():
    global yolo_model
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("YOLO model loaded successfully")
        return True
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return False

# ===================== Pellet Stabilizer =====================
class PelletStabilizer:
    def __init__(self, maxlen=30):
        self.diam_history = deque(maxlen=maxlen)
        self.len_history = deque(maxlen=maxlen)
        self.display_diam = 0.0
        self.display_len = 0.0
        self.locked = False

    def update(self, raw_d, raw_l):
        self.diam_history.append(raw_d)
        self.len_history.append(raw_l)
        avg_d = np.mean(self.diam_history)
        avg_l = np.mean(self.len_history)

        if len(self.diam_history) > 5:
            std_d = np.std(self.diam_history)
            std_l = np.std(self.len_history)
        else:
            std_d = std_l = 1.0

        if self.locked:
            if abs(avg_d - self.display_diam) > 0.2 or abs(avg_l - self.display_len) > 0.2:
                self.locked = False
                self.diam_history.clear()
                self.len_history.clear()
                return raw_d, raw_l
            return self.display_diam, self.display_len
        else:
            self.display_diam = avg_d
            self.display_len = avg_l
            if len(self.diam_history) >= 30 and std_d < 0.05 and std_l < 0.05:
                self.locked = True
            return self.display_diam, self.display_len

# ===================== HELPERS =========================
def get_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def should_process_pellet(d, l):
    if d < 0.5 or l < 0.5: return False
    return (DIAMETER_EXCLUDE_MIN <= d <= DIAMETER_EXCLUDE_MAX and
            LENGTH_EXCLUDE_MIN <= l <= LENGTH_EXCLUDE_MAX)

def is_within_tolerance(d, l):
    return (DIAMETER_MIN <= d <= DIAMETER_MAX and LENGTH_MIN <= l <= LENGTH_MAX)

# ===================== YOLO + RULER =====================
def run_yolo_detection(frame):
    if yolo_model is None: return [], None
    try:
        results = yolo_model(frame, conf=0.35, verbose=False)[0]
        best_map = {}
        best_zone = None
        best_conf = 0

        for box in results.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            name = results.names[int(box.cls[0])]
            if name not in best_map or conf > best_map[name]['conf']:
                best_map[name] = {'box': (x1,y1,x2,y2), 'conf': conf}

        for det in best_map.values():
            if any(k in det['name'].lower() for k in ["mm","zone","ruler"]):
                if det['conf'] > best_conf:
                    best_conf = det['conf']
                    best_zone = det['box']
        return list(best_map.values()), best_zone
    except:
        return [], None

def analyze_structure(frame, bbox):
    DIVISOR = DIVISOR_CM if CALIBRATION_MODE == 'CM' else DIVISOR_INCH
    MIN_LINES = MIN_LINES_CM if CALIBRATION_MODE == 'CM' else MIN_LINES_INCH

    x1,y1,x2,y2 = bbox
    pad = 10
    roi = frame[max(0,y1-pad):min(frame.shape[0],y2+pad),
                max(0,x1-pad):min(frame.shape[1],x2+pad)]
    if roi.size == 0: return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,19,5)

    horizontal = (x2-x1) > (y2-y1)
    k_size = roi.shape[0]//8 if horizontal else roi.shape[1]//8
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,k_size) if horizontal else (k_size,1))
    lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    lines = cv2.dilate(lines, None, iterations=1)
    contours,_ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ticks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 8: continue
        tx,ty,tw,th = cv2.boundingRect(cnt)
        if horizontal and th/tw > 2.0:
            ticks.append({'pos': tx+tw/2, 'len': th})
        elif not horizontal and tw/th > 2.0:
            ticks.append({'pos': ty+th/2, 'len': tw})

    if len(ticks) < MIN_LINES: return None

    max_len = max(t['len'] for t in ticks)
    major = [t for t in ticks if t['len'] > max_len*0.85]
    if len(major) < MIN_LINES: return None

    gaps = [major[i+1]['pos'] - major[i]['pos'] for i in range(len(major)-1)]
    gaps = [g for g in gaps if np.percentile(gaps,10) < g < np.percentile(gaps,90)]
    if len(gaps) < 2: return None
    mean_gap = np.mean(gaps)
    cv = np.std(gaps)/mean_gap if mean_gap > 0 else 1

    if cv > 0.08: return None

    px_per_mm = (mean_gap / DIVISOR) * RULER_HEIGHT_ADJUSTMENT
    if not (3 < px_per_mm < 120): return None

    return {"px_per_mm": px_per_mm}

# ===================== PELLET DETECTION =====================
def detect_pellets(frame, excluded_boxes):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray,9,75,75)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,11,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pellets = []
    ids = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA): continue
        rect = cv2.minAreaRect(cnt)
        box = np.intp(cv2.boxPoints(rect))
        (cx,cy),(w,h),_ = rect

        if any(abs(cx-bcx)<60 and abs(cy-bcy)<60
                for bcx,bcy,_,_ in [obj['box'] for obj in excluded_boxes]):
            continue

        d = min(w,h) / PIXELS_PER_MM
        l = max(w,h) / PIXELS_PER_MM
        pid = f"{int(cx//20)}_{int(cy//20)}"
        ids.append(pid)

        if pid not in pellet_stabilizers:
            pellet_stabilizers[pid] = PelletStabilizer()
        d_stab, l_stab = pellet_stabilizers[pid].update(d, l)

        if should_process_pellet(d_stab, l_stab):
            pellets.append({'box': box, 'diameter': round(d_stab,3), 'length': round(l_stab,3),
                            'is_good': is_within_tolerance(d_stab, l_stab)})

    for k in list(pellet_stabilizers.keys()):
        if k not in ids:
            del pellet_stabilizers[k]
    return pellets

# ===================== MOUSE & MANUAL =====================
def mouse_callback(event, x, y, flags, param):
    global manual_line_start, manual_line_end, is_dragging, in_manual_calib_mode, PIXELS_PER_MM, is_calibrated

    if not in_manual_calib_mode: return

    def inside(px,py,rect):
        rx,ry,rw,rh = rect
        return rx <= px <= rx+rw and ry <= py <= ry+rh

    if event == cv2.EVENT_LBUTTONDOWN:
        if inside(x,y,RESET_BTN):
            manual_line_start = manual_line_end = None
            is_dragging = False
        elif inside(x,y,APPLY_BTN) and manual_line_start and manual_line_end:
            px_dist = get_distance(manual_line_start, manual_line_end)
            if px_dist > 30:
                PIXELS_PER_MM = (px_dist / 76.2) + 0.35
                is_calibrated = True
                update_ranges()
                print(f"Manual calibration applied: {PIXELS_PER_MM:.2f} px/mm")
            in_manual_calib_mode = False
            manual_line_start = manual_line_end = None
        elif inside(x,y,CANCEL_BTN):
            in_manual_calib_mode = False
            manual_line_start = manual_line_end = None
        else:
            manual_line_start = (x,y)
            manual_line_end = (x,y)
            is_dragging = True

    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        manual_line_end = (x,y)
    elif event == cv2.EVENT_LBUTTONUP and is_dragging:
        manual_line_end = (x,y)
        is_dragging = False

def draw_manual_panel(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (MANUAL_PANEL_X,MANUAL_PANEL_Y),
                  (MANUAL_PANEL_X+MANUAL_PANEL_W,MANUAL_PANEL_Y+MANUAL_PANEL_H),(30,30,50),-1)
    cv2.rectangle(overlay, (MANUAL_PANEL_X,MANUAL_PANEL_Y),
                  (MANUAL_PANEL_X+MANUAL_PANEL_W,MANUAL_PANEL_Y+MANUAL_PANEL_H),(100,150,255),3)
    cv2.putText(overlay,"MANUAL CALIBRATION",(50,110),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),2)
    cv2.putText(overlay,"Draw line exactly 76.2 mm (3 inch)",(30,150),cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,255,200),2)

    cv2.rectangle(overlay, RESET_BTN[:2], (RESET_BTN[0]+RESET_BTN[2],RESET_BTN[1]+RESET_BTN[3]),(80,80,80),-1)
    cv2.putText(overlay,"RESET",(RESET_BTN[0]+20,RESET_BTN[1]+28),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    color = (0,220,0) if manual_line_start and manual_line_end else (80,80,80)
    cv2.rectangle(overlay, APPLY_BTN[:2], (APPLY_BTN[0]+APPLY_BTN[2],APPLY_BTN[1]+APPLY_BTN[3]),color,-1)
    cv2.putText(overlay,"APPLY",(APPLY_BTN[0]+20,APPLY_BTN[1]+28),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    cv2.rectangle(overlay, CANCEL_BTN[:2], (CANCEL_BTN[0]+CANCEL_BTN[2],CANCEL_BTN[1]+CANCEL_BTN[3]),(100,50,50),-1)
    cv2.putText(overlay,"CANCEL",(CANCEL_BTN[0]+10,CANCEL_BTN[1]+28),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    cv2.addWeighted(overlay,0.85,frame,0.15,0,frame)

    if manual_line_start and manual_line_end:
        cv2.line(frame, manual_line_start, manual_line_end, (0,255,255),3)
        px = get_distance(manual_line_start, manual_line_end)
        cv2.putText(frame, f"{px:.1f}px",(manual_line_start[0]+30,manual_line_start[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

# ===================== MAIN LOOP =====================
def main():
    global frozen_frame, is_frozen, calibration_on_frozen, in_manual_calib_mode
    global manual_frozen_frame, is_calibrated, PIXELS_PER_MM

    if not load_yolo_model():
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow("Pellet Inspector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Pellet Inspector", mouse_callback)

    print("\nPellet Inspector READY")
    print("SPACE = freeze | c = auto-calibrate frozen | m = manual | r = reset calib | q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()

        # Show frozen frame if active
        if is_frozen and frozen_frame is not None:
            display_frame = frozen_frame.copy()
        elif in_manual_calib_mode and manual_frozen_frame is not None:
            display_frame = manual_frozen_frame.copy()

        # Auto-calibration on frozen frame
        if calibration_on_frozen and frozen_frame is not None:
            yolo_objs, zone = run_yolo_detection(frozen_frame)
            if zone:
                result = analyze_structure(frozen_frame, zone)
                if result:
                    PIXELS_PER_MM = result["px_per_mm"] + 0.35
                    is_calibrated = True
                    update_ranges()
                    print(f"AUTO-CALIBRATION SUCCESS → {PIXELS_PER_MM:.2f} px/mm (locked)")
            calibration_on_frozen = False  # one-shot

        # Manual calibration overlay
        if in_manual_calib_mode:
            draw_manual_panel(display_frame)

        # Always detect pellets on live frame
        yolo_objs, _ = run_yolo_detection(frame)
        pellets = detect_pellets(frame, yolo_objs)

        # UI
        status_col = (0,255,0) if is_calibrated else (0,0,255)
        cv2.putText(display_frame, f"{'CALIBRATED' if is_calibrated else 'UNCALIBRATED'}  {PIXELS_PER_MM:.2f} px/mm",
                    (20,60), cv2.FONT_HERSHEY_DUPLEX, 1.3, status_col, 3)

        if is_frozen:
            cv2.putText(display_frame, "FROZEN FRAME  |  c = auto-calibrate  |  m = manual  |  Esc = live",
                        (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,200), 2)

        # Pellet count
        if pellets:
            good = sum(p['is_good'] for p in pellets)
            cv2.putText(display_frame, f"GOOD: {good}", (1050,60), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0,255,0), 3)
            cv2.putText(display_frame, f"BAD:  {len(pellets)-good}", (1050,110), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0,0,255), 3)

        for p in pellets:
            color = (0,255,0) if p['is_good'] else (0,0,255)
            cv2.drawContours(display_frame, [p['box']], 0, color, 2)
            cx = int(np.mean(p['box'][:,0]))
            cy = int(np.mean(p['box'][:,1]))
            txt = f"{p['diameter']:.2f}×{p['length']:.2f}"
            cv2.putText(display_frame, txt, (cx-70, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if not p['is_good']:
                cv2.putText(display_frame, "BAD", (cx+40, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

        cv2.imshow("Pellet Inspector", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == 32:  # Space
            frozen_frame = frame.copy()
            is_frozen = True
            print("Frame frozen — press 'c' for auto or 'm' for manual")
        elif key == 27:  # Esc
            is_frozen = False
            in_manual_calib_mode = False
            frozen_frame = manual_frozen_frame = None
            manual_line_start = manual_line_end = None
        elif key == ord('c') and is_frozen:
            calibration_on_frozen = True
        elif key == ord('m'):
            in_manual_calib_mode = True
            manual_frozen_frame = frozen_frame.copy() if frozen_frame is not None else frame.copy()
        elif key == ord('r'):
            is_calibrated = False
            PIXELS_PER_MM = 7.0
            update_ranges()
            print("Calibration reset")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()