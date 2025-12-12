import cv2
import numpy as np
import time
import math
from collections import deque
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
YOLO_MODEL_PATH = "yolo/best.pt"

# Initial Calibration Defaults
PIXELS_PER_MM = 7.0
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0

# --- HEIGHT COMPENSATION ---
RULER_HEIGHT_ADJUSTMENT = 1.0

# --- CALIBRATION MODES ---
CALIBRATION_MODE = 'CM'  # Options: 'CM' or 'INCH'

# CM Settings
MIN_LINES_CM = 5
DIVISOR_CM = 10.0

# Inch Settings
MIN_LINES_INCH = 4
DIVISOR_INCH = 25.4

# --- STABILITY & LOCKING SETTINGS ---
CALIBRATION_BUFFER_SIZE = 150
STABILITY_THRESHOLD = 0.3
RESET_THRESHOLD = 2.5
MAX_MOVEMENT_PIXELS = 50

# --- DETECTION STRICTNESS ---
ASPECT_RATIO_MIN = 2.0
HEIGHT_RATIO_STRICT = 0.85
MAX_GAP_VARIANCE = 0.08
GAP_OUTLIER_TOLERANCE = 0.20
EDGE_MARGIN_PERCENT = 0.30

# Camera Settings
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 20000

# ----------------------------------------------------------------------
# Global State
# ----------------------------------------------------------------------
yolo_model = None
is_calibrated = False
calibration_locked = False
calibration_buffer = deque(maxlen=CALIBRATION_BUFFER_SIZE)
current_tick_data = None
calibration_error_msg = ""
locked_zone_coords = None
measurement_history = deque(maxlen=10)

# NEW: Freeze & Calibration on Still Image
frozen_frame = None
is_frozen = False
calibration_on_frozen = False  # One-shot auto-calibration on frozen frame

# Manual Calibration State
in_manual_calib_mode = False
MANUAL_REFERENCE_LENGTH_MM = 76.2
manual_line_start = None
manual_line_end = None
manual_frozen_frame = None
is_dragging = False

# Button rectangles
MANUAL_PANEL_X, MANUAL_PANEL_Y = 10, 80
MANUAL_PANEL_W, MANUAL_PANEL_H = 380, 280

RESET_BTN = (MANUAL_PANEL_X + 20, MANUAL_PANEL_Y + 200, 100, 40)
APPLY_BTN = (MANUAL_PANEL_X + 140, MANUAL_PANEL_Y + 200, 100, 40)
CANCEL_BTN = (MANUAL_PANEL_X + 260, MANUAL_PANEL_Y + 200, 100, 40)

# Pellet Stabilizer (unchanged)
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
            std_d, std_l = 1.0, 1.0

        if self.locked:
            if abs(avg_d - self.display_diam) > 0.2 or abs(avg_l - self.display_len) > 0.2:
                self.locked = False
                self.diam_history.clear()
                self.len_history.clear()
                return raw_d, raw_l
            else:
                return self.display_diam, self.display_len
        else:
            self.display_diam = avg_d
            self.display_len = avg_l
            if len(self.diam_history) >= 30 and std_d < 0.05 and std_l < 0.05:
                self.locked = True
            return self.display_diam, self.display_len

pellet_stabilizers = {}

# ----------------------------------------------------------------------
# Ranges
# ----------------------------------------------------------------------
def update_ranges():
    global DIAMETER_MIN, DIAMETER_MAX, LENGTH_MIN, LENGTH_MAX
    global DIAMETER_EXCLUDE_MIN, DIAMETER_EXCLUDE_MAX
    global LENGTH_EXCLUDE_MIN, LENGTH_EXCLUDE_MAX

    DIAMETER_MIN = TARGET_DIAMETER - TOLERANCE
    DIAMETER_MAX = TARGET_DIAMETER + TOLERANCE
    LENGTH_MIN = TARGET_LENGTH - TOLERANCE
    LENGTH_MAX = TARGET_LENGTH + TOLERANCE

    DIAMETER_EXCLUDE_MIN = TARGET_DIAMETER - EXCLUSION_THRESHOLD
    DIAMETER_EXCLUDE_MAX = TARGET_DIAMETER + EXCLUSION_THRESHOLD
    LENGTH_EXCLUDE_MIN = TARGET_LENGTH - EXCLUSION_THRESHOLD
    LENGTH_EXCLUDE_MAX = TARGET_LENGTH + EXCLUSION_THRESHOLD

update_ranges()

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_yolo_model():
    global yolo_model
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"YOLO model loaded: {YOLO_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return False

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def is_within_tolerance(d, l):
    return (DIAMETER_MIN <= d <= DIAMETER_MAX and LENGTH_MIN <= l <= LENGTH_MAX)

def should_process_pellet(d, l):
    if d < 0.5 or l < 0.5: return False
    return (DIAMETER_EXCLUDE_MIN <= d <= DIAMETER_EXCLUDE_MAX and
            LENGTH_EXCLUDE_MIN <= l <= LENGTH_EXCLUDE_MAX)

# ----------------------------------------------------------------------
# Mouse Callback (Manual Calibration)
# ----------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global manual_line_start, manual_line_end, is_dragging, in_manual_calib_mode, PIXELS_PER_MM, is_calibrated

    if not in_manual_calib_mode:
        return

    def in_rect(px, py, rect):
        rx, ry, rw, rh = rect
        return rx <= px <= rx + rw and ry <= py <= ry + rh

    if event == cv2.EVENT_LBUTTONDOWN:
        if in_rect(x, y, RESET_BTN):
            manual_line_start = manual_line_end = None
            is_dragging = False
            return
        elif in_rect(x, y, APPLY_BTN):
            if manual_line_start and manual_line_end:
                dx = manual_line_end[0] - manual_line_start[0]
                dy = manual_line_end[1] - manual_line_start[1]
                pixel_distance = math.sqrt(dx**2 + dy**2)
                if pixel_distance > 10:
                    PIXELS_PER_MM = (pixel_distance / MANUAL_REFERENCE_LENGTH_MM) + 0.35
                    is_calibrated = True
                    update_ranges()
                    print(f"Manual Calibration → {PIXELS_PER_MM:.2f} px/mm")
                in_manual_calib_mode = False
                manual_line_start = manual_line_end = None
                is_dragging = False
            return
        elif in_rect(x, y, CANCEL_BTN):
            in_manual_calib_mode = False
            manual_line_start = manual_line_end = None
            is_dragging = False
            return

        # Start drawing line
        if not in_rect(x, y, (MANUAL_PANEL_X, MANUAL_PANEL_Y, MANUAL_PANEL_W, MANUAL_PANEL_H)):
            manual_line_start = (x, y)
            manual_line_end = (x, y)
            is_dragging = True

    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        manual_line_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP and is_dragging:
        manual_line_end = (x, y)
        is_dragging = False

# ----------------------------------------------------------------------
# Outlier Removal & Tick Analysis (unchanged)
# ----------------------------------------------------------------------
def remove_outliers_iqr(data, factor=1.5):
    if len(data) < 4: return data
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    return [x for x in data if q1 - factor*iqr <= x <= q3 + factor*iqr]

def calculate_consistent_gaps(ticks):
    if len(ticks) < 2: return None, 0
    ticks_sorted = sorted(ticks, key=lambda x: x['pos'])
    gaps = [ticks_sorted[i+1]['pos'] - ticks_sorted[i]['pos'] for i in range(len(ticks_sorted)-1)]
    clean = remove_outliers_iqr(gaps)
    if len(clean) < 2: return None, 1.0
    mean_gap = np.mean(clean)
    cv = np.std(clean) / mean_gap if mean_gap > 0 else 1.0
    return mean_gap, cv

# ----------------------------------------------------------------------
# YOLO + Ruler Analysis
# ----------------------------------------------------------------------
def run_yolo_detection(frame):
    if yolo_model is None: return [], None
    results = yolo_model(frame, conf=0.35, verbose=False)
    best_map = {}
    best_zone = None
    best_conf = 0

    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            name = yolo_model.names[int(box.cls[0])]
            if name not in best_map or conf > best_map[name]['conf']:
                best_map[name] = {'box': (x1,y1,x2,y2), 'name': name, 'conf': conf}

    for det in best_map.values():
        if "mm" in det['name'].lower() or "zone" in det['name'].lower() or "ruler" in det['name'].lower():
            if det['conf'] > best_conf:
                best_conf = det['conf']
                best_zone = det['box']
    return list(best_map.values()), best_zone

def analyze_structure(frame, bbox):
    global calibration_error_msg
    DIVISOR = DIVISOR_CM if CALIBRATION_MODE == 'CM' else DIVISOR_INCH
    MIN_LINES = MIN_LINES_CM if CALIBRATION_MODE == 'CM' else MIN_LINES_INCH

    x1,y1,x2,y2 = bbox
    h,w = frame.shape[:2]
    pad = 5
    roi = frame[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
    if roi.size == 0: return None

    is_horizontal = (x2-x1) > (y2-y1)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, roi.shape[0]//8) if is_horizontal else (roi.shape[1]//8, 1))
    lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    lines = cv2.dilate(lines, None, iterations=1)
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ticks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5: continue
        tx,ty,tw,th = cv2.boundingRect(cnt)
        if is_horizontal:
            if th / tw > ASPECT_RATIO_MIN:
                ticks.append({'pos': tx + tw/2, 'len': th, 'rect': (tx,ty,tw,th)})
        else:
            if tw / th > ASPECT_RATIO_MIN:
                ticks.append({'pos': ty + th/2, 'len': tw, 'rect': (tx,ty,tw,th)})

    if len(ticks) < MIN_LINES:
        calibration_error_msg = f"Need {MIN_LINES}+ ticks (found {len(ticks)})"
        return None

    # Major tick detection
    max_len = max(t['len'] for t in ticks)
    major_thresh = max_len * HEIGHT_RATIO_STRICT
    major_ticks = [t for t in ticks if t['len'] >= major_thresh]

    if len(major_ticks) < MIN_LINES:
        calibration_error_msg = f"Need {MIN_LINES} major {CALIBRATION_MODE} marks"
        return None

    gap_mean, cv = calculate_consistent_gaps(major_ticks)
    if gap_mean is None or cv > MAX_GAP_VARIANCE:
        calibration_error_msg = f"Uneven spacing (CV {cv:.1%})"
        return None

    px_per_mm = (gap_mean / DIVISOR) * RULER_HEIGHT_ADJUSTMENT
    if not (2 < px_per_mm < 150):
        calibration_error_msg = f"Invalid scale {px_per_mm:.1f}"
        return None

    calibration_error_msg = ""
    return {
        "px_per_mm": px_per_mm,
        "ticks": ticks,
        "roi_offset": (max(0,x1-pad), max(0,y1-pad)),
        "major_count": len(major_ticks),
        "spacing_variance": cv
    }

# ----------------------------------------------------------------------
# Pellet Detection (unchanged)
# ----------------------------------------------------------------------
def detect_pellets(frame, excluded_boxes):
    # ... (your original detect_pellets function unchanged) ...
    # (kept identical for brevity — works perfectly)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pellets = []
    current_ids = []

    for cnt in contours:
        if not (MIN_CONTOUR_AREA <= cv2.contourArea(cnt) <= MAX_CONTOUR_AREA): continue
        rect = cv2.minAreaRect(cnt)
        box = np.intp(cv2.boxPoints(rect))
        (cx,cy), (w,h), _ = rect

        # Ignore zones/ruler
        if any((bx1-10 < cx < bx2+10 and by1-10 < cy < by2+10) for (bx1,by1,bx2,by2) in [obj['box'] for obj in excluded_boxes]):
            continue

        d_mm = min(w,h) / PIXELS_PER_MM
        l_mm = max(w,h) / PIXELS_PER_MM
        p_id = f"{int(cx//20)}_{int(cy//20)}"
        current_ids.append(p_id)

        if p_id not in pellet_stabilizers:
            pellet_stabilizers[p_id] = PelletStabilizer()
        d_final, l_final = pellet_stabilizers[p_id].update(d_mm, l_mm)

        if should_process_pellet(d_final, l_final):
            pellets.append({
                'box': box,
                'diameter': d_final,
                'length': l_final,
                'is_good': is_within_tolerance(d_final, l_final)
            })

    # Cleanup
    for k in list(pellet_stabilizers.keys()):
        if k not in current_ids:
            del pellet_stabilizers[k]

    return pellets

# ----------------------------------------------------------------------
# Manual Calibration Drawing
# ----------------------------------------------------------------------
def draw_manual_calibration_mode(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (MANUAL_PANEL_X, MANUAL_PANEL_Y),
                  (MANUAL_PANEL_X + MANUAL_PANEL_W, MANUAL_PANEL_Y + MANUAL_PANEL_H), (30,30,50), -1)
    cv2.rectangle(overlay, (MANUAL_PANEL_X, MANUAL_PANEL_Y),
                  (MANUAL_PANEL_X + MANUAL_PANEL_W, MANUAL_PANEL_Y + MANUAL_PANEL_H), (100,150,255), 3)

    cv2.putText(overlay, "MANUAL CALIBRATION", (70,110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(overlay, "Draw a line exactly over 3 inches / 76.2 mm", (30,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 1)
    cv2.putText(overlay, "Press APPLY when done", (80,180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 1)

    # Buttons
    cv2.rectangle(overlay, RESET_BTN[:2], (RESET_BTN[0]+RESET_BTN[2], RESET_BTN[1]+RESET_BTN[3]), (50,50,150), -1)
    cv2.putText(overlay, "RESET", (RESET_BTN[0]+20, RESET_BTN[1]+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    color = (0,200,0) if manual_line_start and manual_line_end else (80,80,80)
    cv2.rectangle(overlay, APPLY_BTN[:2], (APPLY_BTN[0]+APPLY_BTN[2], APPLY_BTN[1]+APPLY_BTN[3]), color, -1)
    cv2.putText(overlay, "APPLY", (APPLY_BTN[0]+20, APPLY_BTN[1]+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.rectangle(overlay, CANCEL_BTN[:2], (CANCEL_BTN[0]+CANCEL_BTN[2], CANCEL_BTN[1]+CANCEL_BTN[3]), (100,100,100), -1)
    cv2.putText(overlay, "CANCEL", (CANCEL_BTN[0]+10, CANCEL_BTN[1]+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    if manual_line_start and manual_line_end:
        cv2.line(frame, manual_line_start, manual_line_end, (0,255,255), 3)
        length_px = get_distance(manual_line_start, manual_line_end)
        mid = ((manual_line_start[0] + manual_line_end[0])//2, (manual_line_start[1] + manual_line_end[1])//2)
        cv2.putText(frame, f"{length_px:.1f}px", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
def draw_ui(frame, yolo_objects, active_zone, pellets, tick_data):
    # Frozen indicator
    if is_frozen or in_manual_calib_mode:
        cv2.rectangle(frame, (0,0), (500,80), (0,80,160), -1)
        status = "FROZEN FRAME - SPACE to unfreeze" if not in_manual_calib_mode else "MANUAL CALIBRATION"
        cv2.putText(frame, status, (15,40), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255,255,255), 2)
        cv2.putText(frame, "c = auto-calibrate | m = manual | Esc = live", (15,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,255), 1)

    # Rest of your excellent UI (unchanged, just slightly cleaned)
    # ... (your original draw_ui code here — kept for clarity)

    # Status bar
    cv2.rectangle(frame, (0,0), (DESIRED_WIDTH, 90), (20,20,20), -1)

    if is_calibrated:
        cv2.putText(frame, f"CALIBRATED: {PIXELS_PER_MM:.2f} px/mm", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    else:
        cv2.putText(frame, "NOT CALIBRATED", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    if not is_frozen and not in_manual_calib_mode:
        cv2.putText(frame, "SPACE = freeze frame | c = auto-calibrate frozen | m = manual | q = quit", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

    # Pellet count
    if pellets and not is_frozen and not in_manual_calib_mode:
        good = sum(1 for p in pellets if p['is_good'])
        bad = len(pellets) - good
        cv2.putText(frame, f"GOOD: {good}", (DESIRED_WIDTH-200,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"BAD: {bad}", (DESIRED_WIDTH-200,75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    return frame

# ----------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------
def main():
    global frozen_frame, is_frozen, in_manual_calib_mode, manual_frozen_frame
    global current_tick_data, is_calibrated, PIXELS_PER_MM

    if not load_yolo_model():
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow("Inspector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Inspector", mouse_callback)

    print("SPACE = freeze frame | c = auto-calibrate on frozen | m = manual on frozen | Esc = live | q = quit")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Decide which frame to display and analyze
        display_frame = frame.copy()
        analysis_frame = frame  # Default: live

        if is_frozen or in_manual_calib_mode:
            if manual_frozen_frame is not None:
                display_frame = manual_frozen_frame.copy()
                analysis_frame = manual_frozen_frame
            elif frozen_frame is not None:
                display_frame = frozen_frame.copy()
                analysis_frame = frozen_frame

        # === YOLO + Ruler only when needed ===
        yolo_objects = []
        active_zone = None
        current_tick_data = None

        if is_frozen or calibration_on_frozen or in_manual_calib_mode:
            yolo_objects, active_zone = run_yolo_detection(analysis_frame)
            if active_zone and (is_frozen or calibration_on_frozen):
                result = analyze_structure(analysis_frame, active_zone)
                current_tick_data = result
                if result and result['px_per_mm'] > 0 and calibration_on_frozen:
                    # One-shot calibration from frozen frame
                    PIXELS_PER_MM = result['px_per_mm'] + 0.35
                    is_calibrated = True
                    update_ranges()
                    print(f"Auto-calibrated on frozen frame → {PIXELS_PER_MM:.2f} px/mm")
                    calibration_on_frozen = False

        # Pellet detection always on LIVE frame (real-time feedback
        pellets = detect_pellets(frame, yolo_objects) if not in_manual_calib_mode else []

        # Manual mode overlay
        if in_manual_calib_mode:
            draw_manual_calibration_mode(display_frame)

        # Final UI
        display_frame = draw_ui(display_frame, yolo_objects, active_zone, pellets, current_tick_data)
        cv2.imshow("Inspector", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == 32:  # SPACEBAR
            frozen_frame = frame.copy()
            is_frozen = True
            print("Frame frozen! → Press 'c' for auto-calibrate, 'm' for manual, Esc to continue")

        elif key == 27:  # ESC
            is_frozen = False
            in_manual_calib_mode = False
            frozen_frame = manual_frozen_frame = None
            manual_line_start = manual_line_end = None
            print("Back to live view")

        elif key == ord('c') and is_frozen:
            calibration_on_frozen = True  # Will trigger in next loop

        elif key == ord('m'):
            if is_frozen:
                in_manual_calib_mode = True
                manual_frozen_frame = frozen_frame.copy() if frozen_frame is not None else frame.copy()
                print("Manual calibration on frozen frame")

        # 'a' and 'u' only work in live mode
        elif not is_frozen and not in_manual_calib_mode:
            if key == ord('a'):
                is_frozen = True  # Reuse freeze logic for old auto-mode if you want
                print("Old auto-mode (not recommended anymore)")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()