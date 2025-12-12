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
calibration_on_frozen = False

in_manual_calib_mode = False
manual_line_start = None
manual_line_end = None
manual_frozen_frame = None
is_dragging = False

# For visualization of auto-calibration
last_tick_data = None   # Stores the latest successful tick analysis for drawing

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
        print("YOLO model loaded")
        return True
    except Exception as e:
        print(f"Model load failed: {e}")
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

# ===================== YOLO + RULER (WITH VISUAL DATA) =====================
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

def analyze_structure_and_visualize(frame, bbox):
    """Returns result + tick visualization data"""
    global last_tick_data

    DIVISOR = DIVISOR_CM if CALIBRATION_MODE == 'CM' else DIVISOR_INCH
    MIN_LINES = MIN_LINES_CM if CALIBRATION_MODE == 'CM' else MIN_LINES_INCH

    x1,y1,x2,y2 = bbox
    pad = 15
    roi_x1, roi_y1 = max(0,x1-pad), max(0,y1-pad)
    roi_x2, roi_y2 = min(frame.shape[1],x2+pad), min(frame.shape[0],y2+pad)
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        last_tick_data = None
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,19,5)

    horizontal = (x2-x1) > (y2-y1)
    k_len = roi.shape[0]//8 if horizontal else roi.shape[1]//8
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,k_len) if horizontal else (k_len,1))
    lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    lines = cv2.dilate(lines, None, iterations=1)
    contours,_ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_ticks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 8: continue
        tx,ty,tw,th = cv2.boundingRect(cnt)
        if horizontal:
            if th/tw > 2.0:
                world_x = roi_x1 + tx + tw/2
                world_y = roi_y1 + ty + th/2
                all_ticks.append({'pos': tx + tw/2, 'len': th, 'world': (int(world_x), int(world_y))})
        else:
            if tw/th > 2.0:
                world_x = roi_x1 + tx + tw/2
                world_y = roi_y1 + ty + th/2
                all_ticks.append({'pos': ty + th/2, 'len': tw, 'world': (int(world_x), int(world_y))})

    if len(all_ticks) < MIN_LINES:
        last_tick_data = None
        return None

    max_len = max(t['len'] for t in all_ticks)
    major_ticks = [t for t in all_ticks if t['len'] > max_len * 0.85]

    if len(major_ticks) < MIN_LINES:
        last_tick_data = None
        return None

    # Save for drawing
    last_tick_data = {
        'all_ticks': all_ticks,
        'major_ticks': major_ticks,
        'roi_offset': (roi_x1, roi_y1),
        'horizontal': horizontal
    }

    # Calculate gap
    major_sorted = sorted(major_ticks, key=lambda x: x['pos'])
    gaps = [major_sorted[i+1]['pos'] - major_sorted[i]['pos'] for i in range(len(major_sorted)-1)]
    clean_gaps = [g for g in gaps if np.percentile(gaps,10) < g < np.percentile(gaps,90)]
    if len(clean_gaps) < 2:
        return None
    mean_gap = np.mean(clean_gaps)
    cv = np.std(clean_gaps) / mean_gap

    if cv > 0.08:
        return None

    px_per_mm = (mean_gap / DIVISOR) * RULER_HEIGHT_ADJUSTMENT
    if not (3 < px_per_mm < 120):
        return None

    return {"px_per_mm": px_per_mm, "gap_px": mean_gap, "cv": cv, "major_count": len(major_ticks)}

# ===================== VISUALIZE TICKS =====================
def draw_tick_visualization(frame):
    if last_tick_data is None:
        return

    data = last_tick_data
    off_x, off_y = data['roi_offset']
    is_horiz = data['horizontal']

    # Draw all ticks (cyan)
    for t in data['all_ticks']:
        wx, wy = t['world']
        length = 30
        if is_horiz:
            cv2.line(frame, (wx, wy - length//2), (wx, wy + length//2), (255,255,0), 2)  # cyan
        else:
            cv2.line(frame, (wx - length//2, wy), (wx + length//2, wy), (255,255,0), 2)

    # Draw major ticks (red, thick)
    for t in data['major_ticks']:
        wx, wy = t['world']
        length = 50
        if is_horiz:
            cv2.line(frame, (wx, wy - length//2), (wx, wy + length//2), (0,0,255), 4)
            cv2.circle(frame, (wx, wy), 8, (0,0,255), -1)
        else:
            cv2.line(frame, (wx - length//2, wy), (wx + length//2, wy), (0,0,255), 4)
            cv2.circle(frame, (wx, wy), 8, (0,0,255), -1)

    # Show stats
    if 'gap_px' in globals() and last_tick_data:
        try:
            result = analyze_structure_and_visualize.last_result  # cache
        except:
            result = None
        if result:
            text = f"AUTO: {result['px_per_mm']+0.35:.2f} px/mm  |  Gap: {result['gap_px']:.1f}px  |  CV: {result['cv']:.1%}"
            cv2.putText(frame, text, (20, frame.shape[0]-40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255,255), 2)

# Save last result for display
analyze_structure_and_visualize.last_result = None

# ===================== PELLET DETECTION =====================
# (same as before — unchanged for brevity)

# ===================== MOUSE & MANUAL =====================
# (same as before)

# ===================== MAIN LOOP =====================
def main():
    global frozen_frame, is_frozen, calibration_on_frozen, in_manual_calib_mode
    global manual_frozen_frame, is_calibrated, PIXELS_PER_MM, last_tick_data

    if not load_yolo_model():
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)

    cv2.namedWindow("Pellet Inspector", cv2, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Pellet Inspector", mouse_callback)

    print("\nPellet Inspector READY — NOW WITH FULL RULER VISUALIZATION")
    print("SPACE = freeze | c = auto-calibrate + show ticks | m = manual | r = reset | q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()

        if is_frozen and frozen_frame is not None:
            display_frame = frozen_frame.copy()
        elif in_manual_calib_mode and manual_frozen_frame is not None:
            display_frame = manual_frozen_frame.copy()

        # AUTO CALIBRATION + VISUALIZATION
        if calibration_on_frozen and frozen_frame is not None:
            yolo_objs, zone = run_yolo_detection(frozen_frame)
            if zone:
                result = analyze_structure_and_visualize(frozen_frame, zone)
                analyze_structure_and_visualize.last_result = result  # save for drawing
                if result:
                    PIXELS_PER_MM = result["px_per_mm"] + 0.35
                    is_calibrated = True
                    update_ranges()
                    print(f"AUTO-CALIBRATION → {PIXELS_PER_MM:.2f} px/mm | {result['major_count']} major ticks")
            calibration_on_frozen = False

        # Always show tick visualization if we have data (even after calibration)
        if is_frozen or in_manual_calib_mode:
            draw_tick_visualization(display_frame)

        # Manual mode overlay
        if in_manual_calib_mode:
            draw_manual_panel(display_frame)

        # Pellet detection (live)
        yolo_objs, _ = run_yolo_detection(frame)
        pellets = detect_pellets(frame, yolo_objs)

        # UI
        col = (0,255,0) if is_calibrated else (0,0,255)
        cv2.putText(display_frame, f"{'CALIBRATED' if is_calibrated else 'NOT CALIBRATED'}  {PIXELS_PER_MM:.2f} px/mm",
                    (20,60), cv2.FONT_HERSHEY_DUPLEX, 1.3, col, 3)

        if is_frozen:
            cv2.putText(display_frame, "FROZEN | c = auto-calibrate + show ticks | m = manual | Esc = live",
                        (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,200), 2)

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

        cv2.imshow("Pellet Inspector", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == 32:  # Space
            frozen_frame = frame.copy()
            is_frozen = True
            last_tick_data = None
        elif key == 27:  # Esc
            is_frozen = False
            in_manual_calib_mode = False
            frozen_frame = manual_frozen_frame = None
            manual_line_start = manual_line_end = None
        elif key == ord('c') and is_frozen:
            calibration_on_frozen = True
        elif key == ord('m'):
            in_manual_calib_mode = True
            manual_frozen_frame = frozen_frame.copy() if frozen_frame else frame.copy()
        elif key == ord('r'):
            is_calibrated = False
            PIXELS_PER_MM = 7.0
            update_ranges()
            print("Calibration reset")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()