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
PIXELS_PER_MM = 10.0
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0

# --- HEIGHT COMPENSATION ---
# Kept at 1.06 (Original "Cheat") to compensate for 3mm height difference
# If you place the ruler on a 3mm shim, change this to 1.0
RULER_HEIGHT_ADJUSTMENT = 1.06

# --- CALIBRATION MODES ---
CALIBRATION_MODE = 'CM'

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
EDGE_MARGIN_PERCENT = 0.30

# System State
yolo_model = None
is_calibrated = False
calibration_locked = False
calibration_buffer = deque(maxlen=CALIBRATION_BUFFER_SIZE)
current_tick_data = None
calibration_error_msg = ""
locked_zone_coords = None
measurement_history = deque(maxlen=10)

# Camera Settings
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 20000


# ----------------------------------------------------------------------
# NEW: STABILIZATION CLASS (The "Anti-Jitter" Logic)
# ----------------------------------------------------------------------
class PelletStabilizer:
    def __init__(self, maxlen=30):
        self.diam_history = deque(maxlen=maxlen)
        self.len_history = deque(maxlen=maxlen)

        # The number currently shown on screen
        self.display_diam = 0.0
        self.display_len = 0.0
        self.is_initialized = False

    def update(self, raw_d, raw_l):
        self.diam_history.append(raw_d)
        self.len_history.append(raw_l)

        # Calculate smooth averages
        avg_d = np.mean(self.diam_history)
        avg_l = np.mean(self.len_history)

        # Initialization: Show number immediately
        if not self.is_initialized:
            self.display_diam = avg_d
            self.display_len = avg_l
            if len(self.diam_history) > 5: self.is_initialized = True
            return self.display_diam, self.display_len

        # LOGIC: Only change the number if the change is REAL (large)
        # If the change is tiny (noise), keep the old number.

        # Check difference between new average and displayed number
        diff_d = abs(avg_d - self.display_diam)
        diff_l = abs(avg_l - self.display_len)

        # 0.05mm threshold. If it moves less than 0.05mm, we ignore it (jitter).
        # If it moves MORE than 0.05mm, we update the display.
        if diff_d > 0.05: self.display_diam = avg_d
        if diff_l > 0.05: self.display_len = avg_l

        return self.display_diam, self.display_len


# Global Dictionary to store stabilizers per pellet ID
pellet_stabilizers = {}


# ----------------------------------------------------------------------
# Range Logic
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
        return True
    except:
        return False


def is_within_tolerance(d, l):
    return (DIAMETER_MIN <= d <= DIAMETER_MAX and LENGTH_MIN <= l <= LENGTH_MAX)


def should_process_pellet(d, l):
    if d < 0.5 or l < 0.5: return False
    return (DIAMETER_EXCLUDE_MIN <= d <= DIAMETER_EXCLUDE_MAX and
            LENGTH_EXCLUDE_MIN <= l <= LENGTH_EXCLUDE_MAX)


def remove_outliers_iqr(data, factor=1.5):
    if len(data) < 4: return data
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    return [x for x in data if (q1 - factor * iqr) <= x <= (q3 + factor * iqr)]


def calculate_consistent_gaps(ticks):
    if len(ticks) < 2: return None, 0
    ticks_sorted = sorted(ticks, key=lambda x: x['pos'])
    all_gaps = []
    for i in range(len(ticks_sorted) - 1):
        gap = ticks_sorted[i + 1]['pos'] - ticks_sorted[i]['pos']
        all_gaps.append(gap)
    if not all_gaps: return None, 0
    clean_gaps = remove_outliers_iqr(all_gaps, factor=1.5)
    if len(clean_gaps) < max(2, len(all_gaps) * 0.6): return None, 0
    gap_mean = np.mean(clean_gaps)
    cv = np.std(clean_gaps) / gap_mean if gap_mean > 0 else 1.0
    return gap_mean, cv


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------
def run_yolo_detection(frame):
    if yolo_model is None: return [], None
    results = yolo_model(frame, conf=0.35, verbose=False)
    best_map = {}
    best_zone = None
    best_conf = 0

    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf, cls_id = float(box.conf[0]), int(box.cls[0])
            name = yolo_model.names[cls_id]
            if name not in best_map or conf > best_map[name]['conf']:
                best_map[name] = {'box': (x1, y1, x2, y2), 'name': name, 'conf': conf}

    all_dets = list(best_map.values())
    for det in all_dets:
        if "mm" in det['name'].lower() or "zone" in det['name'].lower():
            if best_zone is None or det['conf'] > best_conf:
                best_conf = det['conf']
                best_zone = det['box']
        elif best_zone is None and "ruler" in det['name'].lower():
            best_zone = det['box']
    return all_dets, best_zone


def analyze_structure(frame, bbox):
    global calibration_error_msg
    x1, y1, x2, y2 = bbox
    pad = 5
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
    roi = frame[cy1:cy2, cx1:cx2]
    if roi.size == 0: return None

    is_horiz = (cx2 - cx1) > (cy2 - cy1)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 5)

    k_size = (1, max(5, roi.shape[0] // 10)) if is_horiz else (max(5, roi.shape[1] // 10), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, k_size)
    lines = cv2.dilate(cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel), None)

    cnts, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_lines = MIN_LINES_CM if CALIBRATION_MODE == 'CM' else MIN_LINES_INCH
    divisor = DIVISOR_CM if CALIBRATION_MODE == 'CM' else DIVISOR_INCH

    all_ticks = []
    for c in cnts:
        if cv2.contourArea(c) < 5: continue
        tx, ty, tw, th = cv2.boundingRect(c)
        ar = th / tw if is_horiz else tw / th
        if ar > ASPECT_RATIO_MIN:
            pos = tx + tw / 2.0 if is_horiz else ty + th / 2.0
            length = th if is_horiz else tw
            all_ticks.append({'pos': pos, 'len': length, 'rect': (tx, ty, tw, th)})

    if len(all_ticks) < min_lines:
        calibration_error_msg = f"Need {min_lines}+ lines"
        return None

    # Filter Major Ticks
    all_ticks.sort(key=lambda x: x['pos'])
    edge_thresh = ((cx2 - cx1) if is_horiz else (cy2 - cy1)) * EDGE_MARGIN_PERCENT
    max_l = max(t['len'] for t in all_ticks)
    major_ticks = []

    for t in all_ticks:
        if is_horiz:
            at_edge = t['rect'][1] < edge_thresh or (t['rect'][1] + t['rect'][3]) > (roi.shape[0] - edge_thresh)
        else:
            at_edge = t['rect'][0] < edge_thresh or (t['rect'][0] + t['rect'][2]) > (roi.shape[1] - edge_thresh)

        if t['len'] >= max_l * HEIGHT_RATIO_STRICT and at_edge:
            t['type'] = 'MAJOR'
            major_ticks.append(t)
        elif t['len'] > max_l * 0.6:
            t['type'] = 'MEDIUM'
        else:
            t['type'] = 'MINOR'

    if len(major_ticks) < min_lines:
        calibration_error_msg = f"Need {min_lines} major lines"
        return {"px_per_mm": 0, "ticks": all_ticks, "roi_offset": (cx1, cy1)}

    mean_gap, cv = calculate_consistent_gaps(major_ticks)
    if mean_gap is None or cv > MAX_GAP_VARIANCE:
        calibration_error_msg = f"Uneven spacing (CV: {cv:.2f})"
        return {"px_per_mm": 0, "ticks": all_ticks, "roi_offset": (cx1, cy1)}

    px_mm = (mean_gap / divisor) * RULER_HEIGHT_ADJUSTMENT
    return {"px_per_mm": px_mm, "ticks": all_ticks, "roi_offset": (cx1, cy1), "spacing_variance": cv}


# ----------------------------------------------------------------------
# 3. Pellet Detection (With STABILITY)
# ----------------------------------------------------------------------
def detect_pellets(frame, excluded_boxes):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pellets = []
    current_ids = []

    for cnt in contours:
        if not (MIN_CONTOUR_AREA <= cv2.contourArea(cnt) <= MAX_CONTOUR_AREA): continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        box = np.intp(cv2.boxPoints(rect))

        is_ignored = False
        for ex_obj in excluded_boxes:
            bx1, by1, bx2, by2 = ex_obj['box']
            if (bx1 - 5) <= cx <= (bx2 + 5) and (by1 - 5) <= cy <= (by2 + 5):
                is_ignored = True;
                break
        if is_ignored: continue

        dim1, dim2 = min(w, h), max(w, h)
        raw_d_mm = dim1 / PIXELS_PER_MM
        raw_l_mm = dim2 / PIXELS_PER_MM

        # --- STABILIZATION ---
        p_id = f"{int(cx // 30)}_{int(cy // 30)}"
        current_ids.append(p_id)

        if p_id not in pellet_stabilizers:
            pellet_stabilizers[p_id] = PelletStabilizer()

        # Get smoothed values
        final_d, final_l = pellet_stabilizers[p_id].update(raw_d_mm, raw_l_mm)

        if should_process_pellet(final_d, final_l):
            pellets.append({
                'box': box,
                'diameter': final_d,
                'length': final_l,
                'is_good': is_within_tolerance(final_d, final_l)
            })

    # Cleanup old trackers
    for k in list(pellet_stabilizers.keys()):
        if k not in current_ids: del pellet_stabilizers[k]

    return pellets


# ----------------------------------------------------------------------
# 4. Visualization (RESTORED TO GREEN/RED)
# ----------------------------------------------------------------------
def draw_ui(frame, yolo_objects, active_zone, pellets, tick_data):
    # Draw Yolo Zones
    for obj in yolo_objects:
        bx1, by1, bx2, by2 = obj['box']
        col = (0, 255, 0) if (active_zone and obj['box'] == active_zone) else (100, 100, 100)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, 2)
        cv2.putText(frame, obj['name'], (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    # Draw Ticks
    if tick_data:
        off_x, off_y = tick_data['roi_offset']
        for t in tick_data['ticks']:
            rx, ry, rw, rh = t['rect']
            cx, cy = int(off_x + rx + rw / 2), int(off_y + ry + rh / 2)
            color = (0, 0, 255) if t['type'] == 'MAJOR' else (255, 255, 0)
            length = 25 if t['type'] == 'MAJOR' else 12
            if rh > rw:
                cv2.line(frame, (cx, cy - length // 2), (cx, cy + length // 2), color, 2 if t['type'] == 'MAJOR' else 1)
            else:
                cv2.line(frame, (cx - length // 2, cy), (cx + length // 2, cy), color, 2 if t['type'] == 'MAJOR' else 1)

    # Draw Pellets
    for p in pellets:
        box = p['box']
        # COLOR LOGIC: Green if Good, Red if Bad
        color = (0, 255, 0) if p['is_good'] else (0, 0, 255)

        cv2.drawContours(frame, [box], 0, color, 2)

        M = cv2.moments(box)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else box[0][0]
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else box[0][1]

        txt = f"{p['diameter']:.2f} x {p['length']:.2f}"

        # Shadow
        cv2.putText(frame, txt, (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        # Main Text (White)
        cv2.putText(frame, txt, (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if not p['is_good']:
            cv2.putText(frame, "!", (cx - 45, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Status Bar
    cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, 70), (20, 20, 20), -1)
    if is_calibrated:
        if calibration_locked:
            msg = f"CALIBRATED: {PIXELS_PER_MM:.2f} px/mm"
            col = (0, 255, 0)
        else:
            msg = f"Stabilizing... {int(len(calibration_buffer) / CALIBRATION_BUFFER_SIZE * 100)}%"
            col = (0, 255, 255)
    else:
        msg = calibration_error_msg if calibration_error_msg else "UNCALIBRATED"
        col = (0, 0, 255)

    cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    cv2.putText(frame, f"MODE: {CALIBRATION_MODE}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


# ----------------------------------------------------------------------
# Main Logic
# ----------------------------------------------------------------------
def process_calibration(result):
    global PIXELS_PER_MM, is_calibrated, calibration_locked

    # Filter calibration jitter
    measurement_history.append(result['px_per_mm'])
    smoothed_px = np.median(measurement_history) if len(measurement_history) >= 5 else result['px_per_mm']

    if calibration_locked:
        if abs(smoothed_px - PIXELS_PER_MM) > RESET_THRESHOLD:
            calibration_locked = False
            calibration_buffer.clear()
        return

    calibration_buffer.append(smoothed_px)

    if len(calibration_buffer) == CALIBRATION_BUFFER_SIZE:
        std = np.std(calibration_buffer)
        if std < STABILITY_THRESHOLD:
            PIXELS_PER_MM = np.mean(calibration_buffer)
            is_calibrated = True
            calibration_locked = True
            print(f"LOCKED CALIBRATION: {PIXELS_PER_MM:.2f}")
        else:
            calibration_buffer.popleft()


def main():
    global CALIBRATION_MODE, calibration_locked, is_calibrated
    if not load_yolo_model(): return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        yolo_objects, active_zone = run_yolo_detection(frame)

        if active_zone:
            res = analyze_structure(frame, active_zone)
            current_tick_data = res
            if res and res['px_per_mm'] > 0: process_calibration(res)
        else:
            current_tick_data = None
            if not calibration_locked: calibration_buffer.clear()

        pellets = detect_pellets(frame, yolo_objects)
        frame = draw_ui(frame, yolo_objects, active_zone, pellets, current_tick_data)

        cv2.imshow("Inspector", frame)
        k = cv2.waitKey(1)
        if k == ord('q'): break
        if k == ord('u'):
            CALIBRATION_MODE = 'INCH' if CALIBRATION_MODE == 'CM' else 'CM'
            calibration_locked = False
            calibration_buffer.clear()
            pellet_stabilizers.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()