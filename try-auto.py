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
# Set to 1.0 assuming you are calibrating with a ruler raised 3mm (shimmed)
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

# System State
yolo_model = None
is_calibrated = False
calibration_locked = False
calibration_buffer = deque(maxlen=CALIBRATION_BUFFER_SIZE)
current_tick_data = None
calibration_error_msg = ""
locked_zone_coords = None
measurement_history = deque(maxlen=10)

# NEW: Calibration Mode State
calibration_active = False  # Whether we're currently in calibration mode

# Manual Calibration State
in_manual_calib_mode = False
MANUAL_REFERENCE_LENGTH_MM = 76.2
manual_line_start = None
manual_line_end = None
manual_frozen_frame = None
is_dragging = False

# Button rectangles for manual calibration
MANUAL_PANEL_X, MANUAL_PANEL_Y = 10, 80
MANUAL_PANEL_W, MANUAL_PANEL_H = 380, 280

RESET_BTN = (MANUAL_PANEL_X + 20, MANUAL_PANEL_Y + 200, 100, 40)
APPLY_BTN = (MANUAL_PANEL_X + 140, MANUAL_PANEL_Y + 200, 100, 40)
CANCEL_BTN = (MANUAL_PANEL_X + 260, MANUAL_PANEL_Y + 200, 100, 40)

# Camera Settings
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 20000


# ----------------------------------------------------------------------
# NEW: STABILIZATION LOGIC
# ----------------------------------------------------------------------
class PelletStabilizer:
    def __init__(self, maxlen=30):
        # Deep Buffering: Remembers last 30 measurements
        self.diam_history = deque(maxlen=maxlen)
        self.len_history = deque(maxlen=maxlen)

        # Display state
        self.display_diam = 0.0
        self.display_len = 0.0
        self.locked = False

    def update(self, raw_d, raw_l):
        self.diam_history.append(raw_d)
        self.len_history.append(raw_l)

        # Calculate averages
        avg_d = np.mean(self.diam_history)
        avg_l = np.mean(self.len_history)

        # Calculate Volatility (Standard Deviation)
        # We check if the measurements are jittering or stable
        if len(self.diam_history) > 5:
            std_d = np.std(self.diam_history)
            std_l = np.std(self.len_history)
        else:
            std_d, std_l = 1.0, 1.0  # High volatility if not enough data

        # LOCKING LOGIC
        if self.locked:
            # If locked, we only UNLOCK if the pellet physically moves significantly.
            # Movement Threshold: 0.2mm difference from the locked value
            if abs(avg_d - self.display_diam) > 0.2 or abs(avg_l - self.display_len) > 0.2:
                self.locked = False
                self.diam_history.clear()  # Reset buffer on movement
                self.len_history.clear()
                # Return current raw value temporarily while buffer rebuilds
                return raw_d, raw_l
            else:
                # Return the locked, solid number
                return self.display_diam, self.display_len
        else:
            # Not locked: update display with the smooth average
            self.display_diam = avg_d
            self.display_len = avg_l

            # Check if we SHOULD lock
            # Requirement: Buffer full (30 frames) AND very low volatility (stable)
            if len(self.diam_history) >= 30 and std_d < 0.05 and std_l < 0.05:
                self.locked = True

            return self.display_diam, self.display_len


# Dictionary to hold stabilizers for each specific pellet ID
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
        print(f"✓ YOLO model loaded: {YOLO_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        return False


def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ----------------------------------------------------------------------
# Mouse Callback for Manual Calibration
# ----------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global manual_line_start, manual_line_end, is_dragging
    global in_manual_calib_mode, PIXELS_PER_MM

    if not in_manual_calib_mode:
        return

    def in_rect(px, py, rect):
        rx, ry, rw, rh = rect
        return rx <= px <= rx + rw and ry <= py <= ry + rh

    # Handle button clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at ({x}, {y})")  # Debug

        if in_rect(x, y, RESET_BTN):
            print("RESET button clicked")
            manual_line_start = None
            manual_line_end = None
            is_dragging = False
            return
        elif in_rect(x, y, APPLY_BTN):
            print("APPLY button clicked")
            if manual_line_start and manual_line_end:
                dx = manual_line_end[0] - manual_line_start[0]
                dy = manual_line_end[1] - manual_line_start[1]
                pixel_distance = math.sqrt(dx ** 2 + dy ** 2)

                if pixel_distance > 10:
                    # Apply the +0.35 adjustment to match auto-calibration
                    PIXELS_PER_MM = (pixel_distance / MANUAL_REFERENCE_LENGTH_MM) + 0.35
                    update_ranges()
                    print(
                        f"Manual Calibration: {PIXELS_PER_MM:.4f} px/mm (base: {pixel_distance / MANUAL_REFERENCE_LENGTH_MM:.4f} + 0.35)")

                in_manual_calib_mode = False
                manual_line_start = None
                manual_line_end = None
                is_dragging = False
            return
        elif in_rect(x, y, CANCEL_BTN):
            print("CANCEL button clicked")
            in_manual_calib_mode = False
            manual_line_start = None
            manual_line_end = None
            is_dragging = False
            return

        if not in_rect(x, y, (MANUAL_PANEL_X, MANUAL_PANEL_Y, MANUAL_PANEL_W, MANUAL_PANEL_H)):
            print(f"Starting line at ({x}, {y})")
            manual_line_start = (x, y)
            manual_line_end = (x, y)
            is_dragging = True

    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        manual_line_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if is_dragging:
            print(f"Ending line at ({x}, {y})")
            manual_line_end = (x, y)
            is_dragging = False


def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def is_within_tolerance(d, l):
    return (DIAMETER_MIN <= d <= DIAMETER_MAX and LENGTH_MIN <= l <= LENGTH_MAX)


def should_process_pellet(d, l):
    if d < 0.5 or l < 0.5: return False
    return (DIAMETER_EXCLUDE_MIN <= d <= DIAMETER_EXCLUDE_MAX and
            LENGTH_EXCLUDE_MIN <= l <= LENGTH_EXCLUDE_MAX)


# ----------------------------------------------------------------------
# Logic
# ----------------------------------------------------------------------
def remove_outliers_iqr(data, factor=1.5):
    if len(data) < 4: return data
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)
    return [x for x in data if lower_bound <= x <= upper_bound]


def calculate_consistent_gaps(ticks):
    if len(ticks) < 2: return None, 0
    ticks_sorted = sorted(ticks, key=lambda x: x['pos'])
    all_gaps = []
    for i in range(len(ticks_sorted) - 1):
        gap = ticks_sorted[i + 1]['pos'] - ticks_sorted[i]['pos']
        all_gaps.append(gap)
    if len(all_gaps) == 0: return None, 0
    clean_gaps = remove_outliers_iqr(all_gaps, factor=1.5)
    if len(clean_gaps) < max(2, len(all_gaps) * 0.6): return None, 0
    gap_mean = np.mean(clean_gaps)
    gap_std = np.std(clean_gaps)
    cv = gap_std / gap_mean if gap_mean > 0 else 1.0
    return gap_mean, cv


# ----------------------------------------------------------------------
# Detection & Analysis
# ----------------------------------------------------------------------
def run_yolo_detection(frame):
    if yolo_model is None: return [], None
    results = yolo_model(frame, conf=0.35, verbose=False)
    best_detections_map = {}
    best_zone_box = None
    overall_best_conf = 0

    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = yolo_model.names[cls_id]
            if name not in best_detections_map or conf > best_detections_map[name]['conf']:
                best_detections_map[name] = {'box': (x1, y1, x2, y2), 'name': name, 'conf': conf}

    all_detections = list(best_detections_map.values())
    for det in all_detections:
        is_preferred = "mm" in det['name'].lower() or "zone" in det['name'].lower()
        if is_preferred:
            if best_zone_box is None or det['conf'] > overall_best_conf:
                overall_best_conf = det['conf']
                best_zone_box = det['box']
        elif best_zone_box is None and "ruler" in det['name'].lower():
            best_zone_box = det['box']
    return all_detections, best_zone_box


def analyze_structure(frame, bbox):
    global calibration_error_msg
    if CALIBRATION_MODE == 'CM':
        MIN_LINES = MIN_LINES_CM
        DIVISOR = DIVISOR_CM
    else:
        MIN_LINES = MIN_LINES_INCH
        DIVISOR = DIVISOR_INCH

    x1, y1, x2, y2 = bbox
    h_img, w_img = frame.shape[:2]
    pad = 5
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w_img, x2 + pad), min(h_img, y2 + pad)
    roi = frame[cy1:cy2, cx1:cx2]
    if roi.size == 0: return None

    is_horizontal_ruler = (cx2 - cx1) > (cy2 - cy1)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 5)

    if is_horizontal_ruler:
        k_height = max(5, roi.shape[0] // 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_height))
    else:
        k_width = max(5, roi.shape[1] // 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_width, 1))

    lines_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    lines_img = cv2.dilate(lines_img, None, iterations=1)
    contours, _ = cv2.findContours(lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_ticks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5: continue
        tx, ty, tw, th = cv2.boundingRect(cnt)
        if is_horizontal_ruler:
            aspect_ratio = th / float(tw) if tw > 0 else 0
            if aspect_ratio > ASPECT_RATIO_MIN:
                pos = tx + tw / 2.0
                all_ticks.append({'pos': pos, 'len': th, 'rect': (tx, ty, tw, th)})
        else:
            aspect_ratio = tw / float(th) if th > 0 else 0
            if aspect_ratio > ASPECT_RATIO_MIN:
                pos = ty + th / 2.0
                all_ticks.append({'pos': pos, 'len': tw, 'rect': (tx, ty, tw, th)})

    if len(all_ticks) < MIN_LINES:
        calibration_error_msg = f"Need {MIN_LINES}+ lines (found {len(all_ticks)})"
        return None

    ticks_sorted = sorted(all_ticks, key=lambda x: x['pos'])
    roi_size = (cx2 - cx1) if is_horizontal_ruler else (cy2 - cy1)
    edge_size = roi_size * EDGE_MARGIN_PERCENT
    max_length = max(t['len'] for t in all_ticks)
    major_threshold = max_length * HEIGHT_RATIO_STRICT

    major_ticks = []
    for t in all_ticks:
        if is_horizontal_ruler:
            ty = t['rect'][1]
            ruler_height = roi.shape[0]
            is_at_edge = (ty < edge_size) or ((ty + t['rect'][3]) > (ruler_height - edge_size))
        else:
            tx = t['rect'][0]
            ruler_width = roi.shape[1]
            is_at_edge = (tx < edge_size) or ((tx + t['rect'][2]) > (ruler_width - edge_size))

        if t['len'] >= major_threshold and is_at_edge:
            t['type'] = 'MAJOR'
            major_ticks.append(t)
        elif t['len'] > (max_length * 0.60):
            t['type'] = 'MEDIUM'
        else:
            t['type'] = 'MINOR'

    if len(major_ticks) < MIN_LINES:
        calibration_error_msg = f"Need {MIN_LINES} {CALIBRATION_MODE} lines"
        return {"px_per_mm": 0, "ticks": all_ticks, "roi_offset": (cx1, cy1)}

    gap_mean, cv = calculate_consistent_gaps(major_ticks)
    if gap_mean is None or cv > MAX_GAP_VARIANCE:
        calibration_error_msg = f"Spacing uneven (CV: {cv * 100:.1f}%)"
        return {"px_per_mm": 0, "ticks": all_ticks, "roi_offset": (cx1, cy1)}

    px_per_mm = (gap_mean / DIVISOR) * RULER_HEIGHT_ADJUSTMENT
    if px_per_mm < 2 or px_per_mm > 150:
        calibration_error_msg = f"Scale invalid ({px_per_mm:.2f})"
        return None

    calibration_error_msg = ""
    return {
        "px_per_mm": px_per_mm,
        "ticks": all_ticks,
        "roi_offset": (cx1, cy1),
        "major_count": len(major_ticks),
        "spacing_variance": cv
    }


# ----------------------------------------------------------------------
# 3. Pellet Detection (MODIFIED WITH DEEP BUFFERING & LOCKING)
# ----------------------------------------------------------------------
def detect_pellets(frame, excluded_boxes):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
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
                is_ignored = True
                break
        if is_ignored: continue

        d1, d2 = get_distance(box[0], box[1]), get_distance(box[1], box[2])
        raw_w, raw_h = min(d1, d2), max(d1, d2)

        # Spatial ID for tracking specific pellets
        p_id = f"{int(cx // 20)}_{int(cy // 20)}"
        current_ids.append(p_id)

        # Raw MM values
        d_mm_raw = raw_w / PIXELS_PER_MM
        l_mm_raw = raw_h / PIXELS_PER_MM

        # --- NEW STABILIZATION LOGIC ---
        if p_id not in pellet_stabilizers:
            pellet_stabilizers[p_id] = PelletStabilizer()  # Create new stabilizer for this pellet

        # Feed raw values into stabilizer, get back locked/smoothed values
        final_d, final_l = pellet_stabilizers[p_id].update(d_mm_raw, l_mm_raw)

        if should_process_pellet(final_d, final_l):
            pellets.append({
                'box': box,
                'diameter': final_d,
                'length': final_l,
                'is_good': is_within_tolerance(final_d, final_l)
            })

    # Cleanup old stabilizers for pellets that disappeared
    keys = list(pellet_stabilizers.keys())
    for k in keys:
        if k not in current_ids: del pellet_stabilizers[k]

    return pellets


# ----------------------------------------------------------------------
# 4. Visualization
# ----------------------------------------------------------------------
def draw_manual_calibration_mode(frame):
    """Draw the manual ruler calibration interface"""
    overlay = frame.copy()

    cv2.rectangle(overlay, (MANUAL_PANEL_X, MANUAL_PANEL_Y),
                  (MANUAL_PANEL_X + MANUAL_PANEL_W, MANUAL_PANEL_Y + MANUAL_PANEL_H),
                  (30, 30, 50), -1)
    cv2.rectangle(overlay, (MANUAL_PANEL_X, MANUAL_PANEL_Y),
                  (MANUAL_PANEL_X + MANUAL_PANEL_W, MANUAL_PANEL_Y + MANUAL_PANEL_H),
                  (100, 150, 255), 3)

    cv2.putText(overlay, "MANUAL CALIBRATION", (MANUAL_PANEL_X + 65, MANUAL_PANEL_Y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    instructions = [
        "1. Place a ruler in camera view",
        "2. Click and drag to match the",
        "   reference line (3 inch / 7.62 cm)",
        "3. Click APPLY when aligned"
    ]

    y_offset = MANUAL_PANEL_Y + 60
    for instr in instructions:
        cv2.putText(overlay, instr, (MANUAL_PANEL_X + 20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_offset += 20

    cv2.putText(overlay, "Reference: 3 inch (76.2 mm)",
                (MANUAL_PANEL_X + 60, MANUAL_PANEL_Y + 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)

    # Buttons
    cv2.rectangle(overlay, (RESET_BTN[0], RESET_BTN[1]),
                  (RESET_BTN[0] + RESET_BTN[2], RESET_BTN[1] + RESET_BTN[3]),
                  (50, 50, 200), -1)
    cv2.putText(overlay, "RESET", (RESET_BTN[0] + 15, RESET_BTN[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    apply_enabled = manual_line_start and manual_line_end
    apply_color = (0, 200, 0) if apply_enabled else (100, 100, 100)
    cv2.rectangle(overlay, (APPLY_BTN[0], APPLY_BTN[1]),
                  (APPLY_BTN[0] + APPLY_BTN[2], APPLY_BTN[1] + APPLY_BTN[3]),
                  apply_color, -1)
    cv2.putText(overlay, "APPLY", (APPLY_BTN[0] + 15, APPLY_BTN[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.rectangle(overlay, (CANCEL_BTN[0], CANCEL_BTN[1]),
                  (CANCEL_BTN[0] + CANCEL_BTN[2], CANCEL_BTN[1] + CANCEL_BTN[3]),
                  (100, 100, 100), -1)
    cv2.putText(overlay, "CANCEL", (CANCEL_BTN[0] + 10, CANCEL_BTN[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(overlay, f"Current: {PIXELS_PER_MM:.2f} px/mm",
                (MANUAL_PANEL_X + 20, MANUAL_PANEL_Y + 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2)

    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    # Draw the reference line with tick marks
    if manual_line_start and manual_line_end:
        cv2.line(frame, manual_line_start, manual_line_end, (0, 255, 255), 2)

        # Draw crosshairs at start
        cv2.line(frame, (manual_line_start[0] - 10, manual_line_start[1]),
                 (manual_line_start[0] + 10, manual_line_start[1]), (0, 0, 255), 2)
        cv2.line(frame, (manual_line_start[0], manual_line_start[1] - 10),
                 (manual_line_start[0], manual_line_start[1] + 10), (0, 0, 255), 2)

        # Draw crosshairs at end
        cv2.line(frame, (manual_line_end[0] - 10, manual_line_end[1]),
                 (manual_line_end[0] + 10, manual_line_end[1]), (0, 0, 255), 2)
        cv2.line(frame, (manual_line_end[0], manual_line_end[1] - 10),
                 (manual_line_end[0], manual_line_end[1] + 10), (0, 0, 255), 2)

        dx = manual_line_end[0] - manual_line_start[0]
        dy = manual_line_end[1] - manual_line_start[1]
        length = math.sqrt(dx ** 2 + dy ** 2)

        if length > 10:
            angle_rad = math.atan2(dy, dx)

            def draw_tick(distance_mm, tick_length, thickness=1, color=(180, 180, 180)):
                t = distance_mm / MANUAL_REFERENCE_LENGTH_MM
                x = manual_line_start[0] + dx * t
                y = manual_line_start[1] + dy * t

                px = int(tick_length * math.sin(angle_rad))
                py = int(-tick_length * math.cos(angle_rad))

                pt1 = (int(x - px), int(y - py))
                pt2 = (int(x + px), int(y + py))
                cv2.line(frame, pt1, pt2, color, thickness)

            # Draw centimeter marks
            for cm in range(1, 8):
                draw_tick(cm * 10, tick_length=12, thickness=2, color=(255, 255, 255))
                t = (cm * 10) / MANUAL_REFERENCE_LENGTH_MM
                x = manual_line_start[0] + dx * t
                y = manual_line_start[1] + dy * t
                label_x = int(x + 25 * math.sin(angle_rad))
                label_y = int(y - 25 * math.cos(angle_rad))
                cv2.putText(frame, str(cm), (label_x - 8, label_y + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # Draw half-centimeter marks
            for half_cm in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]:
                draw_tick(half_cm * 10, tick_length=10, thickness=1, color=(200, 200, 200))

        # Display length
        mid_x = (manual_line_start[0] + manual_line_end[0]) // 2
        mid_y = (manual_line_start[1] + manual_line_end[1]) // 2
        angle = math.atan2(dy, dx)
        text_offset_x = int(-20 * math.sin(angle))
        text_offset_y = int(20 * math.cos(angle))

        cv2.putText(frame, f"{length:.1f} px",
                    (mid_x + text_offset_x, mid_y + text_offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def draw_ui(frame, yolo_objects, active_zone_box, pellets, tick_data):
    # Draw Objects (only in calibration mode)
    if calibration_active:
        for obj in yolo_objects:
            bx1, by1, bx2, by2 = obj['box']
            color = (0, 255, 0) if (active_zone_box and obj['box'] == active_zone_box) else (100, 100, 100)
            label = f"{obj['name']}"
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(frame, label, (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw Ticks (only in calibration mode)
    if calibration_active and tick_data:
        off_x, off_y = tick_data['roi_offset']

        # --- CONFIG FOR UNIFORM VISUALIZATION ---
        VISUAL_LEN_MAJOR = 25  # Fixed length for major ticks
        VISUAL_LEN_MINOR = 12  # Fixed length for minor ticks
        COLOR_MAJOR = (0, 0, 255)  # Red for Major
        COLOR_MINOR = (255, 255, 0)  # Cyan/Yellow for Minor
        THICK_MAJOR = 2
        THICK_MINOR = 1

        for t in tick_data['ticks']:
            rx, ry, rw, rh = t['rect']
            center_x = int(off_x + rx + rw / 2)
            center_y = int(off_y + ry + rh / 2)
            is_vertical_tick = rh > rw

            # 1. Determine Style
            if t['type'] == 'MAJOR':
                length = VISUAL_LEN_MAJOR
                color = COLOR_MAJOR
                thick = THICK_MAJOR
            else:
                length = VISUAL_LEN_MINOR
                color = COLOR_MINOR
                thick = THICK_MINOR

            # 2. Draw Uniform Lines (T-like shape)
            if is_vertical_tick:
                # Tick is tall (|), draw vertical line centered
                pt1 = (center_x, center_y - length // 2)
                pt2 = (center_x, center_y + length // 2)
            else:
                # Tick is wide (-), draw horizontal line centered
                pt1 = (center_x - length // 2, center_y)
                pt2 = (center_x + length // 2, center_y)

            cv2.line(frame, pt1, pt2, color, thick)

    # Draw Pellets (only in main mode)
    if not calibration_active:
        for p in pellets:
            box = p['box']
            color = (0, 255, 0) if p['is_good'] else (0, 0, 255)
            cv2.drawContours(frame, [box], 0, color, 1)

            M = cv2.moments(box)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            else:
                cx, cy = box[0]

            txt = f"{p['diameter']:.2f}x{p['length']:.2f}"
            cv2.putText(frame, txt, (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(frame, txt, (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if not p['is_good']:
                cv2.putText(frame, "!", (cx - 45, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Status Bar
    cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, 90), (20, 20, 20), -1)

    if calibration_active:
        # Calibration mode status
        if is_calibrated:
            if calibration_locked:
                msg = f"CALIBRATION COMPLETE: {PIXELS_PER_MM:.2f} px/mm"
                col = (0, 255, 0)
            else:
                pct = int((len(calibration_buffer) / CALIBRATION_BUFFER_SIZE) * 100)
                msg = f"Calibrating... {pct}%"
                col = (0, 255, 255)
        else:
            msg = f"CALIBRATION MODE: {calibration_error_msg}" if calibration_error_msg else "Place ruler in view"
            col = (0, 165, 255)

        cv2.putText(frame, msg, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        mode_text = f"MODE: {CALIBRATION_MODE} | Press 'u' to switch | Press 'q' to exit"
        cv2.putText(frame, mode_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if is_calibrated and not calibration_locked:
            progress = len(calibration_buffer) / CALIBRATION_BUFFER_SIZE
            cv2.rectangle(frame, (500, 20), (500 + int(200 * progress), 40), (0, 255, 255), -1)
            cv2.rectangle(frame, (500, 20), (700, 40), (255, 255, 255), 1)
    else:
        # Main screen status
        if is_calibrated:
            msg = f"CALIBRATED: {PIXELS_PER_MM:.2f} px/mm"
            col = (0, 255, 0)
        else:
            msg = "NOT CALIBRATED - Press 'a' to calibrate"
            col = (0, 0, 255)

        cv2.putText(frame, msg, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        info_text = "Press 'a' to calibrate | Press 'm' for manual | Press 'q' to exit"
        cv2.putText(frame, info_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Display pellet count in main mode
        if pellets:
            good_count = sum(1 for p in pellets if p['is_good'])
            bad_count = len(pellets) - good_count

            # Count display on the right side
            count_x = DESIRED_WIDTH - 350
            cv2.putText(frame, f"IN SPEC: {good_count}", (count_x, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"OUT OF SPEC: {bad_count}", (count_x, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def process_calibration(result):
    global PIXELS_PER_MM, is_calibrated, calibration_locked, locked_zone_coords, calibration_active
    new_px = result['px_per_mm']
    new_coords = result['roi_offset']
    if new_px == 0: return

    measurement_history.append(new_px)
    smoothed_px = np.median(measurement_history) if len(measurement_history) >= 3 else new_px

    if calibration_locked:
        # Check if calibration just finished - return to main screen
        if len(calibration_buffer) == CALIBRATION_BUFFER_SIZE:
            print(f"✓ Calibration complete! Returning to main screen...")
            calibration_active = False
        return

    if len(calibration_buffer) > 0 and abs(smoothed_px - np.mean(calibration_buffer)) > 2.0:
        calibration_buffer.clear()
        measurement_history.clear()

    calibration_buffer.append(smoothed_px)
    avg_px = np.mean(calibration_buffer)
    std_dev = np.std(calibration_buffer)
    PIXELS_PER_MM = avg_px + 0.35
    is_calibrated = True
    update_ranges()

    if len(calibration_buffer) == CALIBRATION_BUFFER_SIZE:
        if std_dev < STABILITY_THRESHOLD:
            calibration_locked = True
            locked_zone_coords = new_coords
            print(f"🔒 LOCKED at {avg_px:.2f} px/mm")
            # Auto-return to main screen after 1 second
            time.sleep(1)
            calibration_active = False
        else:
            for _ in range(50):
                if len(calibration_buffer) > 0: calibration_buffer.popleft()


def main():
    global current_tick_data, CALIBRATION_MODE, calibration_locked, is_calibrated, calibration_active
    global in_manual_calib_mode, manual_frozen_frame

    if not load_yolo_model(): return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    window_name = "Inspector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Handle manual calibration mode
        if in_manual_calib_mode:
            if manual_frozen_frame is None:
                manual_frozen_frame = frame.copy()
            display_frame = manual_frozen_frame.copy()

            # Draw the UI with manual calibration interface
            yolo_objects = []
            active_zone = None
            pellets = []
            display_frame = draw_ui(display_frame, yolo_objects, active_zone, pellets, None)
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('m'):
                # Exit manual calibration
                in_manual_calib_mode = False
                manual_frozen_frame = None
                manual_line_start = None
                manual_line_end = None
                is_dragging = False

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue

        # Reset frozen frame when not in manual mode
        manual_frozen_frame = None

        # Only run YOLO detection and calibration when in auto-calibration mode
        if calibration_active:
            yolo_objects, active_zone = run_yolo_detection(frame)

            if not active_zone:
                current_tick_data = None
                if not calibration_locked: calibration_buffer.clear()
            else:
                result = analyze_structure(frame, active_zone)
                current_tick_data = result
                if result and result['px_per_mm'] > 0:
                    process_calibration(result)
                elif not calibration_locked:
                    calibration_buffer.clear()
        else:
            yolo_objects = []
            active_zone = None
            current_tick_data = None

        # Detect pellets only in main mode
        pellets = detect_pellets(frame, yolo_objects) if not calibration_active else []

        frame = draw_ui(frame, yolo_objects, active_zone, pellets, current_tick_data)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('a'):
            # Toggle auto calibration mode
            if not in_manual_calib_mode:  # Prevent switching while in manual mode
                calibration_active = not calibration_active
                if calibration_active:
                    print("🔧 Entering auto-calibration mode...")
                    # Reset calibration state
                    calibration_locked = False
                    calibration_buffer.clear()
                    measurement_history.clear()
                    is_calibrated = False
                else:
                    print("📊 Returning to main screen...")
        elif key == ord('m'):
            # Toggle manual calibration mode
            if not calibration_active:  # Prevent switching while in auto-calibration mode
                in_manual_calib_mode = not in_manual_calib_mode
                if in_manual_calib_mode:
                    print("📏 Entering manual calibration mode...")
                    manual_line_start = None
                    manual_line_end = None
                    is_dragging = False
                    manual_frozen_frame = None
                else:
                    print("📊 Exiting manual calibration...")
                    manual_line_start = None
                    manual_line_end = None
                    is_dragging = False
                    manual_frozen_frame = None
        elif key == ord('u') and calibration_active:
            # Only allow mode switching during calibration
            CALIBRATION_MODE = 'INCH' if CALIBRATION_MODE == 'CM' else 'CM'
            calibration_locked = False
            calibration_buffer.clear()
            measurement_history.clear()
            pellet_stabilizers.clear()
            is_calibrated = False

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()