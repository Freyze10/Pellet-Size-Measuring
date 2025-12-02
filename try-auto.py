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

# --- HEIGHT COMPENSATION (3mm pellet height vs flat ruler) ---
# Simple cheat: pellets are 3mm higher than ruler, so they appear ~1.5% larger in perspective
HEIGHT_COMPENSATION_FACTOR = 1.215  # Compensate for 3mm height difference

# --- CALIBRATION MODES ---
CALIBRATION_MODE = 'CM'  # Options: 'CM' or 'INCH'

# CM Settings
MIN_LINES_CM = 5  # Need 5 lines (4 gaps)
DIVISOR_CM = 10.0  # 1 cm = 10 mm

# Inch Settings
MIN_LINES_INCH = 4  # Need 4 lines (3 consecutive equal spaces)
DIVISOR_INCH = 25.4  # 1 inch = 25.4 mm

# --- STABILITY & LOCKING SETTINGS ---
CALIBRATION_BUFFER_SIZE = 150
STABILITY_THRESHOLD = 0.3  # IMPROVED: Tighter stability check
RESET_THRESHOLD = 2.5  # IMPROVED: Less sensitive to noise
MAX_MOVEMENT_PIXELS = 50

# --- DETECTION STRICTNESS ---
ASPECT_RATIO_MIN = 2.0
HEIGHT_RATIO_STRICT = 0.85  # IMPROVED: Slightly relaxed from 0.90
MAX_GAP_VARIANCE = 0.08  # IMPROVED: More realistic tolerance (was 0.03)
GAP_OUTLIER_TOLERANCE = 0.20  # IMPROVED: Better outlier detection (was 0.15)
EDGE_MARGIN_PERCENT = 0.30  # NEW: How far from edge to look for lines

# System State
yolo_model = None
is_calibrated = False
calibration_locked = False
calibration_buffer = deque(maxlen=CALIBRATION_BUFFER_SIZE)
current_tick_data = None
calibration_error_msg = ""
locked_zone_coords = None

# NEW: Temporal smoothing buffer
measurement_history = deque(maxlen=10)

# Camera Settings
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 20000


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


def is_within_tolerance(d, l):
    return (DIAMETER_MIN <= d <= DIAMETER_MAX and LENGTH_MIN <= l <= LENGTH_MAX)


def should_process_pellet(d, l):
    if d < 0.5 or l < 0.5: return False
    return (DIAMETER_EXCLUDE_MIN <= d <= DIAMETER_EXCLUDE_MAX and
            LENGTH_EXCLUDE_MIN <= l <= LENGTH_EXCLUDE_MAX)


# ----------------------------------------------------------------------
# IMPROVED: Robust statistical filtering
# ----------------------------------------------------------------------
def remove_outliers_iqr(data, factor=1.5):
    """Remove outliers using Interquartile Range method - more robust than simple tolerance"""
    if len(data) < 4:
        return data

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)

    return [x for x in data if lower_bound <= x <= upper_bound]


def calculate_consistent_gaps(ticks):
    """IMPROVED: More robust gap calculation with better outlier handling"""
    if len(ticks) < 2:
        return None, 0

    ticks_sorted = sorted(ticks, key=lambda x: x['pos'])
    all_gaps = []

    for i in range(len(ticks_sorted) - 1):
        gap = ticks_sorted[i + 1]['pos'] - ticks_sorted[i]['pos']
        all_gaps.append(gap)

    if len(all_gaps) == 0:
        return None, 0

    # IMPROVED: Use IQR method for outlier removal
    clean_gaps = remove_outliers_iqr(all_gaps, factor=1.5)

    if len(clean_gaps) < max(2, len(all_gaps) * 0.6):  # Need at least 60% valid gaps
        return None, 0

    gap_mean = np.mean(clean_gaps)
    gap_std = np.std(clean_gaps)

    # Calculate coefficient of variation (CV) - more meaningful than raw variance
    cv = gap_std / gap_mean if gap_mean > 0 else 1.0

    return gap_mean, cv


# ----------------------------------------------------------------------
# 1. YOLO Detection
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


# ----------------------------------------------------------------------
# 2. IMPROVED DYNAMIC STRUCTURE ANALYSIS
# ----------------------------------------------------------------------
def analyze_structure(frame, bbox):
    global calibration_error_msg

    # Determine thresholds based on current mode
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
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 19, 5)

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

    # IMPROVED: Filter for Major Lines - only at edges to avoid numbers
    # Sort ticks to find edge regions
    ticks_sorted = sorted(all_ticks, key=lambda x: x['pos'])
    roi_size = (cx2 - cx1) if is_horizontal_ruler else (cy2 - cy1)

    # Define edge zones (outer 30% on each side)
    edge_size = roi_size * EDGE_MARGIN_PERCENT

    max_length = max(t['len'] for t in all_ticks)
    major_threshold = max_length * HEIGHT_RATIO_STRICT

    major_ticks = []
    for t in all_ticks:
        # Check if tick is in edge region (to avoid numbers in center)
        if is_horizontal_ruler:
            # For horizontal ruler, check if at TOP or BOTTOM edge
            ty = t['rect'][1]  # y position
            ruler_height = roi.shape[0]
            is_at_top_edge = ty < edge_size
            is_at_bottom_edge = (ty + t['rect'][3]) > (ruler_height - edge_size)
            is_at_edge = is_at_top_edge or is_at_bottom_edge
        else:
            # For vertical ruler, check if at LEFT or RIGHT edge
            tx = t['rect'][0]  # x position
            ruler_width = roi.shape[1]
            is_at_left_edge = tx < edge_size
            is_at_right_edge = (tx + t['rect'][2]) > (ruler_width - edge_size)
            is_at_edge = is_at_left_edge or is_at_right_edge

        # Only consider long lines at edges as MAJOR
        if t['len'] >= major_threshold and is_at_edge:
            t['type'] = 'MAJOR'
            major_ticks.append(t)
        elif t['len'] > (max_length * 0.60):
            t['type'] = 'MEDIUM'
        else:
            t['type'] = 'MINOR'

    if len(major_ticks) < MIN_LINES:
        calibration_error_msg = f"Need {MIN_LINES} {CALIBRATION_MODE} lines (found {len(major_ticks)})"
        return {"px_per_mm": 0, "ticks": all_ticks, "roi_offset": (cx1, cy1)}

    # IMPROVED: Robust gap analysis
    gap_mean, cv = calculate_consistent_gaps(major_ticks)

    if gap_mean is None:
        calibration_error_msg = f"Cannot determine consistent spacing"
        return {"px_per_mm": 0, "ticks": all_ticks, "roi_offset": (cx1, cy1)}

    if cv > MAX_GAP_VARIANCE:
        calibration_error_msg = f"Spacing uneven (CV: {cv * 100:.1f}%)"
        return {"px_per_mm": 0, "ticks": all_ticks, "roi_offset": (cx1, cy1)}

    # Calculate px_per_mm based on current mode divisor
    px_per_mm = gap_mean / DIVISOR

    if px_per_mm < 2 or px_per_mm > 150:
        calibration_error_msg = f"Scale invalid ({px_per_mm:.2f} px/mm)"
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
# 3. Pellet Detection
# ----------------------------------------------------------------------
pellet_history = {}


def detect_pellets(frame, excluded_boxes):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pellets = []
    current_ids = []

    # CHEAT: Compensate for 3mm height difference between ruler and pellet
    effective_px_per_mm = PIXELS_PER_MM * HEIGHT_COMPENSATION_FACTOR

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

        d1 = get_distance(box[0], box[1])
        d2 = get_distance(box[1], box[2])
        raw_w = min(d1, d2)
        raw_h = max(d1, d2)

        p_id = f"{int(cx // 20)}_{int(cy // 20)}"
        current_ids.append(p_id)

        if p_id in pellet_history:
            prev_w, prev_h = pellet_history[p_id]
            s_w = (prev_w * 0.7) + (raw_w * 0.3)
            s_h = (prev_h * 0.7) + (raw_h * 0.3)
        else:
            s_w, s_h = raw_w, raw_h

        pellet_history[p_id] = (s_w, s_h)

        # Use compensated scale for accurate measurements
        d_mm = s_w / effective_px_per_mm
        l_mm = s_h / effective_px_per_mm

        if should_process_pellet(d_mm, l_mm):
            pellets.append({
                'box': box,
                'diameter': d_mm,
                'length': l_mm,
                'is_good': is_within_tolerance(d_mm, l_mm)
            })

    keys = list(pellet_history.keys())
    for k in keys:
        if k not in current_ids: del pellet_history[k]

    return pellets


# ----------------------------------------------------------------------
# 4. Visualization
# ----------------------------------------------------------------------
def draw_ui(frame, yolo_objects, active_zone_box, pellets, tick_data):
    # Draw Objects
    for obj in yolo_objects:
        bx1, by1, bx2, by2 = obj['box']
        color = (0, 255, 0) if (active_zone_box and obj['box'] == active_zone_box) else (100, 100, 100)
        label = f"{obj['name']} [ACTIVE]" if color == (0, 255, 0) else obj['name']
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(frame, label, (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw Ticks
    if tick_data:
        off_x, off_y = tick_data['roi_offset']
        MARKER_VISUAL_LENGTH = 20
        THICKNESS = 2

        for t in tick_data['ticks']:
            rx, ry, rw, rh = t['rect']
            center_x = int(off_x + rx + rw / 2)
            center_y = int(off_y + ry + rh / 2)
            is_vertical_tick = rh > rw

            if is_vertical_tick:
                pt1 = (center_x, center_y - MARKER_VISUAL_LENGTH // 2)
                pt2 = (center_x, center_y + MARKER_VISUAL_LENGTH // 2)
            else:
                pt1 = (center_x - MARKER_VISUAL_LENGTH // 2, center_y)
                pt2 = (center_x + MARKER_VISUAL_LENGTH // 2, center_y)

            if t['type'] == 'MAJOR':
                cv2.line(frame, pt1, pt2, (0, 0, 255), THICKNESS)  # Red
            elif t['type'] == 'MEDIUM':
                cv2.line(frame, pt1, pt2, (255, 255, 0), 1)  # Cyan
            else:
                cv2.line(frame, pt1, pt2, (255, 255, 0), 1)

    # Draw Pellets
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
    cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, 70), (20, 20, 20), -1)

    # Row 1: Calibration Status
    if is_calibrated:
        if calibration_locked:
            msg = f"LOCKED: {PIXELS_PER_MM:.2f} px/mm"
            col = (0, 255, 0)
        else:
            pct = int((len(calibration_buffer) / CALIBRATION_BUFFER_SIZE) * 100)
            msg = f"Stabilizing (5s)... {pct}%"
            col = (0, 255, 255)
    else:
        msg = f"UNCALIBRATED: {calibration_error_msg}" if calibration_error_msg else "UNCALIBRATED"
        col = (0, 0, 255)

    cv2.putText(frame, msg, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

    # Row 2: Mode Status
    mode_text = f"MODE: {CALIBRATION_MODE} (Press 'u' to switch)"
    cv2.putText(frame, mode_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if is_calibrated and not calibration_locked:
        progress = len(calibration_buffer) / CALIBRATION_BUFFER_SIZE
        cv2.rectangle(frame, (500, 20), (500 + int(200 * progress), 40), (0, 255, 255), -1)
        cv2.rectangle(frame, (500, 20), (700, 40), (255, 255, 255), 1)

    return frame


# ----------------------------------------------------------------------
# IMPROVED: Main Logic with better stability
# ----------------------------------------------------------------------
def reset_stabilization():
    global calibration_locked, measurement_history
    if not calibration_locked:
        calibration_buffer.clear()
        measurement_history.clear()


def process_calibration(result):
    global PIXELS_PER_MM, is_calibrated, calibration_locked, locked_zone_coords

    new_px = result['px_per_mm']
    new_coords = result['roi_offset']
    if new_px == 0: return

    # IMPROVED: Add temporal smoothing
    measurement_history.append(new_px)
    if len(measurement_history) >= 3:
        smoothed_px = np.median(measurement_history)  # Use median for robustness
    else:
        smoothed_px = new_px

    if calibration_locked:
        # Check for significant scale change
        if abs(smoothed_px - PIXELS_PER_MM) > RESET_THRESHOLD:
            print(f"⚠ Scale changed! ({PIXELS_PER_MM:.2f} -> {smoothed_px:.2f}) Re-calibrating...")
            calibration_locked = False
            calibration_buffer.clear()
            measurement_history.clear()
            return

        # Check for camera movement
        if locked_zone_coords:
            lx, ly = locked_zone_coords
            nx, ny = new_coords
            dist = math.sqrt((lx - nx) ** 2 + (ly - ny) ** 2)
            if dist > MAX_MOVEMENT_PIXELS:
                print(f"⚠ Camera moved {dist:.1f}px! Re-calibrating...")
                calibration_locked = False
                calibration_buffer.clear()
                measurement_history.clear()
        return

    # IMPROVED: More intelligent buffer management
    if len(calibration_buffer) > 0:
        current_avg = np.mean(calibration_buffer)
        # Use smoothed value and more reasonable threshold
        if abs(smoothed_px - current_avg) > 2.0:  # Changed from 1.0 to 2.0
            print(f"⚠ Measurement unstable, resetting buffer")
            calibration_buffer.clear()
            measurement_history.clear()

    calibration_buffer.append(smoothed_px)
    avg_px = np.mean(calibration_buffer)
    std_dev = np.std(calibration_buffer)

    PIXELS_PER_MM = avg_px
    is_calibrated = True
    update_ranges()

    # IMPROVED: Lock when buffer is full AND stable
    if len(calibration_buffer) == CALIBRATION_BUFFER_SIZE:
        if std_dev < STABILITY_THRESHOLD:
            calibration_locked = True
            locked_zone_coords = new_coords
            print(f"🔒 LOCKED at {avg_px:.2f} px/mm | StdDev: {std_dev:.3f} | CV: {(std_dev / avg_px) * 100:.2f}%")
        else:
            print(f"⚠ Buffer full but unstable (StdDev: {std_dev:.3f}), continuing...")
            # Remove oldest 50 samples to continue stabilizing
            for _ in range(50):
                if len(calibration_buffer) > 0:
                    calibration_buffer.popleft()


def main():
    global current_tick_data, CALIBRATION_MODE, calibration_locked

    if not load_yolo_model(): return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    window_name = "Inspector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("Running...")
    print(f"Height compensation: {HEIGHT_COMPENSATION_FACTOR}x (for 3mm pellet height)")

    while True:
        ret, frame = cap.read()
        if not ret: break

        yolo_objects, active_zone = run_yolo_detection(frame)

        if not active_zone:
            current_tick_data = None
            reset_stabilization()
        else:
            result = analyze_structure(frame, active_zone)
            current_tick_data = result

            if result and result['px_per_mm'] > 0:
                process_calibration(result)
            else:
                reset_stabilization()

        pellets = detect_pellets(frame, yolo_objects)
        frame = draw_ui(frame, yolo_objects, active_zone, pellets, current_tick_data)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('u'):
            # Switch Mode
            if CALIBRATION_MODE == 'CM':
                CALIBRATION_MODE = 'INCH'
            else:
                CALIBRATION_MODE = 'CM'

            # Reset Calibration State
            calibration_locked = False
            calibration_buffer.clear()
            measurement_history.clear()
            is_calibrated = False
            print(f"Switched to {CALIBRATION_MODE} Mode")

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()