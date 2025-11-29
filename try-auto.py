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

# Stability & Accuracy Settings
CALIBRATION_BUFFER_SIZE = 30
STABILITY_THRESHOLD = 0.5
RESET_THRESHOLD = 3.0

# STRICT MEASUREMENT RULES
MAX_LENGTH_VARIANCE = 0.15  # 15% max difference in tick lengths allowed
MAX_GAP_VARIANCE = 0.05  # 5% max difference in spacing allowed (Strict!)

# System State
yolo_model = None
is_calibrated = False
calibration_locked = False
calibration_buffer = deque(maxlen=CALIBRATION_BUFFER_SIZE)
current_tick_data = None
calibration_error_msg = ""  # To tell user why it failed

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
# 1. YOLO Detection
# ----------------------------------------------------------------------
def run_yolo_detection(frame):
    if yolo_model is None: return [], None

    results = yolo_model(frame, conf=0.35, verbose=False)

    best_detections_map = {}
    calibration_zone = None
    overall_best_conf = 0
    best_zone_box = None

    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = yolo_model.names[cls_id]

            if name not in best_detections_map or conf > best_detections_map[name]['conf']:
                best_detections_map[name] = {
                    'box': (x1, y1, x2, y2),
                    'name': name,
                    'conf': conf
                }

    all_detections = list(best_detections_map.values())

    for det in all_detections:
        name = det['name']
        conf = det['conf']
        box = det['box']

        is_preferred = "mm" in name.lower() or "zone" in name.lower()

        if is_preferred:
            if best_zone_box is None or conf > overall_best_conf:
                overall_best_conf = conf
                best_zone_box = box
        elif best_zone_box is None and "ruler" in name.lower():
            best_zone_box = box

    return all_detections, best_zone_box


# ----------------------------------------------------------------------
# 2. Strict Calibration Logic
# ----------------------------------------------------------------------
def analyze_structure(frame, bbox):
    global calibration_error_msg
    x1, y1, x2, y2 = bbox
    h_img, w_img = frame.shape[:2]

    pad = 2
    cx1, cy1 = max(0, x1 + pad), max(0, y1 + pad)
    cx2, cy2 = min(w_img, x2 - pad), min(h_img, y2 - pad)

    roi = frame[cy1:cy2, cx1:cx2]
    if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
        calibration_error_msg = "ROI too small"
        return None

    is_horizontal_ruler = (cx2 - cx1) > (cy2 - cy1)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 4)

    if is_horizontal_ruler:
        kernel_len = max(5, roi.shape[0] // 8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    else:
        kernel_len = max(5, roi.shape[1] // 8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))

    clean_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    clean_lines = cv2.dilate(clean_lines, None, iterations=1)

    contours, _ = cv2.findContours(clean_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_ticks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5: continue
        tx, ty, tw, th = cv2.boundingRect(cnt)
        if is_horizontal_ruler:
            pos = tx + tw / 2.0
            length = th
            if th > tw * 2: all_ticks.append({'pos': pos, 'len': length, 'rect': (tx, ty, tw, th)})
        else:
            pos = ty + th / 2.0
            length = tw
            if tw > th * 2: all_ticks.append({'pos': pos, 'len': length, 'rect': (tx, ty, tw, th)})

    if len(all_ticks) < 5:
        calibration_error_msg = "Not enough lines detected"
        return None

    # --- CHECK 1: Length Uniformity ---
    # We find the max length, and assume those are CM lines.
    max_len = max(t['len'] for t in all_ticks)
    # Filter candidates
    cm_candidates = [t for t in all_ticks if t['len'] > (max_len * 0.85)]

    if len(cm_candidates) < 5:
        calibration_error_msg = "No clear CM lines found"
        return None

    # Strict check: Do these candidates actually look like each other?
    candidate_lengths = [t['len'] for t in cm_candidates]
    median_length = np.median(candidate_lengths)

    # Remove any candidate that deviates too much from the median length
    filtered_chain = []
    for t in cm_candidates:
        if abs(t['len'] - median_length) < (median_length * MAX_LENGTH_VARIANCE):
            filtered_chain.append(t)

    if len(filtered_chain) < 5:
        calibration_error_msg = "Tick lengths inconsistent (Shadows?)"
        return None

    filtered_chain.sort(key=lambda x: x['pos'])

    # --- CHECK 2: Spacing Uniformity (The most important accuracy check) ---
    # Calculate gaps
    gaps = []
    for i in range(len(filtered_chain) - 1):
        gaps.append(filtered_chain[i + 1]['pos'] - filtered_chain[i]['pos'])

    if not gaps: return None

    mean_gap = np.mean(gaps)
    std_dev_gap = np.std(gaps)

    # Coefficient of Variation (StdDev / Mean)
    # If this is high (> 5%), the ruler is likely tilted or detection is bad.
    spacing_variation = std_dev_gap / mean_gap

    if spacing_variation > MAX_GAP_VARIANCE:
        calibration_error_msg = f"Uneven Spacing (Tilt?): {spacing_variation * 100:.1f}% var"
        return None

    # If we get here, the lines are the same size AND same spacing.
    # We can trust this data.

    # Use Linear Regression for maximum sub-pixel accuracy
    y_coords = np.array([t['pos'] for t in filtered_chain])
    x_coords = np.arange(len(y_coords))
    slope, intercept = np.polyfit(x_coords, y_coords, 1)

    px_per_mm = slope / 10.0

    if px_per_mm < 2 or px_per_mm > 150:
        calibration_error_msg = "Scale out of bounds"
        return None

    calibration_error_msg = ""  # Clear error
    return {
        "px_per_mm": px_per_mm,
        "ticks": filtered_chain,
        "roi_offset": (cx1, cy1),
        "count": len(filtered_chain),
        "gap_mean": mean_gap,
        "gap_std": std_dev_gap
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

        d_mm = s_w / PIXELS_PER_MM
        l_mm = s_h / PIXELS_PER_MM

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
        for t in tick_data['ticks']:
            rx, ry, rw, rh = t['rect']
            ax, ay = int(off_x + rx), int(off_y + ry)
            aw, ah = int(rw), int(rh)
            cv2.rectangle(frame, (ax, ay), (ax + aw, ay + ah), (0, 0, 255), 1)

    # Draw Pellets
    for p in pellets:
        box = p['box']
        color = (0, 255, 0) if p['is_good'] else (0, 0, 255)
        cv2.drawContours(frame, [box], 0, color, 2)

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
    cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, 60), (20, 20, 20), -1)

    if is_calibrated:
        if calibration_locked:
            msg = f"LOCKED: {PIXELS_PER_MM:.2f} px/mm"
            col = (0, 255, 0)
        else:
            msg = f"Stabilizing... {PIXELS_PER_MM:.2f}"
            col = (0, 255, 255)
    else:
        # Show specific error if available
        msg = f"UNCALIBRATED: {calibration_error_msg}" if calibration_error_msg else "UNCALIBRATED"
        col = (0, 0, 255)

    cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

    # Draw Lock Progress
    if is_calibrated and not calibration_locked:
        progress = len(calibration_buffer) / CALIBRATION_BUFFER_SIZE
        cv2.rectangle(frame, (500, 20), (500 + int(200 * progress), 40), (0, 255, 255), -1)
        cv2.rectangle(frame, (500, 20), (700, 40), (255, 255, 255), 1)

    return frame


# ----------------------------------------------------------------------
# Main Logic
# ----------------------------------------------------------------------
def process_calibration(result):
    global PIXELS_PER_MM, is_calibrated, calibration_locked

    if result is None: return

    new_px = result['px_per_mm']

    if calibration_locked:
        if abs(new_px - PIXELS_PER_MM) > RESET_THRESHOLD:
            print("Movement detected! Re-calibrating...")
            calibration_locked = False
            calibration_buffer.clear()
        return

    calibration_buffer.append(new_px)
    avg_px = np.mean(calibration_buffer)
    std_dev = np.std(calibration_buffer)

    PIXELS_PER_MM = avg_px
    is_calibrated = True
    update_ranges()

    if len(calibration_buffer) == CALIBRATION_BUFFER_SIZE:
        if std_dev < STABILITY_THRESHOLD:
            calibration_locked = True


def main():
    global current_tick_data

    if not load_yolo_model(): return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    window_name = "Inspector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret: break

        yolo_objects, active_zone = run_yolo_detection(frame)

        if not active_zone:
            current_tick_data = None
        else:
            result = analyze_structure(frame, active_zone)
            current_tick_data = result  # Show detected ticks even if rejected by logic
            if result:
                process_calibration(result)

        pellets = detect_pellets(frame, yolo_objects)
        frame = draw_ui(frame, yolo_objects, active_zone, pellets, current_tick_data)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) == ord('q'): break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()