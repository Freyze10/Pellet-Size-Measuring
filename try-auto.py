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

# --- TIMING & STABILITY SETTINGS ---
REQUIRED_HOLD_TIME = 5.0  # Seconds to hold still before locking
MAX_JITTER_PER_FRAME = 0.5  # If px/mm changes > 0.5 instantly, reset timer
RESET_THRESHOLD = 3.0  # If locked and value changes > 3.0, unlock

# --- STRICT DETECTION RULES ---
MIN_CONSECUTIVE_LINES = 5  # Must see at least 5 lines in a row
CM_STRICT_HEIGHT_RATIO = 0.90  # Line must be 90% of max height
MAX_GAP_VARIANCE = 0.05  # Spacing must be within 5% tolerance

# System State
yolo_model = None
is_calibrated = False
calibration_locked = False

# Timing State
stabilization_start_time = None
last_valid_px = None
calibration_buffer = deque(maxlen=60)  # Used only for averaging the final value

current_tick_data = None
calibration_error_msg = ""

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
    best_zone_box = None
    overall_best_conf = 0

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
# 2. Strict Structure Analysis (5 Consecutive Lines)
# ----------------------------------------------------------------------
def analyze_structure(frame, bbox):
    global calibration_error_msg

    x1, y1, x2, y2 = bbox
    h_img, w_img = frame.shape[:2]

    pad = 5
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w_img, x2 + pad), min(h_img, y2 + pad)

    roi = frame[cy1:cy2, cx1:cx2]
    if roi.size == 0: return None

    is_horizontal_ruler = (cx2 - cx1) > (cy2 - cy1)

    # Pre-process
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
            aspect_ratio = th / float(tw)
            if aspect_ratio > 2.0:
                pos = tx + tw / 2.0
                all_ticks.append({'pos': pos, 'len': th, 'rect': (tx, ty, tw, th)})
        else:
            aspect_ratio = tw / float(th)
            if aspect_ratio > 2.0:
                pos = ty + th / 2.0
                all_ticks.append({'pos': pos, 'len': tw, 'rect': (tx, ty, tw, th)})

    if len(all_ticks) < MIN_CONSECUTIVE_LINES:
        calibration_error_msg = "Not enough lines"
        return None

    # --- LENGTH FILTERING ---
    sorted_by_len = sorted(all_ticks, key=lambda x: x['len'], reverse=True)
    top_n = min(len(sorted_by_len), 3)
    reference_cm_height = np.median([t['len'] for t in sorted_by_len[:top_n]])

    cm_ticks = []
    for t in all_ticks:
        if t['len'] > (reference_cm_height * CM_STRICT_HEIGHT_RATIO):
            t['type'] = 'CM'
            cm_ticks.append(t)
        elif t['len'] > (reference_cm_height * 0.60):
            t['type'] = 'HALF'
        else:
            t['type'] = 'MM'

    if len(cm_ticks) < MIN_CONSECUTIVE_LINES:
        calibration_error_msg = "Need 5+ CM lines"
        return {"px_per_mm": 0, "ticks": all_ticks, "roi_offset": (cx1, cy1)}

    # --- CONSECUTIVE CHAIN ANALYSIS ---
    cm_ticks.sort(key=lambda x: x['pos'])

    # Find the longest sequence of equally spaced lines
    longest_chain = []
    current_chain = [cm_ticks[0]]

    # Calculate median gap of ALL lines first to get a baseline estimate
    all_gaps = [cm_ticks[i + 1]['pos'] - cm_ticks[i]['pos'] for i in range(len(cm_ticks) - 1)]
    if not all_gaps: return None
    baseline_gap = np.median(all_gaps)

    for i in range(len(cm_ticks) - 1):
        gap = cm_ticks[i + 1]['pos'] - cm_ticks[i]['pos']

        # Check if this gap matches the baseline
        if abs(gap - baseline_gap) < (baseline_gap * MAX_GAP_VARIANCE):
            current_chain.append(cm_ticks[i + 1])
        else:
            if len(current_chain) > len(longest_chain):
                longest_chain = current_chain
            current_chain = [cm_ticks[i + 1]]

    if len(current_chain) > len(longest_chain):
        longest_chain = current_chain

    # --- FINAL VALIDATION ---
    if len(longest_chain) < MIN_CONSECUTIVE_LINES:
        calibration_error_msg = "Lines not consecutive/equal"
        return {"px_per_mm": 0, "ticks": all_ticks, "roi_offset": (cx1, cy1)}

    # Calculate precise average from the valid chain
    chain_gaps = []
    for i in range(len(longest_chain) - 1):
        chain_gaps.append(longest_chain[i + 1]['pos'] - longest_chain[i]['pos'])

    avg_gap_px = np.mean(chain_gaps)
    px_per_mm = avg_gap_px / 10.0

    if px_per_mm < 2 or px_per_mm > 150:
        calibration_error_msg = "Scale Error"
        return None

    calibration_error_msg = ""
    return {
        "px_per_mm": px_per_mm,
        "ticks": longest_chain,  # Only return the valid chain for red coloring
        "roi_offset": (cx1, cy1)
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
            # All ticks returned here are part of the valid chain (CM lines)
            rx, ry, rw, rh = t['rect']
            ax, ay = int(off_x + rx), int(off_y + ry)
            aw, ah = int(rw), int(rh)

            # Red Line (Used for Calibration)
            cv2.rectangle(frame, (ax, ay), (ax + aw, ay + ah), (0, 0, 255), 1)

    # Draw Pellets (Thinner Border = 1)
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
    cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, 60), (20, 20, 20), -1)

    if is_calibrated:
        if calibration_locked:
            msg = f"LOCKED: {PIXELS_PER_MM:.2f} px/mm"
            col = (0, 255, 0)
        else:
            # Calculate time progress
            if stabilization_start_time is not None:
                elapsed = time.time() - stabilization_start_time
                msg = f"Hold Still... {elapsed:.1f}s / {REQUIRED_HOLD_TIME:.1f}s"
            else:
                msg = "Stabilizing..."
            col = (0, 255, 255)
    else:
        msg = f"UNCALIBRATED: {calibration_error_msg}" if calibration_error_msg else "UNCALIBRATED"
        col = (0, 0, 255)

    cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

    # Time Progress Bar
    if is_calibrated and not calibration_locked and stabilization_start_time:
        elapsed = time.time() - stabilization_start_time
        progress = min(elapsed / REQUIRED_HOLD_TIME, 1.0)
        cv2.rectangle(frame, (500, 20), (500 + int(200 * progress), 40), (0, 255, 255), -1)
        cv2.rectangle(frame, (500, 20), (700, 40), (255, 255, 255), 1)

    return frame


# ----------------------------------------------------------------------
# Main Logic (Time-Based)
# ----------------------------------------------------------------------
def reset_stabilization():
    global calibration_locked, stabilization_start_time, last_valid_px
    if not calibration_locked:
        stabilization_start_time = None
        last_valid_px = None
        calibration_buffer.clear()


def process_calibration(result):
    global PIXELS_PER_MM, is_calibrated, calibration_locked
    global stabilization_start_time, last_valid_px

    new_px = result['px_per_mm']
    if new_px == 0: return

    # 1. Handle Locked State
    if calibration_locked:
        if abs(new_px - PIXELS_PER_MM) > RESET_THRESHOLD:
            print("Movement detected! Re-calibrating...")
            calibration_locked = False
            reset_stabilization()
        return

        # 2. Check Jitter
    if last_valid_px is not None:
        if abs(new_px - last_valid_px) > MAX_JITTER_PER_FRAME:
            # Jittered too much, reset timer
            reset_stabilization()
            last_valid_px = new_px
            return

    last_valid_px = new_px

    # 3. Handle Timer
    if stabilization_start_time is None:
        stabilization_start_time = time.time()

    elapsed = time.time() - stabilization_start_time

    # Add to buffer for averaging later
    calibration_buffer.append(new_px)

    # Update display (preview)
    PIXELS_PER_MM = np.mean(calibration_buffer)
    is_calibrated = True
    update_ranges()

    # 4. Check for Lock
    if elapsed >= REQUIRED_HOLD_TIME:
        # Final Lock Calculation
        final_val = np.mean(calibration_buffer)
        PIXELS_PER_MM = final_val
        calibration_locked = True
        print(f"LOCKED at {final_val:.4f} (held for {elapsed:.1f}s)")


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

    print("Running...")

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

        if cv2.waitKey(1) == ord('q'): break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()