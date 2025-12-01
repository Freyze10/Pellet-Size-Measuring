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

# --- STABILITY & LOCKING SETTINGS ---
CALIBRATION_BUFFER_SIZE = 150  # 5 Seconds @ 30fps
STABILITY_THRESHOLD = 0.5  # Std Dev limit
RESET_THRESHOLD = 3.0  # Scale change limit (Zoom/Distance)
MAX_MOVEMENT_PIXELS = 50  # Position change limit (Pan/Tilt)

# --- DETECTION STRICTNESS ---
ASPECT_RATIO_MIN = 2.0
CM_STRICT_HEIGHT_RATIO = 0.85  # CM lines must be at least 85% of tallest line
MIN_CM_LINES_REQUIRED = 5  # MUST have at least 5 CM lines
MAX_GAP_VARIANCE = 0.03  # Stricter: 3% variance
GAP_OUTLIER_TOLERANCE = 0.15  # Individual gaps can vary by max 15%

# System State
yolo_model = None
is_calibrated = False
calibration_locked = False
calibration_buffer = deque(maxlen=CALIBRATION_BUFFER_SIZE)
current_tick_data = None
calibration_error_msg = ""

# Position Tracking
locked_zone_coords = None

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
# 2. IMPROVED STRUCTURE ANALYSIS
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

    # Determine Orientation
    roi_h, roi_w = roi.shape[:2]
    is_horizontal_ruler = roi_w > roi_h

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 19, 5)

    if is_horizontal_ruler:
        k_height = max(5, roi_h // 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_height))
    else:
        k_width = max(5, roi_w // 10)
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
                # Math Center X
                pos = tx + tw / 2.0
                all_ticks.append({'pos': pos, 'len': th, 'rect': (tx, ty, tw, th)})
        else:
            aspect_ratio = tw / float(th) if th > 0 else 0
            if aspect_ratio > ASPECT_RATIO_MIN:
                # Math Center Y
                pos = ty + th / 2.0
                all_ticks.append({'pos': pos, 'len': tw, 'rect': (tx, ty, tw, th)})

    if len(all_ticks) < MIN_CM_LINES_REQUIRED:
        calibration_error_msg = f"Need {MIN_CM_LINES_REQUIRED}+ lines (found {len(all_ticks)})"
        return None

    # Length Filtering
    max_length = max(t['len'] for t in all_ticks)
    cm_threshold = max_length * CM_STRICT_HEIGHT_RATIO

    cm_ticks = []
    for t in all_ticks:
        if t['len'] >= cm_threshold:
            t['type'] = 'CM'
            cm_ticks.append(t)
        elif t['len'] > (max_length * 0.60):
            t['type'] = 'HALF'
        else:
            t['type'] = 'MM'

    if len(cm_ticks) < MIN_CM_LINES_REQUIRED:
        calibration_error_msg = f"Need {MIN_CM_LINES_REQUIRED} CM lines (found {len(cm_ticks)})"
        return {"px_per_mm": 0, "ticks": all_ticks, "roi_offset": (cx1, cy1), "is_horiz": is_horizontal_ruler,
                "roi_dims": (roi_w, roi_h)}

    # Spacing Analysis
    cm_ticks.sort(key=lambda x: x['pos'])
    gaps = []
    for i in range(len(cm_ticks) - 1):
        gaps.append(cm_ticks[i + 1]['pos'] - cm_ticks[i]['pos'])

    if len(gaps) < (MIN_CM_LINES_REQUIRED - 1):
        calibration_error_msg = "Not enough gaps"
        return {"px_per_mm": 0, "ticks": all_ticks, "roi_offset": (cx1, cy1), "is_horiz": is_horizontal_ruler,
                "roi_dims": (roi_w, roi_h)}

    median_gap = np.median(gaps)
    valid_gaps = [g for g in gaps if abs(g - median_gap) / median_gap < GAP_OUTLIER_TOLERANCE]

    if len(valid_gaps) < (MIN_CM_LINES_REQUIRED - 1):
        calibration_error_msg = "Gaps uneven"
        return {"px_per_mm": 0, "ticks": all_ticks, "roi_offset": (cx1, cy1), "is_horiz": is_horizontal_ruler,
                "roi_dims": (roi_w, roi_h)}

    gap_mean = np.mean(valid_gaps)
    gap_std = np.std(valid_gaps)
    spacing_variance = gap_std / gap_mean if gap_mean > 0 else 1.0

    if spacing_variance > MAX_GAP_VARIANCE:
        calibration_error_msg = f"Variance High: {spacing_variance * 100:.1f}%"
        return {"px_per_mm": 0, "ticks": all_ticks, "roi_offset": (cx1, cy1), "is_horiz": is_horizontal_ruler,
                "roi_dims": (roi_w, roi_h)}

    px_per_mm = gap_mean / 10.0

    if px_per_mm < 2 or px_per_mm > 150:
        calibration_error_msg = "Scale invalid"
        return None

    calibration_error_msg = ""
    return {
        "px_per_mm": px_per_mm,
        "ticks": all_ticks,
        "roi_offset": (cx1, cy1),
        "is_horiz": is_horizontal_ruler,
        "roi_dims": (roi_w, roi_h)
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
# 4. Visualization (Thin Uniform Lines)
# ----------------------------------------------------------------------
def draw_ui(frame, yolo_objects, active_zone_box, pellets, tick_data):
    # Draw YOLO Objects
    for obj in yolo_objects:
        bx1, by1, bx2, by2 = obj['box']
        color = (0, 255, 0) if (active_zone_box and obj['box'] == active_zone_box) else (100, 100, 100)
        label = f"{obj['name']} [ACTIVE]" if color == (0, 255, 0) else obj['name']
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(frame, label, (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # --- UPDATED TICK DRAWING ---
    if tick_data:
        off_x, off_y = tick_data['roi_offset']
        is_horiz = tick_data['is_horiz']
        roi_w, roi_h = tick_data['roi_dims']

        # We define a fixed line length based on the zone size to make them uniform
        fixed_len = roi_h if is_horiz else roi_w

        for t in tick_data['ticks']:
            pos_val = t['pos']

            # Determine color and visual priority
            if t['type'] == 'CM':
                color = (0, 0, 255)  # Red for CM
                thickness = 2
            elif t['type'] == 'HALF':
                color = (0, 255, 255)  # Yellow for half
                thickness = 1
            else:
                color = (255, 255, 0)  # Cyan for mm
                thickness = 1

            if is_horiz:
                # Vertical Line (x is pos, y spans height)
                x = int(off_x + pos_val)
                start_pt = (x, int(off_y))
                end_pt = (x, int(off_y + fixed_len))
            else:
                # Horizontal Line (y is pos, x spans width)
                y = int(off_y + pos_val)
                start_pt = (int(off_x), y)
                end_pt = (int(off_x + fixed_len), y)

            # Draw the clean line
            cv2.line(frame, start_pt, end_pt, color, thickness)

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
    cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, 60), (20, 20, 20), -1)

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

    cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

    # Simple Legend
    cv2.putText(frame, "RED = CM Lines (Used)", (450, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if is_calibrated and not calibration_locked:
        progress = len(calibration_buffer) / CALIBRATION_BUFFER_SIZE
        cv2.rectangle(frame, (500, 20), (500 + int(200 * progress), 40), (0, 255, 255), -1)
        cv2.rectangle(frame, (500, 20), (700, 40), (255, 255, 255), 1)

    return frame


# ----------------------------------------------------------------------
# Main Logic
# ----------------------------------------------------------------------
def reset_stabilization():
    global calibration_locked
    if not calibration_locked:
        calibration_buffer.clear()


def process_calibration(result):
    global PIXELS_PER_MM, is_calibrated, calibration_locked, locked_zone_coords

    new_px = result['px_per_mm']
    new_coords = result['roi_offset']

    if new_px == 0: return

    if calibration_locked:
        if abs(new_px - PIXELS_PER_MM) > RESET_THRESHOLD:
            print("⚠ Scale changed! Re-calibrating...")
            calibration_locked = False
            calibration_buffer.clear()
            return
        if locked_zone_coords:
            lx, ly = locked_zone_coords
            nx, ny = new_coords
            dist = math.sqrt((lx - nx) ** 2 + (ly - ny) ** 2)
            if dist > MAX_MOVEMENT_PIXELS:
                print("⚠ Camera moved! Re-calibrating...")
                calibration_locked = False
                calibration_buffer.clear()
        return

    if len(calibration_buffer) > 0:
        current_avg = np.mean(calibration_buffer)
        if abs(new_px - current_avg) > 1.0:
            calibration_buffer.clear()

    calibration_buffer.append(new_px)
    avg_px = np.mean(calibration_buffer)
    std_dev = np.std(calibration_buffer)

    PIXELS_PER_MM = avg_px
    is_calibrated = True
    update_ranges()

    if len(calibration_buffer) == CALIBRATION_BUFFER_SIZE:
        if std_dev < STABILITY_THRESHOLD:
            calibration_locked = True
            locked_zone_coords = new_coords
            print(f"🔒 LOCKED at {avg_px:.2f} px/mm | StdDev: {std_dev:.3f}")


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