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
STABILITY_THRESHOLD = 0.5
RESET_THRESHOLD = 3.0
MAX_MOVEMENT_PIXELS = 50

# --- DETECTION STRICTNESS ---
ASPECT_RATIO_MIN = 2.0
HEIGHT_RATIO_STRICT = 0.90  # Major lines must be 90% of tallest
MAX_GAP_VARIANCE = 0.03
GAP_OUTLIER_TOLERANCE = 0.15

# System State
yolo_model = None
is_calibrated = False
calibration_locked = False
calibration_buffer = deque(maxlen=CALIBRATION_BUFFER_SIZE)
current_tick_data = None
calibration_error_msg = ""
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
# 2. DYNAMIC STRUCTURE ANALYSIS (CM / INCH)
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

    # Filter for Major Lines (CM or Inch)
    max_length = max(t['len'] for t in all_ticks)
    major_threshold = max_length * HEIGHT_RATIO_STRICT

    major_ticks = []
    for t in all_ticks:
        if t['len'] >= major_threshold:
            t['type'] = 'MAJOR'  # Represents CM or INCH depending on mode
            major_ticks.append(t)
        elif t['len'] > (max_length * 0.60):
            t['type'] = 'MEDIUM'
        else:
            t['type'] = 'MINOR'

    if len(major_ticks) < MIN_LINES:
        calibration_error_msg = f"Need {MIN_LINES} {CALIBRATION_MODE} lines (found {len(major_ticks)})"
        return {"px_per_mm": 0, "ticks": all_ticks, "roi_offset": (cx1, cy1)}

    # Spacing Analysis
    major_ticks.sort(key=lambda x: x['pos'])
    gaps = []
    for i in range(len(major_ticks) - 1):
        gaps.append(major_ticks[i + 1]['pos'] - major_ticks[i]['pos'])

    median_gap = np.median(gaps)
    valid_gaps = [g for g in gaps if abs(g - median_gap) / median_gap < GAP_OUTLIER_TOLERANCE]

    if len(valid_gaps) < (MIN_LINES - 1):
        calibration_error_msg = f"Gaps too uneven (only {len(valid_gaps)} valid)"
        return {"px_per_mm": 0, "ticks": all_ticks, "roi_offset": (cx1, cy1)}

    gap_mean = np.mean(valid_gaps)
    gap_std = np.std(valid_gaps)
    spacing_variance = gap_std / gap_mean if gap_mean > 0 else 1.0

    if spacing_variance > MAX_GAP_VARIANCE:
        calibration_error_msg = f"Spacing uneven ({spacing_variance * 100:.1f}%)"
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
        "spacing_variance": spacing_variance
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

            if t['type'] == 'MAJOR':  # CM or INCH
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
            is_calibrated = False
            print(f"Switched to {CALIBRATION_MODE} Mode")

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBL34E) < 1: break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()