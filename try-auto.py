import cv2
import numpy as np
import time
import math
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

# System State
yolo_model = None
is_calibrated = False
last_calibration_time = 0
CALIBRATION_INTERVAL = 0.5
current_tick_data = None

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

    all_detections = []
    calibration_zone = None
    best_conf = 0

    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = yolo_model.names[cls_id]

            all_detections.append({
                'box': (x1, y1, x2, y2),
                'name': name,
                'conf': conf
            })

            is_preferred = "mm" in name.lower() or "zone" in name.lower()

            if is_preferred:
                if conf > best_conf:
                    best_conf = conf
                    calibration_zone = (x1, y1, x2, y2)
            elif calibration_zone is None and "ruler" in name.lower():
                calibration_zone = (x1, y1, x2, y2)

    return all_detections, calibration_zone


# ----------------------------------------------------------------------
# 2. Strict CM-Sequence Calibration
# ----------------------------------------------------------------------
def analyze_structure(frame, bbox):
    x1, y1, x2, y2 = bbox
    h_img, w_img = frame.shape[:2]

    pad = 2
    cx1, cy1 = max(0, x1 + pad), max(0, y1 + pad)
    cx2, cy2 = min(w_img, x2 - pad), min(h_img, y2 - pad)

    roi = frame[cy1:cy2, cx1:cx2]
    if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20: return None

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
            pos = tx + tw / 2
            length = th
            if th > tw * 2:
                all_ticks.append({'pos': pos, 'len': length, 'rect': (tx, ty, tw, th)})
        else:
            pos = ty + th / 2
            length = tw
            if tw > th * 2:
                all_ticks.append({'pos': pos, 'len': length, 'rect': (tx, ty, tw, th)})

    if len(all_ticks) < 5: return None

    # Length Filter (Longest 85%)
    max_len = max(t['len'] for t in all_ticks)
    cm_candidates = [t for t in all_ticks if t['len'] > (max_len * 0.85)]

    if len(cm_candidates) < 5: return None

    # Sequence Logic
    cm_candidates.sort(key=lambda x: x['pos'])

    raw_gaps = []
    for i in range(len(cm_candidates) - 1):
        raw_gaps.append(cm_candidates[i + 1]['pos'] - cm_candidates[i]['pos'])

    if not raw_gaps: return None
    median_gap = np.median(raw_gaps)

    current_chain = [cm_candidates[0]]
    longest_chain = []

    for i in range(len(cm_candidates) - 1):
        gap = cm_candidates[i + 1]['pos'] - cm_candidates[i]['pos']

        if abs(gap - median_gap) < (median_gap * 0.1):
            current_chain.append(cm_candidates[i + 1])
        else:
            if len(current_chain) > len(longest_chain):
                longest_chain = current_chain
            current_chain = [cm_candidates[i + 1]]

    if len(current_chain) > len(longest_chain):
        longest_chain = current_chain

    if len(longest_chain) < 5: return None

    final_gaps = []
    for i in range(len(longest_chain) - 1):
        final_gaps.append(longest_chain[i + 1]['pos'] - longest_chain[i]['pos'])

    avg_gap_px = np.mean(final_gaps)
    px_per_mm = avg_gap_px / 10.0

    if px_per_mm < 2 or px_per_mm > 150: return None

    return {
        "px_per_mm": px_per_mm,
        "ticks": longest_chain,
        "roi_offset": (cx1, cy1),
        "count": len(longest_chain)
    }


# ----------------------------------------------------------------------
# 3. Pellet Detection (With Exclusion)
# ----------------------------------------------------------------------
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
    for cnt in contours:
        if not (MIN_CONTOUR_AREA <= cv2.contourArea(cnt) <= MAX_CONTOUR_AREA): continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        box = np.intp(cv2.boxPoints(rect))

        # --- EXCLUSION CHECK ---
        # If this pellet is inside ANY YOLO box (ruler, zone), skip it
        is_ignored = False
        for ex_obj in excluded_boxes:
            bx1, by1, bx2, by2 = ex_obj['box']
            # Simple point-in-box check
            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                is_ignored = True
                break

        if is_ignored:
            continue
        # -----------------------

        d1 = get_distance(box[0], box[1])
        d2 = get_distance(box[1], box[2])

        width_px = min(d1, d2)
        height_px = max(d1, d2)

        d_mm = width_px / PIXELS_PER_MM
        l_mm = height_px / PIXELS_PER_MM

        if should_process_pellet(d_mm, l_mm):
            pellets.append({
                'box': box,
                'diameter': d_mm,
                'length': l_mm,
                'is_good': is_within_tolerance(d_mm, l_mm)
            })
    return pellets


# ----------------------------------------------------------------------
# 4. Visualization
# ----------------------------------------------------------------------
def draw_ui(frame, yolo_objects, active_zone_box, pellets, tick_data):
    # A. Draw YOLO Objects
    for obj in yolo_objects:
        bx1, by1, bx2, by2 = obj['box']

        if active_zone_box and obj['box'] == active_zone_box:
            color = (0, 255, 0)  # Green for Active Zone
            label = f"{obj['name']} [ACTIVE]"
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        else:
            color = (100, 100, 100)  # Gray for passive objects
            label = obj['name']
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)

        cv2.putText(frame, label, (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # B. Draw Analysed Ticks
    if tick_data:
        off_x, off_y = tick_data['roi_offset']
        for t in tick_data['ticks']:
            rx, ry, rw, rh = t['rect']
            ax, ay = int(off_x + rx), int(off_y + ry)
            aw, ah = int(rw), int(rh)

            # THIN Red Line (Thickness = 1)
            cv2.rectangle(frame, (ax, ay), (ax + aw, ay + ah), (0, 0, 255), 1)

    # C. Draw Pellets
    for p in pellets:
        box = p['box']
        color = (0, 255, 0) if p['is_good'] else (0, 0, 255)

        cv2.drawContours(frame, [box], 0, color, 2)

        top_pt = min(box, key=lambda x: x[1])
        tx, ty = int(top_pt[0]), int(top_pt[1])

        txt = f"{p['diameter']:.2f}x{p['length']:.2f}"
        cv2.putText(frame, txt, (tx - 10, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, txt, (tx - 10, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if not p['is_good']:
            cv2.putText(frame, "!", (tx - 25, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # D. Status Overlay
    cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, 45), (20, 20, 20), -1)

    if is_calibrated:
        msg = f"CALIBRATED: {PIXELS_PER_MM:.2f} px/mm"
        col = (0, 255, 0)
    else:
        msg = "UNCALIBRATED (Need 5+ clear CM lines)"
        col = (0, 0, 255)

    cv2.putText(frame, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    return frame


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    global PIXELS_PER_MM, is_calibrated, last_calibration_time, current_tick_data

    if not load_yolo_model(): return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)

    window_name = "Inspector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Running...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        yolo_objects, active_zone = run_yolo_detection(frame)

        current_time = time.time()

        if not active_zone: current_tick_data = None

        if active_zone and (current_time - last_calibration_time > CALIBRATION_INTERVAL):
            result = analyze_structure(frame, active_zone)
            if result:
                new_px = result['px_per_mm']

                if is_calibrated:
                    PIXELS_PER_MM = (PIXELS_PER_MM * 0.9) + (new_px * 0.1)
                else:
                    PIXELS_PER_MM = new_px
                    is_calibrated = True

                last_calibration_time = current_time
                update_ranges()
                current_tick_data = result

        # Pass yolo_objects to exclude detection inside them
        pellets = detect_pellets(frame, yolo_objects)

        frame = draw_ui(frame, yolo_objects, active_zone, pellets, current_tick_data)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        # Check if window "X" button was clicked
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()