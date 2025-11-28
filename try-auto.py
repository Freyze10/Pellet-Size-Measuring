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

            # Priority: "mm" or "zone" > "ruler"
            is_preferred = "mm" in name.lower() or "zone" in name.lower()

            if is_preferred:
                if conf > best_conf:
                    best_conf = conf
                    calibration_zone = (x1, y1, x2, y2)
            elif calibration_zone is None and "ruler" in name.lower():
                calibration_zone = (x1, y1, x2, y2)

    return all_detections, calibration_zone


# ----------------------------------------------------------------------
# 2. CM-ONLY CALIBRATION LOGIC
# ----------------------------------------------------------------------
def analyze_structure(frame, bbox):
    x1, y1, x2, y2 = bbox
    h_img, w_img = frame.shape[:2]

    pad = 2
    cx1, cy1 = max(0, x1 + pad), max(0, y1 + pad)
    cx2, cy2 = min(w_img, x2 - pad), min(h_img, y2 - pad)

    roi = frame[cy1:cy2, cx1:cx2]
    if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20: return None

    # Detect Orientation (Width > Height = Horizontal Ruler)
    is_horizontal_ruler = (cx2 - cx1) > (cy2 - cy1)

    # Pre-process
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 4)

    # Line Filter
    if is_horizontal_ruler:
        kernel_len = max(5, roi.shape[0] // 8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))  # Keep Vert lines
    else:
        kernel_len = max(5, roi.shape[1] // 8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))  # Keep Horiz lines

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
            if th > tw * 2:  # Aspect ratio check
                all_ticks.append({'pos': pos, 'len': length, 'rect': (tx, ty, tw, th)})
        else:
            pos = ty + th / 2
            length = tw
            if tw > th * 2:
                all_ticks.append({'pos': pos, 'len': length, 'rect': (tx, ty, tw, th)})

    if len(all_ticks) < 3: return None

    # --- THE "LONG LINE" LOGIC ---

    # 1. Find the Maximum Length found in this ROI
    max_len = max(t['len'] for t in all_ticks)

    # 2. Identify CM Marks
    # Logic: CM marks are the longest.
    # We accept lines that are at least 85% of the longest line.
    # This filters out mm (shortest) and 0.5cm (medium) marks.
    cm_ticks = []
    ignored_ticks = []

    for t in all_ticks:
        if t['len'] > (max_len * 0.85):
            t['type'] = 'CM'
            cm_ticks.append(t)
        else:
            t['type'] = 'ignored'
            ignored_ticks.append(t)

    # Need at least 2 CM marks to measure distance
    if len(cm_ticks) < 2: return None

    # 3. Sort CM ticks by position
    cm_ticks.sort(key=lambda x: x['pos'])

    # 4. Calculate Gaps between CM ticks
    gaps = []
    for i in range(len(cm_ticks) - 1):
        dist = cm_ticks[i + 1]['pos'] - cm_ticks[i]['pos']

        # Robustness: Remove crazy outliers (detected 2 lines right next to each other)
        if dist > 10:
            gaps.append(dist)

    if not gaps: return None

    # 5. Calculate Pixels per MM
    # The gap between these lines is 1 CM (10mm)
    avg_cm_gap_px = np.median(gaps)

    px_per_mm = avg_cm_gap_px / 10.0

    # Sanity Check (e.g., 20px/mm is reasonable, 2000 is not)
    if px_per_mm < 2 or px_per_mm > 150: return None

    return {
        "px_per_mm": px_per_mm,
        "ticks": all_ticks,  # Send all back for drawing
        "roi_offset": (cx1, cy1)
    }


# ----------------------------------------------------------------------
# 3. Pellet Detection
# ----------------------------------------------------------------------
def detect_pellets(frame):
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
        box = np.intp(cv2.boxPoints(rect))

        d1 = get_distance(box[0], box[1])
        d2 = get_distance(box[1], box[2])

        # Independent of rotation
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

            # VISUAL FEEDBACK:
            if t['type'] == 'CM':
                # Thick Red Line for the ones used for Math
                cv2.rectangle(frame, (ax, ay), (ax + aw, ay + ah), (0, 0, 255), -1)
                # Add "CM" text nearby
                cv2.putText(frame, "CM", (ax, ay - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            else:
                # Thin Cyan Line for ignored marks
                cv2.rectangle(frame, (ax, ay), (ax + aw, ay + ah), (255, 255, 0), 1)

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
        msg = "UNCALIBRATED"
        col = (0, 0, 255)

    cv2.putText(frame, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    if is_calibrated:
        cv2.putText(frame, "[RED] = CM Lines (Used)", (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "[CYAN] = Ignored", (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

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

                # Weighted average for stability
                if is_calibrated:
                    PIXELS_PER_MM = (PIXELS_PER_MM * 0.9) + (new_px * 0.1)
                else:
                    PIXELS_PER_MM = new_px
                    is_calibrated = True

                last_calibration_time = current_time
                update_ranges()
                current_tick_data = result

        pellets = detect_pellets(frame)
        frame = draw_ui(frame, yolo_objects, active_zone, pellets, current_tick_data)

        cv2.imshow("Inspector", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()