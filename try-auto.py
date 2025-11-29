import cv2
import numpy as np
import time
import math
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# EXACT Class name for the mm zone in your YOLO model
CALIBRATION_ZONE_CLASS = "mm_zone"

PIXELS_PER_MM = 10.0  # Fallback
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0

# System State
yolo_model = None
model_names = {}
is_calibrated = False
last_calibration_time = 0
CALIBRATION_INTERVAL = 0.5
current_tick_data = None

# Settings
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
    global yolo_model, model_names
    try:
        yolo_model = YOLO("yolo/best.pt")
        model_names = yolo_model.names
        print(f"✓ Model loaded. Calibration Class: '{CALIBRATION_ZONE_CLASS}'")
        return True
    except Exception as e:
        print(f"✗ Failed to load YOLO: {e}")
        return False


def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


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
            name = model_names[cls_id]

            all_detections.append({'box': (x1, y1, x2, y2), 'name': name, 'conf': conf})

            # STRICTLY select the class defined in config
            if name == CALIBRATION_ZONE_CLASS:
                if conf > best_conf:
                    best_conf = conf
                    calibration_zone = (x1, y1, x2, y2)

    return all_detections, calibration_zone


# ----------------------------------------------------------------------
# 2. STRICT CALIBRATION LOGIC
# ----------------------------------------------------------------------
def analyze_structure(frame, bbox):
    x1, y1, x2, y2 = bbox
    h_img, w_img = frame.shape[:2]

    # 1. Crop to Zone
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20: return None

    # 2. Determine Orientation
    # Width > Height means Ruler is Horizontal, so Ticks are Vertical Lines
    is_horizontal_ruler = (x2 - x1) > (y2 - y1)

    # 3. Pre-process (Isolate Lines)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 6)

    # Morphological Filter to keep only lines perpendicular to ruler
    if is_horizontal_ruler:
        # Keep vertical lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, roi.shape[0] // 10))
    else:
        # Keep horizontal lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (roi.shape[1] // 10, 1))

    clean_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(clean_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    # 4. Extract Raw Candidates
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10: continue

        tx, ty, tw, th = cv2.boundingRect(cnt)

        if is_horizontal_ruler:
            # Check aspect ratio (must be tall)
            if th > tw * 2:
                candidates.append({
                    'pos': tx + tw / 2,  # Position along ruler
                    'align_axis': ty + th / 2,  # Vertical alignment center
                    'length': th,
                    'rect': (tx, ty, tw, th),
                    'status': 'unknown'
                })
        else:
            # Check aspect ratio (must be wide)
            if tw > th * 2:
                candidates.append({
                    'pos': ty + th / 2,
                    'align_axis': tx + tw / 2,
                    'length': tw,
                    'rect': (tx, ty, tw, th),
                    'status': 'unknown'
                })

    if not candidates: return None

    # --- FILTER 1: LENGTH (Find Longest Lines) ---
    max_len = max(c['length'] for c in candidates)
    long_ticks = []

    for c in candidates:
        # Must be at least 85% of the longest line detected
        if c['length'] > (max_len * 0.85):
            long_ticks.append(c)
        else:
            c['status'] = 'too_short'

    if len(long_ticks) < 2: return None

    # --- FILTER 2: ALIGNMENT (Straight Line Check) ---
    # Calculate median alignment (e.g., all ticks should be at same height)
    median_align = np.median([c['align_axis'] for c in long_ticks])
    aligned_ticks = []
    alignment_tolerance = max_len * 0.3  # Allow some wiggle room

    for c in long_ticks:
        if abs(c['align_axis'] - median_align) < alignment_tolerance:
            aligned_ticks.append(c)
        else:
            c['status'] = 'misaligned'

    if len(aligned_ticks) < 2: return None

    # --- FILTER 3: EQUAL SPACING (Gap Check) ---
    aligned_ticks.sort(key=lambda x: x['pos'])

    positions = [c['pos'] for c in aligned_ticks]
    gaps = np.diff(positions)

    if len(gaps) == 0: return None

    median_gap = np.median(gaps)

    # Strict Spacing Filter: Gap must be within 10% of median
    # This removes gaps where a line might have been missed
    valid_gaps = []
    for g in gaps:
        if abs(g - median_gap) < (median_gap * 0.1):
            valid_gaps.append(g)

    if len(valid_gaps) == 0: return None  # No consistent spacing found

    avg_gap_px = np.mean(valid_gaps)

    # 5. Final Calculation
    # Assumption: The "Longest" lines on a ruler are CM marks (10mm gap)
    px_per_mm = avg_gap_px / 10.0

    # Sanity Check
    if px_per_mm < 2 or px_per_mm > 200: return None

    # Mark good ticks for drawing
    for c in aligned_ticks:
        c['status'] = 'valid'

    return {
        "px_per_mm": px_per_mm,
        "ticks": candidates,  # Return all for debug drawing
        "roi_offset": (x1, y1)
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
        w_px = min(d1, d2)
        h_px = max(d1, d2)

        d_mm = w_px / PIXELS_PER_MM
        l_mm = h_px / PIXELS_PER_MM

        if should_process_pellet(d_mm, l_mm):
            pellets.append({
                'box': box, 'diameter': d_mm, 'length': l_mm,
                'is_good': (DIAMETER_MIN <= d_mm <= DIAMETER_MAX and LENGTH_MIN <= l_mm <= LENGTH_MAX)
            })
    return pellets


# ----------------------------------------------------------------------
# 4. Visualization
# ----------------------------------------------------------------------
def draw_ui(frame, yolo_objects, active_zone_box, pellets, tick_data):
    # Draw YOLO
    for obj in yolo_objects:
        bx1, by1, bx2, by2 = obj['box']
        color = (0, 255, 0) if obj['box'] == active_zone_box else (100, 100, 100)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2 if color == (0, 255, 0) else 1)
        cv2.putText(frame, obj['name'], (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw Ticks (Debug)
    if tick_data:
        off_x, off_y = tick_data['roi_offset']
        for t in tick_data['ticks']:
            rx, ry, rw, rh = t['rect']
            ax, ay = int(off_x + rx), int(off_y + ry)

            if t['status'] == 'valid':
                # GREEN: Valid Long, Straight, Spaced line
                cv2.rectangle(frame, (ax, ay), (ax + int(rw), ay + int(rh)), (0, 255, 0), -1)
            elif t['status'] == 'too_short':
                # RED: Too short
                cv2.rectangle(frame, (ax, ay), (ax + int(rw), ay + int(rh)), (0, 0, 255), 1)
            elif t['status'] == 'misaligned':
                # YELLOW: Long enough, but not straight
                cv2.rectangle(frame, (ax, ay), (ax + int(rw), ay + int(rh)), (0, 255, 255), 1)
            else:
                # GRAY: Failed other checks
                cv2.rectangle(frame, (ax, ay), (ax + int(rw), ay + int(rh)), (100, 100, 100), 1)

    # Draw Pellets
    for p in pellets:
        box = p['box']
        color = (0, 255, 0) if p['is_good'] else (0, 0, 255)
        cv2.drawContours(frame, [box], 0, color, 2)
        top = min(box, key=lambda x: x[1])
        cv2.putText(frame, f"{p['diameter']:.2f}x{p['length']:.2f}", (int(top[0]), int(top[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if not p['is_good']:
            cv2.putText(frame, "!", (int(top[0] - 15), int(top[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Status
    cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, 45), (20, 20, 20), -1)
    msg = f"CALIBRATED: {PIXELS_PER_MM:.2f} px/mm" if is_calibrated else f"WAITING FOR '{CALIBRATION_ZONE_CLASS}'..."
    col = (0, 255, 0) if is_calibrated else (0, 0, 255)
    cv2.putText(frame, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    if is_calibrated:
        cv2.putText(frame, "GREEN: Used | RED: Short | YELLOW: Crooked", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)

    return frame


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    global PIXELS_PER_MM, is_calibrated, last_calibration_time, current_tick_data
    if not load_yolo_model(): return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(3, DESIRED_WIDTH);
    cap.set(4, DESIRED_HEIGHT)

    print("Running...")
    while True:
        ret, frame = cap.read()
        if not ret: break

        yolo_objects, active_zone = run_yolo_detection(frame)

        if active_zone and (time.time() - last_calibration_time > CALIBRATION_INTERVAL):
            result = analyze_structure(frame, active_zone)
            if result:
                new_px = result['px_per_mm']
                if is_calibrated:
                    PIXELS_PER_MM = (PIXELS_PER_MM * 0.9) + (new_px * 0.1)
                else:
                    PIXELS_PER_MM = new_px; is_calibrated = True

                last_calibration_time = time.time()
                update_ranges()
                current_tick_data = result
            else:
                current_tick_data = None  # Clear debug if lost

        pellets = detect_pellets(frame)
        frame = draw_ui(frame, yolo_objects, active_zone, pellets, current_tick_data)
        cv2.imshow("Inspector", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release();
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()