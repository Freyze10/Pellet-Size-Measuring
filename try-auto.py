import cv2
import numpy as np
import time
import math
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
YOLO_MODEL_PATH = "yolo/best.pt"  # Make sure this points to your specific model

# Global Calibration Settings
PIXELS_PER_MM = 10.0  # Safe default
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0

# State Variables
yolo_model = None
is_calibrated = False
last_calibration_time = 0
CALIBRATION_INTERVAL = 0.5  # Recalibrate every 0.5 seconds

# Camera Settings
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720

# Pellet Filter Settings
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 20000


# ----------------------------------------------------------------------
# Range Management
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
        # Print class names to verify
        print(f"  Classes: {yolo_model.names}")
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
# 1. Intelligent YOLO Detection
# ----------------------------------------------------------------------
def run_yolo_detection(frame):
    """
    Returns:
    1. all_detections: List of dicts for UI drawing
    2. calibration_zone: The specific box (x1,y1,x2,y2) to use for math
    """
    if yolo_model is None: return [], None

    results = yolo_model(frame, conf=0.4, verbose=False)

    all_detections = []
    calibration_zone = None
    best_conf = 0

    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = yolo_model.names[cls_id]

            # Store for display
            all_detections.append({
                'box': (x1, y1, x2, y2),
                'name': name,
                'conf': conf
            })

            # LOGIC: Find the best zone for calibration
            # We prioritize classes named "mm", "zone", or "scale"
            is_preferred = "mm" in name.lower() or "zone" in name.lower()

            if is_preferred:
                # If it's an MM zone, take it immediately if confidence is good
                if conf > best_conf:
                    best_conf = conf
                    calibration_zone = (x1, y1, x2, y2)
            elif calibration_zone is None and "ruler" in name.lower():
                # Fallback: If no MM zone found yet, accept a "ruler" temporarily
                calibration_zone = (x1, y1, x2, y2)

    return all_detections, calibration_zone


# ----------------------------------------------------------------------
# 2. Mathematical Calibration (Inside the Zone)
# ----------------------------------------------------------------------
def analyze_calibration_zone(frame, bbox):
    """
    Runs image processing ONLY inside the detected bbox to find tick marks.
    """
    x1, y1, x2, y2 = bbox

    # Add slight padding check
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Crop
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10: return None

    # Pre-process
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold to handle lighting changes on metal/plastic
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 6)

    # Find contours (ticks)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_h, roi_w = roi.shape[:2]
    # Determine orientation based on aspect ratio of the ZONE
    is_vertical_ruler = roi_h > roi_w

    ticks = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter noise
        if area < 5 or area > 500: continue

        tx, ty, tw, th = cv2.boundingRect(cnt)
        aspect = float(tw) / th

        if is_vertical_ruler:
            # Ticks should be horizontal lines (width > height)
            if aspect > 1.2:
                ticks.append(ty + th / 2)  # Use Y position
        else:
            # Ticks should be vertical lines (height > width)
            if aspect < 0.8:
                ticks.append(tx + tw / 2)  # Use X position

    if len(ticks) < 5: return None

    # Calculate Median Interval
    ticks.sort()
    intervals = []
    for i in range(len(ticks) - 1):
        gap = ticks[i + 1] - ticks[i]
        # Filter obvious outliers (like gaps between numbers)
        if 2 < gap < (max(roi_w, roi_h) / 5):
            intervals.append(gap)

    if len(intervals) < 3: return None

    median_gap = np.median(intervals)

    # Second pass: clean statistics
    clean_intervals = [i for i in intervals if abs(i - median_gap) < (median_gap * 0.2)]

    if len(clean_intervals) < 3: return None

    px_per_mm = np.mean(clean_intervals)

    # Sanity check (prevent 0.1 or 1000 px/mm)
    if px_per_mm < 2 or px_per_mm > 100: return None

    return px_per_mm


# ----------------------------------------------------------------------
# 3. Precise Pellet Detection
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

        # Calculate size independent of rotation
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
# 4. Visualization (UI)
# ----------------------------------------------------------------------
def draw_ui(frame, yolo_objects, active_zone_box, pellets):
    overlay = frame.copy()

    # --- A. Draw YOLO Objects (Background) ---
    for obj in yolo_objects:
        bx1, by1, bx2, by2 = obj['box']
        name = obj['name']

        # Determine color: Green if it's the active calibration zone, Blue otherwise
        if active_zone_box and obj['box'] == active_zone_box:
            color = (0, 255, 0)  # Green for Active Zone
            label = f"{name} [ACTIVE]"
            thick = 2
        else:
            color = (255, 100, 0)  # Blue/Orange for other detections
            label = name
            thick = 1

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, thick)
        cv2.putText(frame, label, (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # --- B. Draw Pellets ---
    for p in pellets:
        box = p['box']
        color = (0, 255, 0) if p['is_good'] else (0, 0, 255)

        cv2.drawContours(frame, [box], 0, color, 2)

        # Text positioning
        top_pt = min(box, key=lambda x: x[1])
        tx, ty = int(top_pt[0]), int(top_pt[1])

        # Clean text with outline
        txt = f"{p['diameter']:.2f} x {p['length']:.2f}"
        cv2.putText(frame, txt, (tx - 10, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Outline
        cv2.putText(frame, txt, (tx - 10, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if not p['is_good']:
            cv2.putText(frame, "!", (tx - 25, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- C. Status Bar ---
    # Top background bar
    cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, 50), (30, 30, 30), -1)

    # Calibration Status
    if is_calibrated:
        cal_color = (0, 255, 0)
        cal_txt = f"CALIBRATED: {PIXELS_PER_MM:.2f} px/mm"
    else:
        cal_color = (0, 0, 255)
        cal_txt = "NO CALIBRATION - SHOW MM ZONE"

    cv2.putText(frame, cal_txt, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cal_color, 2)

    # Counts
    total = len(pellets)
    good = sum(1 for p in pellets if p['is_good'])
    stats = f"Count: {total} | OK: {good} | NG: {total - good}"
    cv2.putText(frame, stats, (DESIRED_WIDTH - 400, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return frame


# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
def main():
    global PIXELS_PER_MM, is_calibrated, last_calibration_time

    print("--- Pellet Inspector with YOLO Zone Detection ---")

    if not load_yolo_model(): return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Run YOLO to find Ruler/Zone
        yolo_objects, active_zone = run_yolo_detection(frame)

        # 2. Attempt Calibration if Zone is valid and time has passed
        current_time = time.time()
        if active_zone and (current_time - last_calibration_time > CALIBRATION_INTERVAL):
            px_val = analyze_calibration_zone(frame, active_zone)
            if px_val:
                # Weighted average for stability
                if is_calibrated:
                    PIXELS_PER_MM = (PIXELS_PER_MM * 0.8) + (px_val * 0.2)
                else:
                    PIXELS_PER_MM = px_val
                    is_calibrated = True

                last_calibration_time = current_time
                update_ranges()

        # 3. Detect Pellets using updated calibration
        pellets = detect_pellets(frame)

        # 4. Draw Everything
        frame = draw_ui(frame, yolo_objects, active_zone, pellets)

        cv2.imshow("Inspector", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()