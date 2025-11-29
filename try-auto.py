import cv2
import numpy as np
import time
import sys
import math
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# IMPORTANT: Change this to the exact class name of your mm strip in YOLO
CALIBRATION_CLASS_NAME = "mm_zone"

# Initial Fallback
PIXELS_PER_MM = 6.0

# Pellet Tolerances
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0

# Detection Settings
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 25000

# Global State
yolo_model = None
model_names = {}
last_calibration_time = 0
CALIBRATION_INTERVAL = 0.2  # Fast updates for visual feedback
is_calibrated = False


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
def load_yolo_model():
    global yolo_model, model_names
    try:
        yolo_model = YOLO("yolo/best.pt")
        model_names = yolo_model.names
        print(f"✓ Model Loaded. Classes: {model_names}")
        return True
    except Exception as e:
        print(f"✗ Failed to load YOLO: {e}")
        return False


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
# 1. Custom NMS (Filter Overlapping Boxes)
# ----------------------------------------------------------------------
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)


def filter_detections(detections, iou_threshold=0.3):
    if not detections: return []
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    final_detections = []
    while len(detections) > 0:
        current = detections.pop(0)
        final_detections.append(current)
        remaining = []
        for other in detections:
            overlap = calculate_iou(current['bbox'], other['bbox'])
            if current['class_name'] == other['class_name'] and overlap > iou_threshold:
                continue
            remaining.append(other)
        detections = remaining
    return final_detections


# ----------------------------------------------------------------------
# 2. YOLO Detection
# ----------------------------------------------------------------------
def detect_objects(frame):
    if yolo_model is None: return []
    raw_detections = []
    results = yolo_model(frame, conf=0.4, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            name = model_names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            raw_detections.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'class_name': name,
                'confidence': conf
            })
    return filter_detections(raw_detections)


# ----------------------------------------------------------------------
# 3. OpenCV Calibration (Returns Visual Debug Data)
# ----------------------------------------------------------------------
def analyze_mm_zone(frame, bbox):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    # 1. Validation & Crop
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h: return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return None

    # 2. Image Processing
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 5)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. Filter Contours (Find Ticks)
    tick_candidates = []  # Stores {'pos': float, 'point': (x,y)}

    # Check orientation: wider than tall = horizontal ruler
    is_horizontal = (x2 - x1) > (y2 - y1)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5: continue

        tx, ty, tw, th = cv2.boundingRect(cnt)
        ratio = float(tw) / th

        # Calculate global center point for drawing
        global_center = (x1 + tx + tw // 2, y1 + ty + th // 2)

        if is_horizontal:
            # Ticks should be tall and thin
            if ratio < 0.8:
                tick_candidates.append({'val': tx + tw / 2, 'point': global_center})
        else:
            # Ticks should be wide and short
            if ratio > 1.2:
                tick_candidates.append({'val': ty + th / 2, 'point': global_center})

    if len(tick_candidates) < 5: return None

    # 4. Sort by position
    tick_candidates.sort(key=lambda x: x['val'])

    # Extract just the scalar values for math
    scalar_vals = [t['val'] for t in tick_candidates]

    # 5. Calculate Intervals
    intervals = []
    for i in range(len(scalar_vals) - 1):
        gap = scalar_vals[i + 1] - scalar_vals[i]
        # Sanity check: 1mm is usually between 2px and 100px depending on zoom
        if 2 < gap < 100:
            intervals.append(gap)

    if len(intervals) < 3: return None

    # 6. Statistics (Median Filter)
    median_gap = np.median(intervals)

    # Remove outliers (gaps that are double/triple the median or noise)
    valid_gaps = [g for g in intervals if abs(g - median_gap) < 2.0]

    if len(valid_gaps) < 3: return None

    px_per_mm = np.mean(valid_gaps)

    if px_per_mm < 2 or px_per_mm > 100: return None

    # Return both the math result AND the visual points
    return {
        'px_per_mm': px_per_mm,
        'tick_points': [t['point'] for t in tick_candidates]
    }


# ----------------------------------------------------------------------
# 4. Pellet Measurement
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

        d1 = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
        d2 = math.sqrt((box[1][0] - box[2][0]) ** 2 + (box[1][1] - box[2][1]) ** 2)

        w_px = min(d1, d2)
        h_px = max(d1, d2)

        dia = w_px / PIXELS_PER_MM
        length = h_px / PIXELS_PER_MM

        if should_process_pellet(dia, length):
            pellets.append({
                'box': box,
                'diameter': dia,
                'length': length,
                'ok': (DIAMETER_MIN <= dia <= DIAMETER_MAX and LENGTH_MIN <= length <= LENGTH_MAX)
            })
    return pellets


def should_process_pellet(d, l):
    if d < 0.5 or l < 0.5: return False
    return (DIAMETER_EXCLUDE_MIN <= d <= DIAMETER_EXCLUDE_MAX and
            LENGTH_EXCLUDE_MIN <= l <= LENGTH_EXCLUDE_MAX)


# ----------------------------------------------------------------------
# 5. Drawing (With Debug Visualization)
# ----------------------------------------------------------------------
def draw_results(frame, pellets, detections, debug_info):
    # Top Bar
    cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, 40), (20, 20, 20), -1)
    status_col = (0, 255, 0) if is_calibrated else (0, 0, 255)

    # Dynamic status message
    if is_calibrated:
        msg = f"CALIBRATED: {PIXELS_PER_MM:.2f} px/mm"
    else:
        msg = f"WAITING FOR '{CALIBRATION_CLASS_NAME}'..."

    cv2.putText(frame, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2)

    # 1. Draw Detections
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det['class_name']

        if label == CALIBRATION_CLASS_NAME:
            color = (255, 255, 0)  # Cyan for calibration zone
            thick = 2

            # Draw the box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            cv2.putText(frame, f"ZONE: {label}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # --- VISUALIZE TICKS (The "How it works" part) ---
            # If we have debug info for this zone, draw the dots
            if debug_info and 'tick_points' in debug_info:
                ticks = debug_info['tick_points']

                # Draw lines connecting ticks (to show the 'ruler' line)
                if len(ticks) > 1:
                    for i in range(len(ticks) - 1):
                        cv2.line(frame, ticks[i], ticks[i + 1], (0, 255, 255), 1)

                # Draw dots on exact tick centers
                for pt in ticks:
                    cv2.circle(frame, pt, 2, (0, 0, 255), -1)  # Red center
                    cv2.circle(frame, pt, 4, (0, 255, 255), 1)  # Yellow ring
        else:
            # Other objects (dimmed)
            color = (100, 100, 100)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 2. Draw Pellets
    for p in pellets:
        box = p['box']
        color = (0, 255, 0) if p['ok'] else (0, 0, 255)

        cv2.drawContours(frame, [box], 0, color, 2)

        lx = int(min(box[:, 0]))
        ly = int(min(box[:, 1]))

        lines = [f"D:{p['diameter']:.2f}", f"L:{p['length']:.2f}"]
        for i, txt in enumerate(lines):
            y = max(ly - 20 + (i * 15), 15)
            cv2.putText(frame, txt, (lx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(frame, txt, (lx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if not p['ok']:
            cv2.putText(frame, "!", (lx - 15, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    global PIXELS_PER_MM, is_calibrated, last_calibration_time

    print("--- Precision Pellet Inspector ---")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(3, DESIRED_WIDTH)
    cap.set(4, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    if not load_yolo_model(): return

    print(f"\nConfiguration: Looking for YOLO Class '{CALIBRATION_CLASS_NAME}'")

    debug_info = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Detect Objects
        detections = detect_objects(frame)

        # 2. Find Calibration Zone
        # This STRICTLY looks for the class name you set at top of script
        mm_zone = next((d for d in detections if d['class_name'] == CALIBRATION_CLASS_NAME), None)

        if mm_zone and (time.time() - last_calibration_time) > CALIBRATION_INTERVAL:
            result = analyze_mm_zone(frame, mm_zone['bbox'])

            if result:
                new_px_mm = result['px_per_mm']
                debug_info = result  # Save points for drawing

                # Smooth average
                if is_calibrated:
                    PIXELS_PER_MM = (PIXELS_PER_MM * 0.9) + (new_px_mm * 0.1)
                else:
                    PIXELS_PER_MM = new_px_mm
                    is_calibrated = True

                update_ranges()
                last_calibration_time = time.time()
            else:
                debug_info = None
        elif not mm_zone:
            debug_info = None

        # 3. Detect Pellets
        pellets = detect_pellets(frame)

        # 4. Draw
        frame = draw_results(frame, pellets, detections, debug_info)
        cv2.imshow("Inspector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()