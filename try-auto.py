import cv2
import numpy as np
import time
import sys
import math
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# EXACT NAME of the class in your YOLO model that represents the ticks/mm
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
CALIBRATION_INTERVAL = 0.5
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
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute area of both rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute Intersection over Union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def filter_detections(detections, iou_threshold=0.3):
    """
    Sorts detections by confidence.
    Removes lower confidence boxes if they overlap significantly
    with a higher confidence box OF THE SAME CLASS.
    """
    if not detections:
        return []

    # Sort by confidence (Highest first)
    detections.sort(key=lambda x: x['confidence'], reverse=True)

    final_detections = []

    while len(detections) > 0:
        # Pick the most confident box
        current = detections.pop(0)
        final_detections.append(current)

        # Compare with remaining boxes
        # We use a list comprehension to keep only non-overlapping boxes
        # OR boxes that are a different class (e.g. we keep 'ruler' AND 'mm_zone' even if they overlap)
        remaining = []
        for other in detections:
            overlap = calculate_iou(current['bbox'], other['bbox'])
            same_class = (current['class_name'] == other['class_name'])

            # If they are same class and overlap -> Discard (it's a duplicate)
            if same_class and overlap > iou_threshold:
                continue
            else:
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

    # Apply our custom filter to remove overlapping duplicates
    return filter_detections(raw_detections)


# ----------------------------------------------------------------------
# 3. OpenCV Calibration (Inside the MM Zone)
# ----------------------------------------------------------------------
def analyze_mm_zone(frame, bbox):
    x1, y1, x2, y2 = bbox

    # Validation
    h, w = frame.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h: return None

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return None

    # Image Processing to find tick marks
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold works best for black markings on metal/white
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 5)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ticks = []
    # Orientation Check
    is_horizontal = (x2 - x1) > (y2 - y1)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5: continue

        tx, ty, tw, th = cv2.boundingRect(cnt)
        ratio = float(tw) / th

        # Filter for line shapes
        if is_horizontal:
            # Ticks are vertical lines (tall and thin)
            if ratio < 0.8:
                ticks.append(tx + tw / 2)  # Use X position
        else:
            # Ticks are horizontal lines (wide and short)
            if ratio > 1.2:
                ticks.append(ty + th / 2)  # Use Y position

    if len(ticks) < 5: return None

    # Calculate median interval (Assuming 1mm gap)
    ticks.sort()
    intervals = []
    for i in range(len(ticks) - 1):
        gap = ticks[i + 1] - ticks[i]
        if 2 < gap < 100:  # Filter sanity
            intervals.append(gap)

    if len(intervals) < 3: return None

    median_gap = np.median(intervals)

    # Refine mean around median
    valid_gaps = [g for g in intervals if abs(g - median_gap) < 2.0]
    if len(valid_gaps) < 3: return None

    px_per_mm = np.mean(valid_gaps)

    # Sanity limits
    if px_per_mm < 2 or px_per_mm > 100: return None

    return px_per_mm


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

        # Euclidean distance between corners
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
# 5. Drawing
# ----------------------------------------------------------------------
def draw_results(frame, pellets, detections):
    # Top Bar
    cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, 40), (20, 20, 20), -1)
    status_col = (0, 255, 0) if is_calibrated else (0, 0, 255)
    msg = f"CALIBRATED: {PIXELS_PER_MM:.2f} px/mm" if is_calibrated else "WAITING FOR MM ZONE..."
    cv2.putText(frame, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2)

    # Draw YOLO Detections (Filtered)
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det['class_name']

        # Highlight the Calibration Zone differently
        if label == CALIBRATION_CLASS_NAME:
            color = (255, 255, 0)  # Cyan
            thick = 2
            lbl_text = f"CALIBRATION ZONE ({det['confidence']:.2f})"
        else:
            color = (100, 100, 100)  # Gray/Dim for general objects (Ruler, Caliper)
            thick = 1
            lbl_text = f"{label}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        cv2.putText(frame, lbl_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw Pellets
    for p in pellets:
        box = p['box']
        color = (0, 255, 0) if p['ok'] else (0, 0, 255)

        cv2.drawContours(frame, [box], 0, color, 2)

        # Labels
        lx = int(min(box[:, 0]))
        ly = int(min(box[:, 1]))

        # Text with outline
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

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Detect & Clean Duplicates
        detections = detect_objects(frame)

        # 2. Check for Calibration Zone
        mm_zone = next((d for d in detections if d['class_name'] == CALIBRATION_CLASS_NAME), None)

        if mm_zone and (time.time() - last_calibration_time) > CALIBRATION_INTERVAL:
            new_px_mm = analyze_mm_zone(frame, mm_zone['bbox'])
            if new_px_mm:
                # Weighted average to smooth jitter
                if is_calibrated:
                    PIXELS_PER_MM = (PIXELS_PER_MM * 0.9) + (new_px_mm * 0.1)
                else:
                    PIXELS_PER_MM = new_px_mm
                    is_calibrated = True

                update_ranges()
                last_calibration_time = time.time()

        # 3. Detect Pellets
        pellets = detect_pellets(frame)

        # 4. Display
        frame = draw_results(frame, pellets, detections)
        cv2.imshow("Inspector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()