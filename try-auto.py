import cv2
import numpy as np
import time
import sys
import math
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# CHANGE THIS to match the class name in your YOLO dataset for the marks
# Examples: "mm_zone", "ticks", "scale", "measurement_area"
TARGET_CLASS_FOR_CALIBRATION = "mm_zone"

PIXELS_PER_MM = 6.0  # Fallback
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0

# Detection Settings
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 25000
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720

# State
yolo_model = None
model_class_names = {}
last_calibration_time = 0
CALIBRATION_INTERVAL = 0.5
is_calibrated = False


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
def load_yolo_model():
    global yolo_model, model_class_names
    try:
        yolo_model = YOLO("yolo/best.pt")
        model_class_names = yolo_model.names  # Get class names (0: 'ruler', 1: 'mm_zone', etc.)
        print(f"✓ Model loaded. Classes found: {model_class_names}")
        return True
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
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
# Helper Functions
# ----------------------------------------------------------------------
def is_within_tolerance(diameter: float, length: float) -> bool:
    return (DIAMETER_MIN <= diameter <= DIAMETER_MAX and
            LENGTH_MIN <= length <= LENGTH_MAX)


def should_process_pellet(diameter: float, length: float) -> bool:
    if diameter < 0.5 or length < 0.5: return False
    return (DIAMETER_EXCLUDE_MIN <= diameter <= DIAMETER_EXCLUDE_MAX and
            LENGTH_EXCLUDE_MIN <= length <= LENGTH_EXCLUDE_MAX)


def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ----------------------------------------------------------------------
# YOLO Detection (Returns ALL objects)
# ----------------------------------------------------------------------
def detect_all_objects(frame):
    global yolo_model
    if yolo_model is None:
        return []

    detections = []
    try:
        results = yolo_model(frame, conf=0.4, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                class_name = model_class_names.get(cls_id, "unknown")
                conf = float(box.conf[0])

                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'class_name': class_name,
                    'confidence': conf
                })
    except Exception as e:
        print(f"Error in YOLO: {e}")

    return detections


# ----------------------------------------------------------------------
# Analyze Specific MM Zone for Math
# ----------------------------------------------------------------------
def analyze_mm_scale(frame, bbox):
    x1, y1, x2, y2 = bbox

    # Crop to the MM Zone
    h, w = frame.shape[:2]
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(w, x2), min(h, y2)
    roi = frame[cy1:cy2, cx1:cx2]
    if roi.size == 0: return None

    # Adaptive Threshold to find ticks
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 5)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_ticks = []

    # Determine orientation
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    is_horizontal = bbox_w > bbox_h

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5: continue

        x, y, w_c, h_c = cv2.boundingRect(cnt)
        aspect_ratio = float(w_c) / h_c

        # Filter shapes
        if is_horizontal:
            if aspect_ratio < 0.8:  # Vertical lines
                valid_ticks.append({'pos': x + w_c / 2, 'center': (x + w_c / 2, y + h_c / 2)})
        else:
            if aspect_ratio > 1.2:  # Horizontal lines
                valid_ticks.append({'pos': y + h_c / 2, 'center': (x + w_c / 2, y + h_c / 2)})

    if len(valid_ticks) < 4: return None

    # Calculate Intervals
    valid_ticks.sort(key=lambda x: x['pos'])
    positions = [t['pos'] for t in valid_ticks]
    intervals = []

    for i in range(len(positions) - 1):
        gap = positions[i + 1] - positions[i]
        if 2 < gap < 100:
            intervals.append(gap)

    if len(intervals) < 3: return None

    # Median Logic (1mm assumption)
    median_gap = np.median(intervals)
    clean_intervals = [i for i in intervals if abs(i - median_gap) < 2.0]

    if len(clean_intervals) < 3: return None

    final_px_per_mm = np.mean(clean_intervals)

    if final_px_per_mm < 2.0 or final_px_per_mm > 100.0: return None

    return {
        "pixels_per_mm": final_px_per_mm,
        "bbox": bbox,
        "roi_offset": (cx1, cy1),
        "tick_centers": [t['center'] for t in valid_ticks]
    }


# ----------------------------------------------------------------------
# Pellet Detection
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
        area = cv2.contourArea(cnt)
        if not (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA): continue

        rect = cv2.minAreaRect(cnt)
        box = np.intp(cv2.boxPoints(rect))

        edge1 = get_distance(box[0], box[1])
        edge2 = get_distance(box[1], box[2])

        width_px = min(edge1, edge2)
        height_px = max(edge1, edge2)

        diameter = width_px / PIXELS_PER_MM
        length = height_px / PIXELS_PER_MM

        if should_process_pellet(diameter, length):
            pellets.append({
                'box': box,
                'diameter': diameter,
                'length': length,
                'within_tolerance': is_within_tolerance(diameter, length)
            })
    return pellets


# ----------------------------------------------------------------------
# Display Logic
# ----------------------------------------------------------------------
def draw_overlay(frame, pellets, detections, calibration_data):
    # 1. Draw Top Bar
    cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, 40), (20, 20, 20), -1)

    if is_calibrated:
        status_color = (0, 255, 0)
        msg = f"CALIBRATED: {PIXELS_PER_MM:.2f} px/mm"
    else:
        status_color = (0, 0, 255)
        msg = f"Waiting for '{TARGET_CLASS_FOR_CALIBRATION}'..."

    cv2.putText(frame, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # 2. Draw ALL YOLO Detections (Ruler, Caliper, etc.)
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det['class_name']
        conf = det['confidence']

        # Color Coding
        if label == TARGET_CLASS_FOR_CALIBRATION:
            # CYAN for the active calibration zone
            color = (255, 255, 0)
            thickness = 2
        else:
            # GRAY/YELLOW for general objects (Ruler, Caliper)
            color = (0, 200, 255)
            thickness = 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label Text
        txt = f"{label} {conf:.2f}"
        cv2.putText(frame, txt, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 3. Draw Specific Calibration Ticks (Visual Debugging)
    if calibration_data:
        off_x, off_y = calibration_data['roi_offset']
        if 'tick_centers' in calibration_data:
            for i, (tx, ty) in enumerate(calibration_data['tick_centers']):
                if i % 2 == 0:  # Draw every other tick
                    cv2.circle(frame, (int(tx + off_x), int(ty + off_y)), 2, (0, 255, 255), -1)

    # 4. Draw Pellets
    for p in pellets:
        box = p['box']
        color = (0, 255, 0) if p['within_tolerance'] else (0, 0, 255)
        text_col = (255, 255, 255)

        cv2.drawContours(frame, [box], 0, color, 2)

        left_x = int(min(box[:, 0]))
        top_y = int(min(box[:, 1]))
        text_y = max(top_y - 10, 20)

        # Draw text with outline
        labels = [f"D:{p['diameter']:.2f}", f"L:{p['length']:.2f}"]

        for i, line in enumerate(labels):
            y_pos = text_y - (15 * (1 - i))
            cv2.putText(frame, line, (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(frame, line, (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_col, 1)

        if not p['within_tolerance']:
            cv2.putText(frame, "!", (left_x - 15, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame


# ----------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------
def main():
    global PIXELS_PER_MM, last_calibration_time, is_calibrated

    print("Starting System...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    if not load_yolo_model(): return

    print(f"\nSystem Ready.")
    print(f"1. Detects ALL objects trained in model.")
    print(f"2. Uses class '{TARGET_CLASS_FOR_CALIBRATION}' for calibration math.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        current_time = time.time()

        # 1. Get ALL detections (Ruler, Caliper, MM Zone)
        detections = detect_all_objects(frame)

        # 2. Find the specific 'mm_zone' for math
        calibration_data = None
        target_zone = next((d for d in detections if d['class_name'] == TARGET_CLASS_FOR_CALIBRATION), None)

        if target_zone:
            # We found the MM Zone. Do we need to recalc math?
            calibration_data = {'bbox': target_zone['bbox'], 'roi_offset': (0, 0)}

            if (current_time - last_calibration_time) > CALIBRATION_INTERVAL:
                analysis = analyze_mm_scale(frame, target_zone['bbox'])

                if analysis:
                    new_px_mm = analysis['pixels_per_mm']

                    if is_calibrated:
                        PIXELS_PER_MM = (PIXELS_PER_MM * 0.9) + (new_px_mm * 0.1)
                    else:
                        PIXELS_PER_MM = new_px_mm
                        is_calibrated = True

                    update_ranges()
                    last_calibration_time = current_time

                    # Store details for visualization
                    calibration_data['tick_centers'] = analysis['tick_centers']
                    calibration_data['roi_offset'] = analysis['roi_offset']

        # 3. Pellet Detection
        pellets = detect_pellets(frame)

        # 4. Draw Everything
        frame = draw_overlay(frame, pellets, detections, calibration_data)

        cv2.imshow("Multi-Object Inspector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()