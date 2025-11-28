import cv2
import numpy as np
import time
import sys
import math
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Global Settings
# ----------------------------------------------------------------------
PIXELS_PER_MM = 6.0  # Initial fallback
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0

# YOLO Model State
yolo_model = None
last_calibration_time = 0
CALIBRATION_INTERVAL = 1.0
is_calibrated = False

MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 20000

DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720


def load_yolo_model():
    global yolo_model
    try:
        yolo_model = YOLO("yolo/best.pt")
        print("✓ YOLO model loaded successfully")
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
# YOLO Ruler Detection
# ----------------------------------------------------------------------
def detect_ruler_yolo(frame):
    global yolo_model
    if yolo_model is None:
        return None
    try:
        results = yolo_model(frame, conf=0.4, verbose=False)
        if len(results) > 0 and len(results[0].boxes) > 0:
            best_box = max(results[0].boxes, key=lambda x: x.conf[0])
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
            return {'bbox': (int(x1), int(y1), int(x2), int(y2))}
    except Exception as e:
        print(f"Error in YOLO: {e}")
    return None


# ----------------------------------------------------------------------
# ROBUST AUTO-CALIBRATION (Interval Statistics)
# ----------------------------------------------------------------------
def analyze_ruler_region(frame, bbox):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    pad = 5
    cx1, cy1 = max(0, x1 + pad), max(0, y1 + pad)
    cx2, cy2 = min(w, x2 - pad), min(h, y2 - pad)

    roi = frame[cy1:cy2, cx1:cx2]
    if roi.size == 0: return None

    # 1. Preprocessing (Adaptive Threshold to find ticks)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 8)

    # 2. Find Tick Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_ticks = []
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    is_vertical = bbox_h > bbox_w

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5 or area > 1000: continue

        x, y, w_c, h_c = cv2.boundingRect(cnt)
        aspect_ratio = float(w_c) / h_c

        # Filter based on shape (lines vs blobs)
        if is_vertical:
            if aspect_ratio > 1.5:  # Horizontal line
                valid_ticks.append({'pos': y + h_c / 2})
        else:
            if aspect_ratio < 0.6:  # Vertical line
                valid_ticks.append({'pos': x + w_c / 2})

    if len(valid_ticks) < 5: return None

    # 3. Calculate Intervals
    valid_ticks.sort(key=lambda x: x['pos'])
    positions = [t['pos'] for t in valid_ticks]
    intervals = []

    for i in range(len(positions) - 1):
        gap = positions[i + 1] - positions[i]
        if 2 < gap < 100:
            intervals.append(gap)

    if len(intervals) < 3: return None

    # 4. Statistical Median (Finds the 1mm gap)
    median_gap = np.median(intervals)
    clean_intervals = [i for i in intervals if abs(i - median_gap) < 2.0]

    if len(clean_intervals) < 3: return None

    final_px_per_mm = np.mean(clean_intervals)

    if final_px_per_mm < 2.0 or final_px_per_mm > 50.0: return None

    return {
        "pixels_per_mm": final_px_per_mm,
        "bbox": bbox,
        "roi_offset": (cx1, cy1)
    }


# ----------------------------------------------------------------------
# PRECISE PELLET DETECTION (From your requested code)
# ----------------------------------------------------------------------
def detect_pellets(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Bilateral Filter: Keeps edges sharp
    blur = cv2.bilateralFilter(gray, 9, 75, 75)

    # 2. Adaptive Threshold: Tight block size (11) for precision
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 3. Morphological Operations: Clean noise without shrinking object
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pellets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA):
            continue

        rect = cv2.minAreaRect(cnt)
        (center_x, center_y), (w, h), angle = rect
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # 4. Calculate dimensions manually from box corners
        edge1 = get_distance(box[0], box[1])
        edge2 = get_distance(box[1], box[2])

        if edge1 < edge2:
            width_px = edge1
            height_px = edge2
        else:
            width_px = edge2
            height_px = edge1

        # Convert to mm
        diameter = width_px / PIXELS_PER_MM
        length = height_px / PIXELS_PER_MM

        if should_process_pellet(diameter, length):
            pellets.append({
                'box': box,
                'center': (center_x, center_y),
                'diameter': diameter,
                'length': length,
                'within_tolerance': is_within_tolerance(diameter, length)
            })
    return pellets


# ----------------------------------------------------------------------
# Draw Overlay (Clean text, no background box)
# ----------------------------------------------------------------------
def draw_overlay(frame, pellets, ruler_data):
    # --- UI: Top Bar ---
    cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, 40), (20, 20, 20), -1)

    if is_calibrated:
        status_color = (0, 255, 0)
        msg = f"AUTO-CALIBRATED: {PIXELS_PER_MM:.2f} px/mm"
    else:
        status_color = (0, 0, 255)
        msg = "UNCALIBRATED - Show Ruler"

    cv2.putText(frame, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # --- UI: Ruler Box ---
    if ruler_data:
        bbox = ruler_data['bbox']
        # Thin yellow line for ruler
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 1)
        cv2.putText(frame, "Ruler", (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # --- UI: Pellets ---
    total = len(pellets)
    good = sum(1 for p in pellets if p['within_tolerance'])

    stats = f"Total: {total} | Good: {good} | Bad: {total - good}"
    cv2.putText(frame, stats, (DESIRED_WIDTH - 450, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for p in pellets:
        box = p['box']

        # Color logic
        if p['within_tolerance']:
            color = (0, 255, 0)  # Green
            text_color = (255, 255, 255)  # White text
        else:
            color = (0, 0, 255)  # Red
            text_color = (100, 100, 255)  # Red-ish text

        # Draw contour
        cv2.drawContours(frame, [box], 0, color, 2)

        # Position for text
        left_x = int(min(box[:, 0]))
        top_y = int(min(box[:, 1]))
        text_y = max(top_y - 10, 20)

        # Text strings
        txt_d = f"D:{p['diameter']:.2f}"
        txt_l = f"L:{p['length']:.2f}"

        # Draw Text with Outline (Stroke) instead of Background Box
        # This makes it readable on any background without the "ugly" black box

        # Diameter Text
        cv2.putText(frame, txt_d, (left_x, text_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Black Outline
        cv2.putText(frame, txt_d, (left_x, text_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)  # Inner Color

        # Length Text
        cv2.putText(frame, txt_l, (left_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Black Outline
        cv2.putText(frame, txt_l, (left_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)  # Inner Color

        if not p['within_tolerance']:
            # Add an exclamation mark
            cv2.putText(frame, "!", (left_x - 15, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, "!", (left_x - 15, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame


# ----------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------
def main():
    global PIXELS_PER_MM, last_calibration_time, is_calibrated

    print("Loading...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    if not load_yolo_model(): return

    print("System Ready. Show ruler for auto-calibration.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        current_time = time.time()

        # 1. Detect Ruler
        ruler_bbox = detect_ruler_yolo(frame)
        ruler_data = None

        # 2. Analyze Ruler (Fixed Logic)
        if ruler_bbox:
            ruler_data = {'bbox': ruler_bbox['bbox']}

            if (current_time - last_calibration_time) > CALIBRATION_INTERVAL:
                analysis = analyze_ruler_region(frame, ruler_bbox['bbox'])
                if analysis:
                    new_px_mm = analysis['pixels_per_mm']
                    # Smooth averaging
                    if is_calibrated:
                        PIXELS_PER_MM = (PIXELS_PER_MM * 0.9) + (new_px_mm * 0.1)
                    else:
                        PIXELS_PER_MM = new_px_mm
                        is_calibrated = True

                    update_ranges()
                    last_calibration_time = current_time
                    ruler_data = analysis

        # 3. Detect Pellets (Precise Logic)
        pellets = detect_pellets(frame)

        # 4. Draw Overlay (Clean Text)
        frame = draw_overlay(frame, pellets, ruler_data)

        cv2.imshow("Auto-Calibrated Precision Inspector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()