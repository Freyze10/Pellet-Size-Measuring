import cv2
import numpy as np
import time
import sys
import math
from ultralytics import YOLO
from collections import defaultdict

# ----------------------------------------------------------------------
# Global Calibration
# ----------------------------------------------------------------------
PIXELS_PER_MM = 6.0  # Default fallback
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0

# YOLO Model
yolo_model = None
last_calibration_time = 0
CALIBRATION_INTERVAL = 1.0  # Faster updates

# Calibration State
is_calibrated = False


def load_yolo_model():
    global yolo_model
    try:
        # Ensure you have the correct path to your weights
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

MIN_CONTOUR_AREA = 50
MAX_CONTOUR_AREA = 25000

DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720


# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def is_within_tolerance(diameter: float, length: float) -> bool:
    return (DIAMETER_MIN <= diameter <= DIAMETER_MAX and
            LENGTH_MIN <= length <= LENGTH_MAX)


def should_process_pellet(diameter: float, length: float) -> bool:
    # Filter out tiny noise or huge blobs
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
            # Find the box with highest confidence
            best_box = max(results[0].boxes, key=lambda x: x.conf[0])
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
            conf = best_box.conf[0].cpu().numpy()
            return {
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(conf)
            }
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
    return None


# ----------------------------------------------------------------------
# FIXED: Ruler Analysis using Interval Statistics
# ----------------------------------------------------------------------
def analyze_ruler_region(frame, bbox):
    x1, y1, x2, y2 = bbox

    # 1. Crop with slight padding (be careful not to go out of bounds)
    h, w = frame.shape[:2]
    pad = 5
    cx1, cy1 = max(0, x1 + pad), max(0, y1 + pad)
    cx2, cy2 = min(w, x2 - pad), min(h, y2 - pad)

    roi = frame[cy1:cy2, cx1:cx2]
    if roi.size == 0: return None

    # 2. Preprocessing to highlight dark markings
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding handles uneven lighting better than global threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 8)

    # 3. Find Contours (Potential Ticks)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_ticks = []

    # Determine orientation of the ruler based on bounding box aspect ratio
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    is_vertical = bbox_h > bbox_w

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5 or area > 1000: continue  # Filter noise and huge blobs

        rect = cv2.boundingRect(cnt)
        rx, ry, rw, rh = rect

        # 4. Aspect Ratio Filtering
        # Ticks are either thin lines (horiz or vert)
        aspect_ratio = float(rw) / rh

        is_tick = False
        if is_vertical:
            # If ruler is vertical, ticks are horizontal lines (width > height)
            if aspect_ratio > 1.5:
                valid_ticks.append({'pos': ry + rh / 2, 'center': (rx + rw / 2, ry + rh / 2)})
        else:
            # If ruler is horizontal, ticks are vertical lines (height > width)
            if aspect_ratio < 0.6:
                valid_ticks.append({'pos': rx + rw / 2, 'center': (rx + rw / 2, ry + rh / 2)})

    if len(valid_ticks) < 5:
        return None

    # 5. Calculate Intervals
    # Sort ticks by position
    valid_ticks.sort(key=lambda x: x['pos'])

    positions = [t['pos'] for t in valid_ticks]
    intervals = []

    for i in range(len(positions) - 1):
        gap = positions[i + 1] - positions[i]
        # Filter logic: Ignore extremely small gaps (double detections) or huge gaps
        if 2 < gap < 100:
            intervals.append(gap)

    if len(intervals) < 3:
        return None

    # 6. Statistical Analysis (The Key Fix)
    # We look for the most common small interval.
    # Standard rulers have 1mm marks.

    median_gap = np.median(intervals)

    # If the variance is high, the data is noisy. Remove outliers.
    clean_intervals = [i for i in intervals if abs(i - median_gap) < 2.0]

    if len(clean_intervals) < 3:
        return None

    final_px_per_mm = np.mean(clean_intervals)

    # Sanity Check:
    # If pixels per mm is too small (< 2), it's likely noise.
    # If it's too big (> 50), the camera is too close or it detected cm marks as mm.
    if final_px_per_mm < 2.0 or final_px_per_mm > 50.0:
        return None

    return {
        "pixels_per_mm": final_px_per_mm,
        "bbox": bbox,
        "tick_count": len(valid_ticks),
        "roi_offset": (cx1, cy1),
        "tick_centers": [t['center'] for t in valid_ticks]
    }


# ----------------------------------------------------------------------
# Pellet Detection
# ----------------------------------------------------------------------
def detect_pellets(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)

    # Use simple thresholding or adaptive depending on background
    # Setup for light pellets on dark background or vice versa?
    # Assuming dark pellets on light background (Adaptive usually works best)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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

        # MinAreaRect returns width/height in arbitrary order based on angle
        dim1 = w
        dim2 = h

        width_px = min(dim1, dim2)  # Diameter is usually the smaller dimension
        height_px = max(dim1, dim2)  # Length is the longer dimension

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
# Draw Overlay
# ----------------------------------------------------------------------
def draw_overlay(frame, pellets, ruler_data):
    # Draw Calibration Info
    cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, 40), (30, 30, 30), -1)

    if is_calibrated:
        status_color = (0, 255, 0)
        msg = f"CALIBRATED: {PIXELS_PER_MM:.2f} px/mm"
    else:
        status_color = (0, 0, 255)
        msg = "UNCALIBRATED - Using Default"

    cv2.putText(frame, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Draw Ruler Detection Debug
    if ruler_data:
        bbox = ruler_data['bbox']
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
        cv2.putText(frame, "Ruler Found", (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Visualize the detected ticks (Green dots)
        offset_x, offset_y = ruler_data.get('roi_offset', (0, 0))
        if 'tick_centers' in ruler_data:
            for tx, ty in ruler_data['tick_centers']:
                # Draw only every 5th tick to save clutter
                cv2.circle(frame, (int(tx + offset_x), int(ty + offset_y)), 2, (0, 255, 255), -1)

    # Draw Pellets
    total = len(pellets)
    good = sum(1 for p in pellets if p['within_tolerance'])

    stats = f"Total: {total} | Good: {good} | Bad: {total - good}"
    cv2.putText(frame, stats, (DESIRED_WIDTH - 400, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for p in pellets:
        box = p['box']
        color = (0, 255, 0) if p['within_tolerance'] else (0, 0, 255)

        cv2.drawContours(frame, [box], 0, color, 2)

        lbl_x = int(box[0][0])
        lbl_y = int(box[0][1])

        # Label Background
        cv2.putText(frame, f"{p['diameter']:.1f}x{p['length']:.1f}", (lbl_x, lbl_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


# ----------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------
def main():
    global PIXELS_PER_MM, last_calibration_time, is_calibrated

    print("Initializing Camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Try index 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Turn off autofocus for consistent measurement

    if not load_yolo_model():
        return

    print("Starting Loop. Present ruler to camera.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        current_time = time.time()

        # 1. Detect Ruler
        ruler_bbox = detect_ruler_yolo(frame)
        ruler_data = None

        # 2. Analyze Ruler for Calibration
        if ruler_bbox:
            # Draw bbox immediately for feedback
            ruler_data = {'bbox': ruler_bbox['bbox']}

            # Only recalibrate periodically to prevent jitter
            if (current_time - last_calibration_time) > CALIBRATION_INTERVAL:
                analysis = analyze_ruler_region(frame, ruler_bbox['bbox'])

                if analysis:
                    # Apply Smooth Averaging to prevent jumping values
                    new_px_mm = analysis['pixels_per_mm']
                    if is_calibrated:
                        PIXELS_PER_MM = (PIXELS_PER_MM * 0.8) + (new_px_mm * 0.2)
                    else:
                        PIXELS_PER_MM = new_px_mm
                        is_calibrated = True

                    update_ranges()
                    last_calibration_time = current_time
                    ruler_data = analysis  # Update data for overlay
                    print(f"Calibration Updated: {PIXELS_PER_MM:.2f} px/mm")

        # 3. Detect and Measure Pellets
        pellets = detect_pellets(frame)

        # 4. Draw
        frame = draw_overlay(frame, pellets, ruler_data)

        cv2.imshow("Smart Calibration Inspector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()