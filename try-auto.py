import cv2
import numpy as np
import time
import sys
import math
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Global Settings
# ----------------------------------------------------------------------
PIXELS_PER_MM = 6.0  # Initial fallback (will update automatically)
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0

# YOLO Model State
yolo_model = None
last_calibration_time = 0
CALIBRATION_INTERVAL = 0.5  # Fast updates since the box is specific
is_calibrated = False

# Detection Tuning
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 25000

DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720


def load_yolo_model():
    global yolo_model
    try:
        # Load your custom model trained on the "mm_scale"
        yolo_model = YOLO("yolo/best.pt")
        print("✓ Custom YOLO model loaded successfully")
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
# YOLO Detection (Targeting "mm_scale" class)
# ----------------------------------------------------------------------
def detect_calibration_zone(frame):
    global yolo_model
    if yolo_model is None:
        return None
    try:
        # We assume the model detects the 'mm_scale'.
        # We take the detection with highest confidence.
        results = yolo_model(frame, conf=0.4, verbose=False)
        if len(results) > 0 and len(results[0].boxes) > 0:
            best_box = max(results[0].boxes, key=lambda x: x.conf[0])
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
            return {'bbox': (int(x1), int(y1), int(x2), int(y2))}
    except Exception as e:
        print(f"Error in YOLO: {e}")
    return None


# ----------------------------------------------------------------------
# ANALYZE THE MM STRIP
# ----------------------------------------------------------------------
def analyze_mm_scale(frame, bbox):
    x1, y1, x2, y2 = bbox

    # 1. Crop to the detected "mm scale" box
    # We use very little padding because the YOLO box should be precise
    h, w = frame.shape[:2]
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(w, x2), min(h, y2)

    roi = frame[cy1:cy2, cx1:cx2]
    if roi.size == 0: return None

    # 2. Adaptive Threshold to find the black tick lines
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Tuned for reading black ticks on metal/white
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 5)

    # 3. Find Contours (The Ticks)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_ticks = []

    # Determine orientation of the strip
    # If the bounding box is wider than it is tall, the ruler is horizontal
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    is_horizontal = bbox_w > bbox_h

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5: continue  # Ignore dust

        x, y, w_c, h_c = cv2.boundingRect(cnt)
        aspect_ratio = float(w_c) / h_c

        # Filter: We want lines perpendicular to the ruler direction
        if is_horizontal:
            # Ruler is horizontal, ticks are vertical lines (Height > Width)
            if aspect_ratio < 0.8:
                center_x = x + w_c / 2
                valid_ticks.append({'pos': center_x, 'center': (center_x, y + h_c / 2)})
        else:
            # Ruler is vertical, ticks are horizontal lines (Width > Height)
            if aspect_ratio > 1.2:
                center_y = y + h_c / 2
                valid_ticks.append({'pos': center_y, 'center': (x + w_c / 2, center_y)})

    if len(valid_ticks) < 4: return None

    # 4. Calculate Intervals (Gaps between ticks)
    valid_ticks.sort(key=lambda x: x['pos'])
    positions = [t['pos'] for t in valid_ticks]
    intervals = []

    for i in range(len(positions) - 1):
        gap = positions[i + 1] - positions[i]
        # Valid 1mm gap in pixels? (adjust range if 4k camera)
        if 2 < gap < 100:
            intervals.append(gap)

    if len(intervals) < 3: return None

    # 5. Median Logic (The magic step)
    # The median gap is the most common distance, which corresponds to 1mm.
    median_gap = np.median(intervals)

    # Filter out outliers (missing ticks or number gaps)
    clean_intervals = [i for i in intervals if abs(i - median_gap) < 2.0]

    if len(clean_intervals) < 3: return None

    final_px_per_mm = np.mean(clean_intervals)

    # Safety bounds
    if final_px_per_mm < 2.0 or final_px_per_mm > 100.0: return None

    return {
        "pixels_per_mm": final_px_per_mm,
        "bbox": bbox,
        "roi_offset": (cx1, cy1),
        "tick_centers": [t['center'] for t in valid_ticks]
    }


# ----------------------------------------------------------------------
# PELLET DETECTION (High Precision)
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
        (center_x, center_y), (w, h), angle = rect
        box = cv2.boxPoints(rect)
        box = np.intp(box)

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
# Draw Overlay (Visualizes the YOLO box + Ticks)
# ----------------------------------------------------------------------
def draw_overlay(frame, pellets, calib_data):
    # Top Info Bar
    cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, 40), (20, 20, 20), -1)

    if is_calibrated:
        status_color = (0, 255, 0)
        msg = f"CALIBRATED: {PIXELS_PER_MM:.2f} px/mm"
    else:
        status_color = (0, 0, 255)
        msg = "SEARCHING FOR MM SCALE..."

    cv2.putText(frame, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # --- VISUALIZE CALIBRATION ---
    if calib_data:
        bbox = calib_data['bbox']
        off_x, off_y = calib_data['roi_offset']

        # 1. Draw the YOLO Bounding Box (Cyan)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
        cv2.putText(frame, "MM Scale", (bbox[0], bbox[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 2. Draw the detected ticks inside the box (Yellow Dots)
        # This confirms to the user that we are measuring the right thing
        if 'tick_centers' in calib_data:
            for i, (tx, ty) in enumerate(calib_data['tick_centers']):
                # Draw every 2nd tick to keep it clean
                if i % 2 == 0:
                    cv2.circle(frame, (int(tx + off_x), int(ty + off_y)), 2, (0, 255, 255), -1)

    # --- DRAW PELLETS ---
    for p in pellets:
        box = p['box']
        color = (0, 255, 0) if p['within_tolerance'] else (0, 0, 255)
        text_col = (255, 255, 255)

        cv2.drawContours(frame, [box], 0, color, 2)

        left_x = int(min(box[:, 0]))
        top_y = int(min(box[:, 1]))
        text_y = max(top_y - 10, 20)

        # Draw text with outline for readability
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

    print("Ready. Point camera at the 'MM Scale' to calibrate.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        current_time = time.time()

        # 1. Detect the specific "MM Scale" area using YOLO
        scale_zone = detect_calibration_zone(frame)
        calib_data = None

        # 2. Analyze that specific zone
        if scale_zone:
            # Prepare data for drawing
            calib_data = {'bbox': scale_zone['bbox']}

            # Recalibrate
            if (current_time - last_calibration_time) > CALIBRATION_INTERVAL:
                analysis = analyze_mm_scale(frame, scale_zone['bbox'])

                if analysis:
                    new_px_mm = analysis['pixels_per_mm']

                    # Smooth Averaging for stability
                    if is_calibrated:
                        PIXELS_PER_MM = (PIXELS_PER_MM * 0.9) + (new_px_mm * 0.1)
                    else:
                        PIXELS_PER_MM = new_px_mm
                        is_calibrated = True

                    update_ranges()
                    last_calibration_time = current_time

                    # Add tick locations to overlay data for visualization
                    calib_data['tick_centers'] = analysis['tick_centers']
                    calib_data['roi_offset'] = analysis['roi_offset']

        pellets = detect_pellets(frame)
        frame = draw_overlay(frame, pellets, calib_data)

        cv2.imshow("MM Scale Calibration Inspector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()