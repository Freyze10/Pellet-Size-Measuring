import cv2
import numpy as np
import time
import sys
import math
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# PATH TO YOUR TRAINED YOLOv11 MODEL
YOLO_MODEL_PATH = "yolo/best.pt"

# Global Calibration Defaults
PIXELS_PER_MM = 6.0
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0

# ----------------------------------------------------------------------
# Global State Variables
# ----------------------------------------------------------------------
model = None  # To hold the loaded YOLO model
DIAMETER_MIN = 0;
DIAMETER_MAX = 0
LENGTH_MIN = 0;
LENGTH_MAX = 0
DIAMETER_EXCLUDE_MIN = 0;
DIAMETER_EXCLUDE_MAX = 0
LENGTH_EXCLUDE_MIN = 0;
LENGTH_EXCLUDE_MAX = 0


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

MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 20000
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720

# ----------------------------------------------------------------------
# Ruler Calibration State
# ----------------------------------------------------------------------
in_ruler_calib_mode = False
calibration_frozen_frame = None
calibration_result = None
yolo_detection_box = None  # Stores (x1, y1, x2, y2) of the detected mm zone

# UI Button Geometry
RULER_PANEL_X, RULER_PANEL_Y = 10, 80
RULER_PANEL_W, RULER_PANEL_H = 450, 320
APPLY_BTN = (RULER_PANEL_X + 20, RULER_PANEL_Y + 260, 160, 40)
CANCEL_BTN = (RULER_PANEL_X + 200, RULER_PANEL_Y + 260, 160, 40)

detected_unit = None


# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def is_within_tolerance(diameter: float, length: float) -> bool:
    return (DIAMETER_MIN <= diameter <= DIAMETER_MAX and
            LENGTH_MIN <= length <= LENGTH_MAX)


def should_process_pellet(diameter: float, length: float) -> bool:
    return (DIAMETER_EXCLUDE_MIN <= diameter <= DIAMETER_EXCLUDE_MAX and
            LENGTH_EXCLUDE_MIN <= length <= LENGTH_EXCLUDE_MAX)


def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def load_yolo_model():
    global model
    try:
        print(f"Loading YOLO model from: {YOLO_MODEL_PATH}...")
        model = YOLO(YOLO_MODEL_PATH)
        print("✓ YOLO model loaded successfully.")
    except Exception as e:
        print(f"✗ Error loading YOLO model: {e}")
        print("Ensure 'ultralytics' is installed and path is correct.")
        model = None


# ----------------------------------------------------------------------
# YOLO Detection Logic
# ----------------------------------------------------------------------
def run_yolo_calibration(frame):
    """
    Runs YOLO on the frame, finds the 'mm zone', and passes that ROI
    to the tick analysis function.
    """
    if model is None:
        print("YOLO model not loaded.")
        return None, None

    # Run inference
    results = model(frame, verbose=False)

    best_box = None
    best_conf = 0

    # Iterate results to find the best "mm zone"
    # Adjust logic below depending on your specific class names
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            # LOGIC: Prioritize "mm" or "zone" labels, otherwise take highest conf ruler
            # Change "mm" to match your exact class name if needed
            is_target_zone = "mm" in class_name.lower() or "zone" in class_name.lower()

            if is_target_zone and conf > 0.4:
                if conf > best_conf:
                    best_conf = conf
                    best_box = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

    # Fallback: If no specific "mm zone" found, use the generic "ruler" detection
    if best_box is None:
        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) > 0.5:
                    best_box = box.xyxy[0].cpu().numpy()
                    break

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box)

        # Add padding purely for the crop calculation (optional)
        h, w = frame.shape[:2]
        pad = 5
        x1 = max(0, x1 - pad);
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad);
        y2 = min(h, y2 + pad)

        # Call the existing math-heavy analysis on this box
        calib_data = analyze_ruler_box(frame, (x1, y1), (x2, y2))

        return (x1, y1, x2, y2), calib_data

    return None, None


# ----------------------------------------------------------------------
# Ruler Tick Detection (Math/CV Logic)
# ----------------------------------------------------------------------
def analyze_ruler_box(frame, box_start, box_end):
    """Analyze the selected ruler box for ticks."""
    x1, y1 = min(box_start[0], box_end[0]), min(box_start[1], box_end[1])
    x2, y2 = max(box_start[0], box_end[0]), max(box_start[1], box_end[1])

    if x2 - x1 < 20 or y2 - y1 < 20: return None  # Too small

    roi = frame[y1:y2, x1:x2].copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 30, 100, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                            minLineLength=10, maxLineGap=5)

    if lines is None or len(lines) < 3: return None

    # Group lines to find dominant angle
    line_data = []
    for line in lines:
        lx1, ly1, lx2, ly2 = line[0]
        angle = math.degrees(math.atan2(ly2 - ly1, lx2 - lx1))
        if angle < 0: angle += 180
        line_data.append({'angle': angle, 'midpoint': ((lx1 + lx2) / 2, (ly1 + ly2) / 2)})

    # Bin angles
    from collections import defaultdict
    angle_groups = defaultdict(list)
    for ld in line_data:
        angle_bin = round(ld['angle'] / 5) * 5
        angle_groups[angle_bin].append(ld)

    if not angle_groups: return None
    dominant_angle_bin = max(angle_groups.keys(), key=lambda k: len(angle_groups[k]))
    tick_lines = angle_groups[dominant_angle_bin]

    if len(tick_lines) < 3: return None

    # Project to 1D axis
    avg_tick_angle = np.mean([ld['angle'] for ld in tick_lines])
    ruler_angle = (avg_tick_angle + 90) % 180
    ruler_angle_rad = math.radians(ruler_angle)

    origin_x, origin_y = roi.shape[1] / 2, roi.shape[0] / 2

    tick_positions = []
    for ld in tick_lines:
        mx, my = ld['midpoint']
        dx, dy = mx - origin_x, my - origin_y
        projection = dx * math.cos(ruler_angle_rad) + dy * math.sin(ruler_angle_rad)
        tick_positions.append({'projection': projection, 'midpoint': ld['midpoint']})

    tick_positions.sort(key=lambda x: x['projection'])

    # Filter duplicates
    unique_ticks = [tick_positions[0]]
    for i in range(1, len(tick_positions)):
        if abs(tick_positions[i]['projection'] - unique_ticks[-1]['projection']) > 3:
            unique_ticks.append(tick_positions[i])

    if len(unique_ticks) < 3: return None

    # Determine Interval
    projections = [t['projection'] for t in unique_ticks]
    intervals = [projections[i + 1] - projections[i] for i in range(len(projections) - 1)]

    if not intervals: return None

    avg_interval_px = np.median(intervals)  # Median is safer against outliers

    # Filter for evenly spaced ticks
    evenly_spaced_ticks = [unique_ticks[0]]
    for i in range(1, len(unique_ticks)):
        expected = evenly_spaced_ticks[-1]['projection'] + avg_interval_px
        if abs(unique_ticks[i]['projection'] - expected) < avg_interval_px * 0.4:
            evenly_spaced_ticks.append(unique_ticks[i])

    if len(evenly_spaced_ticks) < 3: return None

    # Unit Detection Logic
    detected_unit_local = "mm"  # Default preference based on your model description
    confidence = 0

    # Simple heuristic: mm marks are usually close, cm marks far
    # This relies on camera being at a standard inspection distance
    if avg_interval_px < 15:
        detected_unit_local = "mm"  # High density
        mm_per_unit = 1.0
        confidence = 80
    elif 15 <= avg_interval_px < 80:
        detected_unit_local = "cm"  # Lower density (only cm marks visible)
        mm_per_unit = 10.0
        confidence = 60
    else:
        detected_unit_local = "inch"
        mm_per_unit = 25.4
        confidence = 50

    pixels_per_mm_calc = avg_interval_px / mm_per_unit
    total_length_mm = (evenly_spaced_ticks[-1]['projection'] - evenly_spaced_ticks[0][
        'projection']) / pixels_per_mm_calc

    return {
        "pixels_per_mm": pixels_per_mm_calc,
        "avg_interval_px": avg_interval_px,
        "num_ticks": len(evenly_spaced_ticks),
        "tick_positions": evenly_spaced_ticks,
        "ruler_angle": ruler_angle,
        "tick_angle": avg_tick_angle,
        "roi_offset": (x1, y1),
        "total_length_mm": total_length_mm,
        "detected_unit": detected_unit_local,
        "unit_confidence": confidence,
        "mm_per_tick": mm_per_unit,
        "origin": (origin_x, origin_y)
    }


# ----------------------------------------------------------------------
# Mouse Callback (Buttons Only)
# ----------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global in_ruler_calib_mode, PIXELS_PER_MM, detected_unit
    global calibration_result, yolo_detection_box, calibration_frozen_frame

    if not in_ruler_calib_mode: return

    def in_rect(px, py, rect):
        rx, ry, rw, rh = rect
        return rx <= px <= rx + rw and ry <= py <= ry + rh

    if event == cv2.EVENT_LBUTTONDOWN:
        if in_rect(x, y, APPLY_BTN):
            if calibration_result:
                PIXELS_PER_MM = calibration_result['pixels_per_mm']
                detected_unit = calibration_result['detected_unit']
                update_ranges()
                print(f"✓ Applied: {PIXELS_PER_MM:.4f} px/mm")
                # Reset state
                in_ruler_calib_mode = False
                calibration_result = None
                yolo_detection_box = None
                calibration_frozen_frame = None
            return

        elif in_rect(x, y, CANCEL_BTN):
            in_ruler_calib_mode = False
            calibration_result = None
            yolo_detection_box = None
            calibration_frozen_frame = None
            return


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
        (cx, cy), (w, h), angle = rect
        box = np.intp(cv2.boxPoints(rect))

        # Determine width/length based on geometry
        dim1 = get_distance(box[0], box[1])
        dim2 = get_distance(box[1], box[2])
        width_px = min(dim1, dim2)
        height_px = max(dim1, dim2)

        diameter = width_px / PIXELS_PER_MM
        length = height_px / PIXELS_PER_MM

        if should_process_pellet(diameter, length):
            pellets.append({
                'box': box, 'center': (cx, cy),
                'diameter': diameter, 'length': length,
                'within_tolerance': is_within_tolerance(diameter, length)
            })
    return pellets


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------
def draw_ruler_calibration_mode(frame):
    overlay = frame.copy()

    # Draw YOLO Detection Box (if exists)
    if yolo_detection_box:
        bx1, by1, bx2, by2 = yolo_detection_box
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 100, 0), 2)
        cv2.putText(frame, "YOLO: MM ZONE", (bx1, by1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

    # Draw Detected Ticks inside the box
    if calibration_result:
        x_off, y_off = calibration_result['roi_offset']
        angle_rad = math.radians(calibration_result['tick_angle'])
        tick_len = 20

        for tick in calibration_result['tick_positions']:
            mx, my = tick['midpoint']
            ax, ay = int(mx + x_off), int(my + y_off)

            dx = int(tick_len * math.cos(angle_rad))
            dy = int(tick_len * math.sin(angle_rad))
            cv2.line(frame, (ax - dx, ay - dy), (ax + dx, ay + dy), (0, 255, 0), 2)
            cv2.circle(frame, (ax, ay), 2, (0, 0, 255), -1)

    # Draw UI Panel
    cv2.rectangle(overlay, (RULER_PANEL_X, RULER_PANEL_Y),
                  (RULER_PANEL_X + RULER_PANEL_W, RULER_PANEL_Y + RULER_PANEL_H),
                  (30, 30, 50), -1)
    cv2.rectangle(overlay, (RULER_PANEL_X, RULER_PANEL_Y),
                  (RULER_PANEL_X + RULER_PANEL_W, RULER_PANEL_Y + RULER_PANEL_H),
                  (100, 150, 255), 3)

    cv2.putText(overlay, "AI RULER CALIBRATION", (RULER_PANEL_X + 20, RULER_PANEL_Y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Status Text
    status_y = RULER_PANEL_Y + 70
    if not calibration_result and not yolo_detection_box:
        cv2.putText(overlay, "Detecting ruler...", (RULER_PANEL_X + 20, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    elif yolo_detection_box and not calibration_result:
        cv2.putText(overlay, "Zone found. Analysing ticks...", (RULER_PANEL_X + 20, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
    elif calibration_result:
        res = calibration_result
        lines = [
            f"Unit: {res['detected_unit'].upper()} ({res['mm_per_tick']}mm/tick)",
            f"Ticks: {res['num_ticks']} detected",
            f"Avg Spacing: {res['avg_interval_px']:.2f} px",
            f"Result: {res['pixels_per_mm']:.4f} px/mm"
        ]
        for i, line in enumerate(lines):
            cv2.putText(overlay, line, (RULER_PANEL_X + 20, status_y + (i * 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 1)

    # Buttons
    btn_color = (0, 200, 0) if calibration_result else (80, 80, 80)
    cv2.rectangle(overlay, (APPLY_BTN[0], APPLY_BTN[1]), (APPLY_BTN[0] + APPLY_BTN[2], APPLY_BTN[1] + APPLY_BTN[3]),
                  btn_color, -1)
    cv2.putText(overlay, "APPLY", (APPLY_BTN[0] + 35, APPLY_BTN[1] + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    cv2.rectangle(overlay, (CANCEL_BTN[0], CANCEL_BTN[1]),
                  (CANCEL_BTN[0] + CANCEL_BTN[2], CANCEL_BTN[1] + CANCEL_BTN[3]), (100, 100, 100), -1)
    cv2.putText(overlay, "CANCEL", (CANCEL_BTN[0] + 30, CANCEL_BTN[1] + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)


def draw_overlay(frame, pellets):
    # Top Status Bar
    total = len(pellets)
    within = sum(1 for p in pellets if p['within_tolerance'])
    status_color = (0, 255, 0) if total > 0 and (total - within) == 0 else (0, 0, 255)

    cv2.rectangle(frame, (10, 10), (500, 50), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (500, 50), status_color, 2)
    cv2.putText(frame, f"Good: {within} | Bad: {total - within} | Scale: {PIXELS_PER_MM:.2f}",
                (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw Pellets
    for p in pellets:
        color = (0, 255, 0) if p['within_tolerance'] else (0, 0, 255)
        cv2.drawContours(frame, [p['box']], 0, color, 1)

        # Label
        x, y = p['box'][1]  # Top-ish corner
        cv2.putText(frame, f"{p['diameter']:.2f}x{p['length']:.2f}", (int(x), int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Instructions
    if in_ruler_calib_mode:
        draw_ruler_calibration_mode(frame)
    else:
        cv2.putText(frame, "[R] Calibrate (AI) | [Q] Quit", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    return frame


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    global in_ruler_calib_mode, calibration_frozen_frame, yolo_detection_box, calibration_result

    print("Initializing...")
    load_yolo_model()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)

    cv2.namedWindow("Pellet Inspector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Pellet Inspector", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Logic Branching
        if in_ruler_calib_mode:
            # 1. Freeze frame if just entered mode
            if calibration_frozen_frame is None:
                calibration_frozen_frame = frame.copy()

                # 2. RUN AI ONCE
                print("Running YOLO detection...")
                box, res = run_yolo_calibration(calibration_frozen_frame)
                yolo_detection_box = box
                calibration_result = res

                if not box:
                    print("⚠ YOLO did not find a ruler/mm-zone.")

            # 3. Draw UI on frozen frame
            display_frame = calibration_frozen_frame.copy()
            draw_ruler_calibration_mode(display_frame)

        else:
            # Normal Pellet Mode
            pellets = detect_pellets(frame)
            display_frame = draw_overlay(frame.copy(), pellets)

        cv2.imshow("Pellet Inspector", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Toggle Mode
            in_ruler_calib_mode = not in_ruler_calib_mode
            # Reset detection state when entering/leaving
            calibration_frozen_frame = None
            yolo_detection_box = None
            calibration_result = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()