import cv2
import numpy as np
import time
import sys
import math
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Global Calibration
# ----------------------------------------------------------------------
PIXELS_PER_MM = 6.0
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0

# Auto-calibration state
AUTO_CALIBRATION_ENABLED = True
CALIBRATION_CONFIDENCE_THRESHOLD = 0.3  # Lower threshold for partial rulers
CALIBRATION_SAMPLES = []
MAX_CALIBRATION_SAMPLES = 20  # More samples for stability
STABLE_CALIBRATION_THRESHOLD = 0.05  # Tighter variance (5%)
MIN_SAMPLES_FOR_UPDATE = 15  # Need more samples before updating

# Ruler detection persistence
ruler_detection_history = []
RULER_HISTORY_SIZE = 10
MIN_DETECTIONS_FOR_LOCK = 6  # Must detect in 6/10 frames to stay locked

# Load YOLO model for ruler detection
try:
    ruler_model = YOLO('best.pt')
    print("✓ YOLO ruler detection model loaded successfully")
except Exception as e:
    print(f"⚠ Warning: Could not load YOLO model: {e}")
    print("  Auto-calibration will be disabled. Place 'best.pt' in the same directory.")
    AUTO_CALIBRATION_ENABLED = False
    ruler_model = None


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

# Manual calibration state (kept as fallback)
in_ruler_calib_mode = False
REFERENCE_LENGTH_MM = 76.2
reference_line_start = None
reference_line_end = None
calibration_frozen_frame = None
is_dragging = False

RULER_PANEL_X, RULER_PANEL_Y = 10, 80
RULER_PANEL_W, RULER_PANEL_H = 380, 280
RESET_BTN = (RULER_PANEL_X + 20, RULER_PANEL_Y + 200, 100, 40)
APPLY_BTN = (RULER_PANEL_X + 140, RULER_PANEL_Y + 200, 100, 40)
CANCEL_BTN = (RULER_PANEL_X + 260, RULER_PANEL_Y + 200, 100, 40)


# ----------------------------------------------------------------------
# Auto-Calibration Functions
# ----------------------------------------------------------------------
def detect_ruler_region(frame):
    """Use YOLO to detect ruler in frame with persistence"""
    global ruler_detection_history

    if not AUTO_CALIBRATION_ENABLED or ruler_model is None:
        return None

    try:
        results = ruler_model(frame, verbose=False, conf=0.25)  # Lower conf for partial rulers

        current_detection = None

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()

            if len(confidences) > 0:
                best_idx = np.argmax(confidences)
                if confidences[best_idx] >= CALIBRATION_CONFIDENCE_THRESHOLD:
                    box = boxes.xyxy[best_idx].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box

                    # Expand ROI slightly to catch edge ticks
                    margin = 20
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(frame.shape[1], x2 + margin)
                    y2 = min(frame.shape[0], y2 + margin)

                    current_detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(confidences[best_idx]),
                        'roi': frame[y1:y2, x1:x2].copy()
                    }

        # Add to history
        ruler_detection_history.append(1 if current_detection else 0)
        if len(ruler_detection_history) > RULER_HISTORY_SIZE:
            ruler_detection_history.pop(0)

        # Only return detection if we have consistent detections
        if len(ruler_detection_history) >= MIN_DETECTIONS_FOR_LOCK:
            detection_count = sum(ruler_detection_history)
            if detection_count >= MIN_DETECTIONS_FOR_LOCK:
                return current_detection

        return current_detection

    except Exception as e:
        print(f"Error in ruler detection: {e}")

    return None


def detect_tick_marks(roi):
    """Detect tick marks on ruler using multiple enhancement techniques"""
    if roi is None or roi.size == 0:
        return []

    # Handle small ROIs
    if roi.shape[0] < 20 or roi.shape[1] < 20:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Multi-stage enhancement
    # 1. Enhance contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Denoise while preserving edges
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    # 3. Sharpen to make ticks more prominent
    kernel_sharp = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel_sharp)

    # 4. Edge detection with multiple thresholds
    edges1 = cv2.Canny(sharpened, 30, 100, apertureSize=3)
    edges2 = cv2.Canny(sharpened, 50, 150, apertureSize=3)
    edges = cv2.bitwise_or(edges1, edges2)

    # 5. Morphological closing to connect broken tick lines
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_morph)

    # Detect lines using Hough Transform with relaxed parameters
    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi / 180,
                            threshold=15,  # Lower threshold
                            minLineLength=5,  # Shorter min length for partial rulers
                            maxLineGap=3)

    if lines is None:
        return []

    # Filter and classify tick marks
    tick_marks = []
    roi_height = roi.shape[0]
    roi_width = roi.shape[1]

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate angle from vertical
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:  # Perfectly vertical
            angle_from_vertical = 0
        else:
            angle = abs(math.atan2(dy, dx) * 180 / math.pi)
            angle_from_vertical = min(abs(angle - 90), abs(angle + 90))

        # Keep lines that are close to vertical (within 25 degrees)
        if angle_from_vertical <= 25:
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            # Filter out lines at extreme edges (might be noise)
            if 5 < mid_x < roi_width - 5:
                tick_marks.append({
                    'x': mid_x,
                    'y1': min(y1, y2),
                    'y2': max(y1, y2),
                    'length': length,
                    'line': (x1, y1, x2, y2),
                    'angle_from_vertical': angle_from_vertical
                })

    # Sort by x position
    tick_marks.sort(key=lambda t: t['x'])

    # Remove duplicate ticks (very close x positions)
    filtered_ticks = []
    for i, tick in enumerate(tick_marks):
        if i == 0 or tick['x'] - filtered_ticks[-1]['x'] > 3:
            filtered_ticks.append(tick)

    return filtered_ticks


def calculate_pixel_per_mm_from_ticks(tick_marks, roi_width):
    """Calculate pixels per mm from detected tick marks with robust algorithm"""
    if len(tick_marks) < 3:  # Need at least 3 ticks
        return None

    # Separate ticks by length
    lengths = [t['length'] for t in tick_marks]

    if not lengths:
        return None

    median_length = np.median(lengths)

    # Classify ticks: major (long), medium, minor (short)
    major_ticks = [t for t in tick_marks if t['length'] > median_length * 1.2]
    medium_ticks = [t for t in tick_marks if median_length * 0.7 <= t['length'] <= median_length * 1.2]
    minor_ticks = [t for t in tick_marks if t['length'] < median_length * 0.7]

    # Try multiple strategies
    strategies = []

    # Strategy 1: Major ticks = 1cm (10mm) apart
    if len(major_ticks) >= 2:
        distances = []
        for i in range(len(major_ticks) - 1):
            dist = major_ticks[i + 1]['x'] - major_ticks[i]['x']
            if roi_width * 0.05 < dist < roi_width * 0.4:  # Reasonable range
                distances.append(dist)

        if distances:
            median_major_dist = np.median(distances)
            strategies.append(('major_10mm', median_major_dist / 10.0))

    # Strategy 2: All ticks - find most common distance (likely 1mm or 5mm)
    all_distances = []
    for i in range(len(tick_marks) - 1):
        dist = tick_marks[i + 1]['x'] - tick_marks[i]['x']
        if 2 < dist < roi_width * 0.2:
            all_distances.append(dist)

    if len(all_distances) >= 3:
        # Cluster distances to find common spacing
        hist, bins = np.histogram(all_distances, bins=20)
        most_common_idx = np.argmax(hist)
        most_common_dist = (bins[most_common_idx] + bins[most_common_idx + 1]) / 2

        # Check if it's likely 1mm, 5mm, or 10mm spacing
        if most_common_dist > 0:
            # Try different assumptions
            for mm_value in [1.0, 5.0, 10.0]:
                px_per_mm = most_common_dist / mm_value
                if 2.0 <= px_per_mm <= 20.0:  # Sanity check
                    strategies.append((f'common_{mm_value}mm', px_per_mm))

    # Strategy 3: Medium ticks = 5mm apart (half cm)
    if len(medium_ticks) >= 2:
        distances = []
        for i in range(len(medium_ticks) - 1):
            dist = medium_ticks[i + 1]['x'] - medium_ticks[i]['x']
            if roi_width * 0.03 < dist < roi_width * 0.3:
                distances.append(dist)

        if distances:
            median_medium_dist = np.median(distances)
            strategies.append(('medium_5mm', median_medium_dist / 5.0))

    if not strategies:
        return None

    # Use median of all valid strategies
    valid_values = [val for _, val in strategies if 2.0 <= val <= 20.0]

    if valid_values:
        return np.median(valid_values)

    return None


def update_calibration_with_sample(new_px_per_mm):
    """Add calibration sample and update if stable - with outlier rejection"""
    global PIXELS_PER_MM, CALIBRATION_SAMPLES

    # Reject obvious outliers before adding
    if CALIBRATION_SAMPLES:
        current_median = np.median(CALIBRATION_SAMPLES)
        # Reject if more than 30% different from current median
        if abs(new_px_per_mm - current_median) / current_median > 0.3:
            return False

    CALIBRATION_SAMPLES.append(new_px_per_mm)

    # Keep only recent samples
    if len(CALIBRATION_SAMPLES) > MAX_CALIBRATION_SAMPLES:
        CALIBRATION_SAMPLES.pop(0)

    # Need sufficient samples for stability check
    if len(CALIBRATION_SAMPLES) >= MIN_SAMPLES_FOR_UPDATE:
        # Use median instead of mean (more robust to outliers)
        median_val = np.median(CALIBRATION_SAMPLES)
        std_val = np.std(CALIBRATION_SAMPLES)

        # Check if calibration is stable (low variance)
        if std_val / median_val < STABLE_CALIBRATION_THRESHOLD:
            # Update calibration
            old_val = PIXELS_PER_MM
            PIXELS_PER_MM = median_val
            update_ranges()

            if abs(old_val - median_val) > 0.1:
                print(f"✓ Auto-calibrated: {PIXELS_PER_MM:.4f} px/mm (σ={std_val:.4f}, n={len(CALIBRATION_SAMPLES)})")

            return True

    return False


def auto_calibrate_from_frame(frame):
    """Main auto-calibration function"""
    ruler_data = detect_ruler_region(frame)

    if ruler_data is None:
        return None, None

    roi = ruler_data['roi']
    tick_marks = detect_tick_marks(roi)

    if len(tick_marks) < 2:
        return ruler_data, None

    px_per_mm = calculate_pixel_per_mm_from_ticks(tick_marks, roi.shape[1])

    if px_per_mm is not None:
        update_calibration_with_sample(px_per_mm)

    return ruler_data, tick_marks


def draw_ruler_detection(frame, ruler_data, tick_marks):
    """Draw detected ruler and tick marks on frame with enhanced visualization"""
    if ruler_data is None:
        return

    x1, y1, x2, y2 = ruler_data['bbox']
    conf = ruler_data['confidence']

    # Draw semi-transparent overlay on ruler region
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Draw confidence label
    label = f"Ruler {conf:.2f}"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 10, y1), (0, 255, 255), -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Draw detected tick marks with classification
    if tick_marks:
        lengths = [t['length'] for t in tick_marks]
        if lengths:
            median_length = np.median(lengths)

            for tick in tick_marks:
                tx1, ty1, tx2, ty2 = tick['line']

                # Color code by tick type
                if tick['length'] > median_length * 1.2:
                    color = (255, 0, 255)  # Magenta - major ticks
                    thickness = 2
                elif tick['length'] > median_length * 0.7:
                    color = (255, 100, 255)  # Light magenta - medium ticks
                    thickness = 2
                else:
                    color = (200, 150, 255)  # Pale magenta - minor ticks
                    thickness = 1

                cv2.line(frame, (x1 + tx1, y1 + ty1), (x1 + tx2, y1 + ty2),
                         color, thickness)

        # Draw tick count info
        tick_info = f"Ticks: {len(tick_marks)}"
        cv2.putText(frame, tick_info, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


# ----------------------------------------------------------------------
# Helper Checks
# ----------------------------------------------------------------------
def is_within_tolerance(diameter: float, length: float) -> bool:
    return (DIAMETER_MIN <= diameter <= DIAMETER_MAX and
            LENGTH_MIN <= length <= LENGTH_MAX)


def should_process_pellet(diameter: float, length: float) -> bool:
    return (DIAMETER_EXCLUDE_MIN <= diameter <= DIAMETER_EXCLUDE_MAX and
            LENGTH_EXCLUDE_MIN <= length <= LENGTH_EXCLUDE_MAX)


def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ----------------------------------------------------------------------
# Mouse Callback for Manual Calibration (Fallback)
# ----------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global reference_line_start, reference_line_end, is_dragging
    global in_ruler_calib_mode, PIXELS_PER_MM

    if not in_ruler_calib_mode:
        return

    def in_rect(px, py, rect):
        rx, ry, rw, rh = rect
        return rx <= px <= rx + rw and ry <= py <= ry + rh

    if event == cv2.EVENT_LBUTTONDOWN:
        if in_rect(x, y, RESET_BTN):
            reference_line_start = None
            reference_line_end = None
            is_dragging = False
            return
        elif in_rect(x, y, APPLY_BTN):
            if reference_line_start and reference_line_end:
                dx = reference_line_end[0] - reference_line_start[0]
                dy = reference_line_end[1] - reference_line_start[1]
                pixel_distance = math.sqrt(dx ** 2 + dy ** 2)

                if pixel_distance > 10:
                    PIXELS_PER_MM = pixel_distance / REFERENCE_LENGTH_MM
                    update_ranges()
                    print(f"Manual calibration: {PIXELS_PER_MM:.4f} px/mm")

                in_ruler_calib_mode = False
                reference_line_start = None
                reference_line_end = None
                is_dragging = False
            return
        elif in_rect(x, y, CANCEL_BTN):
            in_ruler_calib_mode = False
            reference_line_start = None
            reference_line_end = None
            is_dragging = False
            return

        if not in_rect(x, y, (RULER_PANEL_X, RULER_PANEL_Y, RULER_PANEL_W, RULER_PANEL_H)):
            reference_line_start = (x, y)
            reference_line_end = (x, y)
            is_dragging = True

    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        reference_line_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if is_dragging:
            reference_line_end = (x, y)
            is_dragging = False


# ----------------------------------------------------------------------
# Detection with Rotated Bounding Boxes
# ----------------------------------------------------------------------
def detect_pellets(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

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

        edge1 = get_distance(box[0], box[1])
        edge2 = get_distance(box[1], box[2])

        if edge1 < edge2:
            width_px = edge1
            height_px = edge2
        else:
            width_px = edge2
            height_px = edge1

        width_mm = width_px / PIXELS_PER_MM
        height_mm = height_px / PIXELS_PER_MM

        diameter = width_mm
        length = height_mm

        if should_process_pellet(diameter, length):
            pellets.append({
                'contour': cnt,
                'box': box,
                'center': (center_x, center_y),
                'angle': angle,
                'width_px': width_px,
                'height_px': height_px,
                'diameter': diameter,
                'length': length,
                'within_tolerance': is_within_tolerance(diameter, length)
            })
    return pellets


# ----------------------------------------------------------------------
# Draw Manual Calibration Mode
# ----------------------------------------------------------------------
def draw_ruler_calibration_mode(frame):
    overlay = frame.copy()

    cv2.rectangle(overlay, (RULER_PANEL_X, RULER_PANEL_Y),
                  (RULER_PANEL_X + RULER_PANEL_W, RULER_PANEL_Y + RULER_PANEL_H),
                  (30, 30, 50), -1)
    cv2.rectangle(overlay, (RULER_PANEL_X, RULER_PANEL_Y),
                  (RULER_PANEL_X + RULER_PANEL_W, RULER_PANEL_Y + RULER_PANEL_H),
                  (100, 150, 255), 3)

    cv2.putText(overlay, "MANUAL CALIBRATION", (RULER_PANEL_X + 70, RULER_PANEL_Y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    instructions = [
        "1. Place a ruler in camera view",
        "2. Click and drag to match the",
        "   reference line (3 inch / 7.62 cm)",
        "3. Click APPLY when aligned"
    ]

    y_offset = RULER_PANEL_Y + 60
    for instr in instructions:
        cv2.putText(overlay, instr, (RULER_PANEL_X + 20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_offset += 20

    cv2.putText(overlay, "Reference: 3 inch (76.2 mm)",
                (RULER_PANEL_X + 60, RULER_PANEL_Y + 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)

    # Buttons
    cv2.rectangle(overlay, (RESET_BTN[0], RESET_BTN[1]),
                  (RESET_BTN[0] + RESET_BTN[2], RESET_BTN[1] + RESET_BTN[3]),
                  (50, 50, 200), -1)
    cv2.putText(overlay, "RESET", (RESET_BTN[0] + 15, RESET_BTN[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    apply_enabled = reference_line_start and reference_line_end
    apply_color = (0, 200, 0) if apply_enabled else (100, 100, 100)
    cv2.rectangle(overlay, (APPLY_BTN[0], APPLY_BTN[1]),
                  (APPLY_BTN[0] + APPLY_BTN[2], APPLY_BTN[1] + APPLY_BTN[3]),
                  apply_color, -1)
    cv2.putText(overlay, "APPLY", (APPLY_BTN[0] + 15, APPLY_BTN[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.rectangle(overlay, (CANCEL_BTN[0], CANCEL_BTN[1]),
                  (CANCEL_BTN[0] + CANCEL_BTN[2], CANCEL_BTN[1] + CANCEL_BTN[3]),
                  (100, 100, 100), -1)
    cv2.putText(overlay, "CANCEL", (CANCEL_BTN[0] + 10, CANCEL_BTN[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(overlay, f"Current: {PIXELS_PER_MM:.2f} px/mm",
                (RULER_PANEL_X + 20, RULER_PANEL_Y + 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2)

    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    # Draw reference line
    if reference_line_start and reference_line_end:
        cv2.line(frame, reference_line_start, reference_line_end, (0, 255, 255), 2)
        cv2.circle(frame, reference_line_start, 5, (0, 0, 255), -1)
        cv2.circle(frame, reference_line_end, 5, (0, 0, 255), -1)


# ----------------------------------------------------------------------
# Main Overlay
# ----------------------------------------------------------------------
def draw_overlay(frame, pellets, ruler_data=None, tick_marks=None):
    total = len(pellets)
    within = sum(1 for p in pellets if p['within_tolerance'])
    out_of = total - within
    status_text = f"In: {within}   Out: {out_of}   Total: {total}"
    status_color = (0, 255, 0) if out_of == 0 else (0, 0, 255)

    cv2.rectangle(frame, (10, 10), (460, 50), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (460, 50), status_color, 2)
    cv2.putText(frame, status_text, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

    # Draw calibration status with more detail
    calib_status = f"Cal: {PIXELS_PER_MM:.3f} px/mm"
    sample_info = f"[{len(CALIBRATION_SAMPLES)}/{MAX_CALIBRATION_SAMPLES}]"

    if AUTO_CALIBRATION_ENABLED and ruler_data is not None:
        if len(CALIBRATION_SAMPLES) >= MIN_SAMPLES_FOR_UPDATE:
            std_val = np.std(CALIBRATION_SAMPLES) if len(CALIBRATION_SAMPLES) > 1 else 0
            stability = 100 * (1 - std_val / np.median(CALIBRATION_SAMPLES)) if CALIBRATION_SAMPLES else 0
            calib_status += f" AUTO {sample_info} {stability:.0f}%"
            calib_color = (0, 255, 0) if stability > 95 else (0, 255, 255)
        else:
            calib_status += f" CALIBRATING {sample_info}"
            calib_color = (0, 165, 255)  # Orange
    else:
        calib_status += " [MANUAL]"
        calib_color = (200, 200, 200)

    cv2.putText(frame, calib_status, (470, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, calib_color, 2)

    # Draw ruler detection
    if AUTO_CALIBRATION_ENABLED and ruler_data is not None:
        draw_ruler_detection(frame, ruler_data, tick_marks)

    for p in pellets:
        box = p['box']
        center = p['center']
        color = (0, 255, 0) if p['within_tolerance'] else (0, 0, 255)

        cv2.drawContours(frame, [box], 0, color, 2)
        cv2.circle(frame, (int(center[0]), int(center[1])), 3, color, -1)

        top_y = int(min(box[:, 1]))
        left_x = int(min(box[:, 0]))
        bg_y = max(top_y - 30, 0)

        cv2.rectangle(frame, (left_x, bg_y), (left_x + 75, top_y - 5), (0, 0, 0), -1)
        cv2.putText(frame, f"D: {p['diameter']:.2f}mm", (left_x + 3, bg_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(frame, f"L: {p['length']:.2f}mm", (left_x + 3, bg_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        if not p['within_tolerance']:
            top_right = box[np.argmax(box[:, 0])]
            cv2.circle(frame, tuple(top_right), 8, (0, 0, 255), -1)
            cv2.putText(frame, "!", (top_right[0] - 4, top_right[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if not in_ruler_calib_mode:
        help_text = "Press 'r' for manual calibration | 'q' to quit"
        if AUTO_CALIBRATION_ENABLED:
            help_text = "Auto-calibration ON | Press 'r' for manual | 'q' to quit"
        cv2.putText(frame, help_text,
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1)

    if in_ruler_calib_mode:
        draw_ruler_calibration_mode(frame)

    return frame


# ----------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------
def get_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"------------------------------------------------")
    print(f"Camera Resolution: {int(actual_w)}x{int(actual_h)}")
    print(f"Auto-calibration: {'ENABLED' if AUTO_CALIBRATION_ENABLED else 'DISABLED'}")
    print(f"------------------------------------------------")

    return cap


# ----------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------
def main():
    global in_ruler_calib_mode, calibration_frozen_frame

    print("\n" + "=" * 60)
    print("  PELLET INSPECTOR WITH AUTO-CALIBRATION")
    print("=" * 60)
    print("Features:")
    print("  • Automatic ruler detection & calibration (YOLOv11)")
    print("  • Real-time tick mark analysis")
    print("  • Manual calibration fallback (Press 'r')")
    print("  • Press 'q' to quit")
    print("=" * 60 + "\n")

    cap = get_camera()
    if not cap.isOpened():
        print("Cannot open camera.")
        sys.exit(1)

    fps_counter = 0
    fps_start = time.time()
    fps_display = 0
    frame_count = 0

    window_name = "Pellet Inspector - Auto Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera lost – reconnecting...")
            cap.release()
            time.sleep(1)
            cap = get_camera()
            if not cap.isOpened():
                break
            continue

        if in_ruler_calib_mode and calibration_frozen_frame is None:
            calibration_frozen_frame = frame.copy()
        elif not in_ruler_calib_mode:
            calibration_frozen_frame = None

        display_frame = calibration_frozen_frame.copy() if in_ruler_calib_mode else frame.copy()

        ruler_data = None
        tick_marks = None

        if not in_ruler_calib_mode and AUTO_CALIBRATION_ENABLED:
            # Run auto-calibration every 3 frames to balance accuracy and performance
            if frame_count % 3 == 0:
                ruler_data, tick_marks = auto_calibrate_from_frame(display_frame)

        if not in_ruler_calib_mode:
            pellets = detect_pellets(display_frame)
            display_frame = draw_overlay(display_frame, pellets, ruler_data, tick_marks)
        else:
            display_frame = draw_overlay(display_frame, [], ruler_data, tick_marks)

        fps_counter += 1
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_counter // int(elapsed)
            fps_counter = 0
            fps_start = time.time()

        cv2.putText(display_frame, f"FPS: {fps_display}",
                    (display_frame.shape[1] - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            if in_ruler_calib_mode:
                in_ruler_calib_mode = False
                reference_line_start = None
                reference_line_end = None
                is_dragging = False
                calibration_frozen_frame = None
            else:
                in_ruler_calib_mode = True

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Shutdown complete.")


if __name__ == "__main__":
    main()