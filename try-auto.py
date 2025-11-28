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

# YOLO Model
yolo_model = None
ruler_detected = False
ruler_bbox = None
last_calibration_time = 0
CALIBRATION_INTERVAL = 2.0  # Recalibrate every 2 seconds


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

MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 20000

DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720

detected_unit = None
auto_calibration_active = True


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


# ----------------------------------------------------------------------
# YOLO Ruler Detection
# ----------------------------------------------------------------------
def detect_ruler_yolo(frame):
    """Detect ruler using YOLO model"""
    global yolo_model

    if yolo_model is None:
        return None

    try:
        results = yolo_model(frame, conf=0.5, verbose=False)

        if len(results) > 0 and len(results[0].boxes) > 0:
            # Get the first detected ruler (highest confidence)
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()

                # Return bounding box
                return {
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(conf)
                }
    except Exception as e:
        print(f"Error in YOLO detection: {e}")

    return None


# ----------------------------------------------------------------------
# Ruler Tick Detection and Auto-Calibration
# ----------------------------------------------------------------------
def analyze_ruler_region(frame, bbox):
    """
    Analyze the ruler region and AUTO-DETECT tick marks and unit
    Focus on INCH marks (largest/most prominent) for easier detection
    """
    x1, y1, x2, y2 = bbox

    # Add padding to ensure we capture the ruler edges
    padding = 10
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame.shape[1], x2 + padding)
    y2 = min(frame.shape[0], y2 + padding)

    if x2 - x1 < 50 or y2 - y1 < 50:
        return None

    roi = frame[y1:y2, x1:x2].copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Enhance contrast for better tick detection
    gray = cv2.equalizeHist(gray)

    # Apply edge detection with optimized parameters for ruler ticks
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough Transform
    # Adjusted parameters to focus on LONGER lines (inch marks are longest)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=20, maxLineGap=5)

    if lines is None or len(lines) < 2:
        return None

    # Group lines by angle and length to find dominant tick direction
    line_data = []
    for line in lines:
        x1_l, y1_l, x2_l, y2_l = line[0]
        length = math.sqrt((x2_l - x1_l) ** 2 + (y2_l - y1_l) ** 2)
        angle = math.atan2(y2_l - y1_l, x2_l - x1_l) * 180 / np.pi

        # Normalize angle to 0-180 range
        if angle < 0:
            angle += 180

        line_data.append({
            'line': line[0],
            'length': length,
            'angle': angle,
            'midpoint': ((x1_l + x2_l) / 2, (y1_l + y2_l) / 2)
        })

    # Filter for LONGEST lines (these are likely inch marks)
    # Sort by length and take top 50%
    line_data.sort(key=lambda x: x['length'], reverse=True)
    longest_lines = line_data[:max(len(line_data) // 2, 3)]

    # Group the longest lines by angle
    from collections import defaultdict
    angle_groups = defaultdict(list)

    for ld in longest_lines:
        # Group angles in 5-degree bins
        angle_bin = round(ld['angle'] / 5) * 5
        angle_groups[angle_bin].append(ld)

    if not angle_groups:
        return None

    # Get the angle bin with most lines (major tick marks)
    dominant_angle_bin = max(angle_groups.keys(), key=lambda k: len(angle_groups[k]))
    major_tick_lines = angle_groups[dominant_angle_bin]

    if len(major_tick_lines) < 2:
        return None

    # Calculate the ruler's main axis (perpendicular to ticks)
    avg_tick_angle = np.mean([ld['angle'] for ld in major_tick_lines])
    ruler_angle = (avg_tick_angle + 90) % 180

    # Project all major tick midpoints onto the ruler's main axis
    ruler_angle_rad = math.radians(ruler_angle)
    origin_x = roi.shape[1] / 2
    origin_y = roi.shape[0] / 2

    tick_positions = []
    for ld in major_tick_lines:
        mx, my = ld['midpoint']
        dx = mx - origin_x
        dy = my - origin_y
        projection = dx * math.cos(ruler_angle_rad) + dy * math.sin(ruler_angle_rad)
        tick_positions.append({
            'projection': projection,
            'midpoint': ld['midpoint'],
            'length': ld['length']
        })

    # Sort by position along ruler
    tick_positions.sort(key=lambda x: x['projection'])

    # Remove duplicates (ticks detected multiple times)
    unique_ticks = []
    for i, tick in enumerate(tick_positions):
        if i == 0:
            unique_ticks.append(tick)
        else:
            if abs(tick['projection'] - unique_ticks[-1]['projection']) > 10:
                unique_ticks.append(tick)

    if len(unique_ticks) < 2:
        return None

    # Calculate intervals between consecutive major ticks
    projections = [t['projection'] for t in unique_ticks]
    intervals = []
    for i in range(len(projections) - 1):
        interval = projections[i + 1] - projections[i]
        intervals.append(interval)

    if not intervals:
        return None

    # Find the most consistent interval
    from collections import Counter
    intervals_rounded = [round(x) for x in intervals]
    interval_counts = Counter(intervals_rounded)

    if not interval_counts:
        return None

    most_common_interval = interval_counts.most_common(1)[0][0]

    # Filter intervals close to the most common
    filtered_intervals = [x for x in intervals if abs(x - most_common_interval) < most_common_interval * 0.25]

    if not filtered_intervals:
        return None

    # Average interval in pixels (this is likely 1 inch or 1 cm)
    avg_interval_px = np.mean(filtered_intervals)

    # Filter evenly spaced ticks
    evenly_spaced_ticks = [unique_ticks[0]]
    for i in range(1, len(unique_ticks)):
        expected_proj = evenly_spaced_ticks[-1]['projection'] + avg_interval_px
        if abs(unique_ticks[i]['projection'] - expected_proj) < avg_interval_px * 0.35:
            evenly_spaced_ticks.append(unique_ticks[i])

    if len(evenly_spaced_ticks) < 2:
        return None

    # SMART UNIT DETECTION
    # Since we focused on the LONGEST lines, these are most likely INCH marks
    # But we still validate by checking reasonable pixel ranges

    total_length_px = evenly_spaced_ticks[-1]['projection'] - evenly_spaced_ticks[0]['projection']
    num_intervals = len(evenly_spaced_ticks) - 1

    # Test each possible unit
    test_results = []

    for test_unit, mm_per_unit in [("inch", 25.4), ("cm", 10.0), ("mm", 1.0)]:
        test_px_per_mm = avg_interval_px / mm_per_unit
        test_total_mm = total_length_px / test_px_per_mm

        confidence = 100

        # Pixel density check (2-10 px/mm is reasonable for webcam)
        if test_px_per_mm < 2 or test_px_per_mm > 10:
            confidence -= 50
        elif test_px_per_mm < 3 or test_px_per_mm > 8:
            confidence -= 25

        # Total length check
        if test_total_mm < 30 or test_total_mm > 400:
            confidence -= 40
        elif test_total_mm < 50 or test_total_mm > 300:
            confidence -= 20

        # Number of intervals check
        if test_unit == "inch":
            # Expect 2-12 inch marks (we filtered for longest lines, so this should be inches)
            if 2 <= num_intervals <= 12:
                confidence += 30  # BONUS: This is likely inches since we filtered longest lines
            else:
                confidence -= 20
        elif test_unit == "cm":
            if 5 <= num_intervals <= 30:
                confidence += 10
            else:
                confidence -= 25
        elif test_unit == "mm":
            # Individual mm marks would require many ticks
            if num_intervals < 20:
                confidence -= 40

        # Line length bonus (longer lines suggest larger units like inches)
        avg_line_length = np.mean([t['length'] for t in evenly_spaced_ticks])
        if test_unit == "inch" and avg_line_length > 30:
            confidence += 20
        elif test_unit == "cm" and 20 < avg_line_length < 35:
            confidence += 10

        test_results.append({
            'unit': test_unit,
            'pixels_per_mm': test_px_per_mm,
            'total_mm': test_total_mm,
            'confidence': max(0, confidence)
        })

    # Select best result
    best_result = max(test_results, key=lambda x: x['confidence'])

    if best_result['confidence'] < 40:
        print("⚠ Low confidence in unit detection")
        # Default to inch (most likely with longest line filtering)
        detected_unit = "inch"
        pixels_per_mm = avg_interval_px / 25.4
        confidence_score = 50
    else:
        detected_unit = best_result['unit']
        pixels_per_mm = best_result['pixels_per_mm']
        confidence_score = best_result['confidence']

    unit_to_mm = {"inch": 25.4, "cm": 10.0, "mm": 1.0}
    mm_per_tick = unit_to_mm[detected_unit]
    total_length_mm = total_length_px / pixels_per_mm

    return {
        "pixels_per_mm": pixels_per_mm,
        "avg_interval_px": avg_interval_px,
        "num_ticks": len(evenly_spaced_ticks),
        "num_intervals": num_intervals,
        "tick_positions": evenly_spaced_ticks,
        "ruler_angle": ruler_angle,
        "tick_angle": avg_tick_angle,
        "roi_offset": (x1, y1),
        "total_length_mm": total_length_mm,
        "total_length_px": total_length_px,
        "origin": (origin_x, origin_y),
        "detected_unit": detected_unit,
        "unit_confidence": confidence_score,
        "mm_per_tick": mm_per_tick,
        "bbox": bbox
    }


# ----------------------------------------------------------------------
# Detection
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
# Draw Overlay
# ----------------------------------------------------------------------
def draw_overlay(frame, pellets, ruler_info=None):
    total = len(pellets)
    within = sum(1 for p in pellets if p['within_tolerance'])
    out_of = total - within
    status_text = f"In: {within}   Out: {out_of}   Total: {total}"
    status_color = (0, 255, 0) if out_of == 0 else (0, 0, 255)

    cv2.rectangle(frame, (10, 10), (460, 50), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (460, 50), status_color, 2)
    cv2.putText(frame, status_text, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

    # Draw pellets
    for p in pellets:
        box = p['box']
        center = p['center']
        color = (0, 255, 0) if p['within_tolerance'] else (0, 0, 255)

        cv2.drawContours(frame, [box], 0, color, 1)
        cv2.circle(frame, (int(center[0]), int(center[1])), 2, color, -1)

        top_y = int(min(box[:, 1]))
        left_x = int(min(box[:, 0]))
        bg_y = max(top_y - 30, 0)

        cv2.rectangle(frame, (left_x, bg_y), (left_x + 70, top_y - 5), (0, 0, 0), -1)
        cv2.putText(frame, f"D: {p['diameter']:.2f}mm", (left_x + 3, bg_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(frame, f"L: {p['length']:.2f}mm", (left_x + 3, bg_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        if not p['within_tolerance']:
            top_right = box[np.argmax(box[:, 0])]
            cv2.circle(frame, tuple(top_right), 8, (0, 0, 255), -1)
            cv2.putText(frame, "!", (top_right[0] - 4, top_right[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw ruler detection info
    if ruler_info:
        bbox = ruler_info['bbox']
        conf = ruler_info.get('confidence', 0)

        # Draw bounding box around detected ruler
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)

        # Draw calibration info panel
        if 'detected_unit' in ruler_info:
            panel_x, panel_y = 10, 60
            panel_w, panel_h = 380, 110

            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                          (30, 30, 50), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                          (255, 255, 0), 2)

            cv2.putText(frame, "AUTO-CALIBRATION ACTIVE", (panel_x + 10, panel_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            unit_color = (0, 255, 0) if ruler_info['unit_confidence'] >= 70 else (0, 255, 255)
            cv2.putText(frame, f"Unit: {ruler_info['detected_unit'].upper()} ({ruler_info['unit_confidence']}%)",
                        (panel_x + 10, panel_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, unit_color, 1)

            cv2.putText(frame, f"Calibration: {ruler_info['pixels_per_mm']:.3f} px/mm",
                        (panel_x + 10, panel_y + 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)

            cv2.putText(frame, f"Ticks: {ruler_info['num_ticks']} | Intervals: {ruler_info['num_intervals']}",
                        (panel_x + 10, panel_y + 94),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # Draw detected ticks on ruler
            x_offset, y_offset = ruler_info['roi_offset']
            tick_angle_rad = math.radians(ruler_info['tick_angle'])
            tick_length = 20

            for tick in ruler_info['tick_positions']:
                mx, my = tick['midpoint']
                abs_x = int(mx + x_offset)
                abs_y = int(my + y_offset)

                tick_dx = int(tick_length * math.cos(tick_angle_rad))
                tick_dy = int(tick_length * math.sin(tick_angle_rad))

                pt1 = (abs_x - tick_dx, abs_y - tick_dy)
                pt2 = (abs_x + tick_dx, abs_y + tick_dy)
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(frame, (abs_x, abs_y), 3, (0, 255, 255), -1)
        else:
            # Just show ruler detected
            cv2.putText(frame, f"RULER DETECTED ({conf:.0%})", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        # Show searching message
        cv2.putText(frame, "Searching for ruler...", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.putText(frame, "Press 'q' to quit | Auto-calibration: ON",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)

    return frame


# ----------------------------------------------------------------------
# Camera Setup
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
    print(f"Camera Resolution Requested: {DESIRED_WIDTH}x{DESIRED_HEIGHT}")
    print(f"Camera Resolution Actual:    {int(actual_w)}x{int(actual_h)}")
    print(f"------------------------------------------------")

    return cap


# ----------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------
def main():
    global PIXELS_PER_MM, detected_unit, last_calibration_time

    print("\n" + "=" * 60)
    print("Pellet Inspector with YOLO Auto-Calibration")
    print("=" * 60)
    print("Features:")
    print("  • Automatic ruler detection using YOLOv11")
    print("  • Auto-calibration from detected ruler ticks")
    print("  • Intelligent unit detection (inch/cm/mm)")
    print("  • Real-time pellet measurement")
    print("=" * 60)
    print("Press 'q' to quit")
    print("=" * 60 + "\n")

    # Load YOLO model
    if not load_yolo_model():
        print("Cannot proceed without YOLO model. Please check 'yolo/best.pt'")
        sys.exit(1)

    cap = get_camera()
    if not cap.isOpened():
        print("Cannot open camera.")
        sys.exit(1)

    fps_counter = 0
    fps_start = time.time()
    fps_display = 0

    window_name = "Pellet Inspector - YOLO Auto-Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    ruler_info = None

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

        current_time = time.time()

        # Detect ruler with YOLO
        ruler_detection = detect_ruler_yolo(frame)

        # Auto-calibrate if ruler detected and enough time has passed
        if ruler_detection and (current_time - last_calibration_time) > CALIBRATION_INTERVAL:
            calibration_result = analyze_ruler_region(frame, ruler_detection['bbox'])

            if calibration_result:
                PIXELS_PER_MM = calibration_result['pixels_per_mm']
                detected_unit = calibration_result['detected_unit']
                update_ranges()
                last_calibration_time = current_time

                ruler_info = calibration_result.copy()
                ruler_info['confidence'] = ruler_detection['confidence']

                print(f"\n{'=' * 60}")
                print(f"AUTO-CALIBRATION SUCCESSFUL")
                print(f"{'=' * 60}")
                print(
                    f"✓ Unit detected: {detected_unit.upper()} (confidence: {calibration_result['unit_confidence']}%)")
                print(f"✓ Calibration: {PIXELS_PER_MM:.4f} px/mm")
                print(
                    f"✓ Ticks detected: {calibration_result['num_ticks']} ({calibration_result['num_intervals']} intervals)")
                print(f"✓ Ruler length: {calibration_result['total_length_mm']:.1f}mm")
                print(f"{'=' * 60}\n")
            elif ruler_info is None:
                # Show basic ruler detection without calibration
                ruler_info = {'bbox': ruler_detection['bbox'], 'confidence': ruler_detection['confidence']}
        elif not ruler_detection:
            # Ruler lost - keep using last calibration but clear display
            if ruler_info and 'detected_unit' not in ruler_info:
                ruler_info = None

        # Detect and measure pellets
        pellets = detect_pellets(frame)

        # Draw overlay
        frame = draw_overlay(frame, pellets, ruler_info)

        # FPS counter
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_counter // int(elapsed)
            fps_counter = 0
            fps_start = time.time()

        cv2.putText(frame, f"FPS: {fps_display}",
                    (frame.shape[1] - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nShutdown complete.")


if __name__ == "__main__":
    main()