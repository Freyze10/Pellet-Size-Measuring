import cv2
import numpy as np
import time
import sys
import math

# ----------------------------------------------------------------------
# Global Calibration
# ----------------------------------------------------------------------
PIXELS_PER_MM = 6.0
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0


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
# Ruler Calibration State (Box-Based)
# ----------------------------------------------------------------------
in_ruler_calib_mode = False
calibration_frozen_frame = None
ruler_box_start = None
ruler_box_end = None
is_drawing_box = False
detected_ticks = []
calibration_result = None

# Button rectangles
RULER_PANEL_X, RULER_PANEL_Y = 10, 80
RULER_PANEL_W, RULER_PANEL_H = 400, 350

UNIT_CM_BTN = (RULER_PANEL_X + 20, RULER_PANEL_Y + 240, 80, 35)
UNIT_MM_BTN = (RULER_PANEL_X + 110, RULER_PANEL_Y + 240, 80, 35)
UNIT_INCH_BTN = (RULER_PANEL_X + 200, RULER_PANEL_Y + 240, 80, 35)

ANALYZE_BTN = (RULER_PANEL_X + 20, RULER_PANEL_Y + 285, 120, 40)
APPLY_BTN = (RULER_PANEL_X + 150, RULER_PANEL_Y + 285, 100, 40)
CANCEL_BTN = (RULER_PANEL_X + 260, RULER_PANEL_Y + 285, 100, 40)

selected_unit = "cm"  # cm, mm, or inch


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
# Ruler Tick Detection and Analysis
# ----------------------------------------------------------------------
def analyze_ruler_box(frame, box_start, box_end, unit):
    """Analyze the selected ruler box and detect tick marks"""
    x1, y1 = min(box_start[0], box_end[0]), min(box_start[1], box_end[1])
    x2, y2 = max(box_start[0], box_end[0]), max(box_start[1], box_end[1])

    # Ensure box is within frame bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    if x2 - x1 < 50 or y2 - y1 < 50:
        return None

    roi = frame[y1:y2, x1:x2].copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    gray = cv2.equalizeHist(gray)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=20, maxLineGap=5)

    if lines is None:
        return None

    # Separate horizontal and vertical lines
    h_lines = []
    v_lines = []

    for line in lines:
        x1_l, y1_l, x2_l, y2_l = line[0]
        angle = abs(math.atan2(y2_l - y1_l, x2_l - x1_l) * 180 / np.pi)

        if angle < 20 or angle > 160:  # Horizontal
            h_lines.append(line[0])
        elif 70 < angle < 110:  # Vertical
            v_lines.append(line[0])

    # Determine ruler orientation (more lines = primary direction)
    is_horizontal = len(v_lines) > len(h_lines)
    tick_lines = v_lines if is_horizontal else h_lines

    if len(tick_lines) < 2:
        return None

    # Extract tick positions
    tick_positions = []
    for line in tick_lines:
        x1_l, y1_l, x2_l, y2_l = line
        if is_horizontal:
            pos = (x1_l + x2_l) / 2  # X position for vertical ticks
        else:
            pos = (y1_l + y2_l) / 2  # Y position for horizontal ticks
        tick_positions.append(pos)

    tick_positions = sorted(tick_positions)

    # Calculate intervals between consecutive ticks
    intervals = []
    for i in range(len(tick_positions) - 1):
        intervals.append(tick_positions[i + 1] - tick_positions[i])

    if not intervals:
        return None

    # Find the most common interval (mode)
    intervals_rounded = [round(x) for x in intervals]
    mode_interval = max(set(intervals_rounded), key=intervals_rounded.count)

    # Filter intervals close to the mode
    filtered_intervals = [x for x in intervals if abs(x - mode_interval) < mode_interval * 0.3]
    avg_interval_px = np.mean(filtered_intervals) if filtered_intervals else mode_interval

    # Convert unit to mm
    unit_to_mm = {
        "cm": 10.0,
        "mm": 1.0,
        "inch": 25.4
    }

    mm_per_tick = unit_to_mm[unit]
    pixels_per_mm = avg_interval_px / mm_per_tick

    return {
        "pixels_per_mm": pixels_per_mm,
        "avg_interval_px": avg_interval_px,
        "num_ticks": len(tick_positions),
        "tick_positions": tick_positions,
        "is_horizontal": is_horizontal,
        "roi_offset": (x1, y1)
    }


# ----------------------------------------------------------------------
# Mouse Callback for Box Selection
# ----------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global ruler_box_start, ruler_box_end, is_drawing_box
    global in_ruler_calib_mode, PIXELS_PER_MM, selected_unit
    global calibration_result, detected_ticks

    if not in_ruler_calib_mode:
        return

    def in_rect(px, py, rect):
        rx, ry, rw, rh = rect
        return rx <= px <= rx + rw and ry <= py <= ry + rh

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check unit selection buttons
        if in_rect(x, y, UNIT_CM_BTN):
            selected_unit = "cm"
            calibration_result = None
            return
        elif in_rect(x, y, UNIT_MM_BTN):
            selected_unit = "mm"
            calibration_result = None
            return
        elif in_rect(x, y, UNIT_INCH_BTN):
            selected_unit = "inch"
            calibration_result = None
            return

        # Check action buttons
        if in_rect(x, y, ANALYZE_BTN):
            if ruler_box_start and ruler_box_end and calibration_frozen_frame is not None:
                calibration_result = analyze_ruler_box(
                    calibration_frozen_frame, ruler_box_start, ruler_box_end, selected_unit
                )
                if calibration_result:
                    print(f"✓ Detected {calibration_result['num_ticks']} ticks")
                    print(f"✓ Average interval: {calibration_result['avg_interval_px']:.2f} pixels")
                    print(f"✓ Calculated: {calibration_result['pixels_per_mm']:.4f} px/mm")
                else:
                    print("✗ Could not detect ruler ticks. Try adjusting the box.")
            return

        elif in_rect(x, y, APPLY_BTN):
            if calibration_result:
                PIXELS_PER_MM = calibration_result['pixels_per_mm']
                update_ranges()
                print(f"✓ Calibration applied: {PIXELS_PER_MM:.4f} px/mm")
                in_ruler_calib_mode = False
                ruler_box_start = None
                ruler_box_end = None
                calibration_result = None
            return

        elif in_rect(x, y, CANCEL_BTN):
            in_ruler_calib_mode = False
            ruler_box_start = None
            ruler_box_end = None
            calibration_result = None
            return

        # Start drawing box
        if not in_rect(x, y, (RULER_PANEL_X, RULER_PANEL_Y, RULER_PANEL_W, RULER_PANEL_H)):
            ruler_box_start = (x, y)
            ruler_box_end = (x, y)
            is_drawing_box = True
            calibration_result = None

    elif event == cv2.EVENT_MOUSEMOVE and is_drawing_box:
        ruler_box_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if is_drawing_box:
            ruler_box_end = (x, y)
            is_drawing_box = False


# ----------------------------------------------------------------------
# Detection (same as before)
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
# Draw Calibration UI
# ----------------------------------------------------------------------
def draw_ruler_calibration_mode(frame):
    overlay = frame.copy()

    # Main panel
    cv2.rectangle(overlay, (RULER_PANEL_X, RULER_PANEL_Y),
                  (RULER_PANEL_X + RULER_PANEL_W, RULER_PANEL_Y + RULER_PANEL_H),
                  (30, 30, 50), -1)
    cv2.rectangle(overlay, (RULER_PANEL_X, RULER_PANEL_Y),
                  (RULER_PANEL_X + RULER_PANEL_W, RULER_PANEL_Y + RULER_PANEL_H),
                  (100, 150, 255), 3)

    cv2.putText(overlay, "RULER CALIBRATION", (RULER_PANEL_X + 80, RULER_PANEL_Y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    instructions = [
        "1. Draw a box around the ruler",
        "2. Select the unit (cm/mm/inch)",
        "3. Click ANALYZE to detect ticks",
        "4. Click APPLY to use calibration"
    ]

    y_offset = RULER_PANEL_Y + 60
    for instr in instructions:
        cv2.putText(overlay, instr, (RULER_PANEL_X + 20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_offset += 22

    # Unit selection buttons
    cv2.putText(overlay, "Select Unit:", (RULER_PANEL_X + 20, RULER_PANEL_Y + 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    units = [("cm", UNIT_CM_BTN), ("mm", UNIT_MM_BTN), ("inch", UNIT_INCH_BTN)]
    for unit, btn in units:
        color = (0, 180, 0) if unit == selected_unit else (80, 80, 80)
        cv2.rectangle(overlay, (btn[0], btn[1]), (btn[0] + btn[2], btn[1] + btn[3]), color, -1)
        cv2.putText(overlay, unit.upper(), (btn[0] + 15, btn[1] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Action buttons
    analyze_enabled = ruler_box_start and ruler_box_end
    analyze_color = (0, 150, 200) if analyze_enabled else (80, 80, 80)
    cv2.rectangle(overlay, (ANALYZE_BTN[0], ANALYZE_BTN[1]),
                  (ANALYZE_BTN[0] + ANALYZE_BTN[2], ANALYZE_BTN[1] + ANALYZE_BTN[3]),
                  analyze_color, -1)
    cv2.putText(overlay, "ANALYZE", (ANALYZE_BTN[0] + 12, ANALYZE_BTN[1] + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    apply_enabled = calibration_result is not None
    apply_color = (0, 200, 0) if apply_enabled else (80, 80, 80)
    cv2.rectangle(overlay, (APPLY_BTN[0], APPLY_BTN[1]),
                  (APPLY_BTN[0] + APPLY_BTN[2], APPLY_BTN[1] + APPLY_BTN[3]),
                  apply_color, -1)
    cv2.putText(overlay, "APPLY", (APPLY_BTN[0] + 15, APPLY_BTN[1] + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.rectangle(overlay, (CANCEL_BTN[0], CANCEL_BTN[1]),
                  (CANCEL_BTN[0] + CANCEL_BTN[2], CANCEL_BTN[1] + CANCEL_BTN[3]),
                  (100, 100, 100), -1)
    cv2.putText(overlay, "CANCEL", (CANCEL_BTN[0] + 10, CANCEL_BTN[1] + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display current calibration
    cv2.putText(overlay, f"Current: {PIXELS_PER_MM:.3f} px/mm",
                (RULER_PANEL_X + 20, RULER_PANEL_Y + 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 2)

    # Display analysis results
    if calibration_result:
        result_y = RULER_PANEL_Y + RULER_PANEL_H + 20
        cv2.putText(overlay, f"Detected: {calibration_result['num_ticks']} ticks",
                    (RULER_PANEL_X + 20, result_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(overlay, f"Interval: {calibration_result['avg_interval_px']:.2f}px per {selected_unit}",
                    (RULER_PANEL_X + 20, result_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(overlay, f"New calibration: {calibration_result['pixels_per_mm']:.4f} px/mm",
                    (RULER_PANEL_X + 20, result_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Draw selection box
    if ruler_box_start and ruler_box_end:
        cv2.rectangle(frame, ruler_box_start, ruler_box_end, (0, 255, 255), 2)

        # Draw detected ticks if available
        if calibration_result:
            x_offset, y_offset = calibration_result['roi_offset']
            is_horiz = calibration_result['is_horizontal']

            for pos in calibration_result['tick_positions']:
                if is_horiz:
                    pt1 = (int(pos + x_offset), ruler_box_start[1])
                    pt2 = (int(pos + x_offset), ruler_box_end[1])
                else:
                    pt1 = (ruler_box_start[0], int(pos + y_offset))
                    pt2 = (ruler_box_end[0], int(pos + y_offset))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 1)


# ----------------------------------------------------------------------
# Main Overlay
# ----------------------------------------------------------------------
def draw_overlay(frame, pellets):
    total = len(pellets)
    within = sum(1 for p in pellets if p['within_tolerance'])
    out_of = total - within
    status_text = f"In: {within}   Out: {out_of}   Total: {total}"
    status_color = (0, 255, 0) if out_of == 0 else (0, 0, 255)

    cv2.rectangle(frame, (10, 10), (460, 50), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (460, 50), status_color, 2)
    cv2.putText(frame, status_text, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

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

    if not in_ruler_calib_mode:
        cv2.putText(frame, "Press 'r' for ruler calibration | 'q' to quit",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)

    if in_ruler_calib_mode:
        draw_ruler_calibration_mode(frame)

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
    global in_ruler_calib_mode, calibration_frozen_frame
    global ruler_box_start, ruler_box_end, calibration_result

    print("\nPellet Inspector with Box-Based Ruler Calibration")
    print("=" * 60)
    print("Press 'r' -> Enter calibration mode (draw box around ruler)")
    print("Press 'q' -> Quit")
    print("=" * 60)

    cap = get_camera()
    if not cap.isOpened():
        print("Cannot open camera.")
        sys.exit(1)

    fps_counter = 0
    fps_start = time.time()
    fps_display = 0

    window_name = "Pellet Inspector"
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

        if not in_ruler_calib_mode:
            pellets = detect_pellets(display_frame)
            display_frame = draw_overlay(display_frame, pellets)
        else:
            display_frame = draw_overlay(display_frame, [])

        fps_counter += 1
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
                ruler_box_start = None
                ruler_box_end = None
                calibration_result = None
                calibration_frozen_frame = None
            else:
                in_ruler_calib_mode = True

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Shutdown complete.")


if __name__ == "__main__":
    main()