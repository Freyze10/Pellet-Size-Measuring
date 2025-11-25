import cv2
import numpy as np
import time
import sys
import math
from collections import defaultdict

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

# ----------------------------------------------------------------------
# Ruler Calibration State
# ----------------------------------------------------------------------
in_ruler_calib_mode = False
in_auto_detect_mode = False
REFERENCE_LENGTH_MM = 76.2  # 3 inches = 76.2mm
reference_line_start = None
reference_line_end = None
calibration_frozen_frame = None
is_dragging = False

# Auto-detection state
detected_ruler_lines = []
detected_ticks = []
auto_calib_result = None

# Button rectangles for ruler calibration
RULER_PANEL_X, RULER_PANEL_Y = 10, 80
RULER_PANEL_W, RULER_PANEL_H = 380, 340

RESET_BTN = (RULER_PANEL_X + 20, RULER_PANEL_Y + 260, 100, 40)
APPLY_BTN = (RULER_PANEL_X + 140, RULER_PANEL_Y + 260, 100, 40)
CANCEL_BTN = (RULER_PANEL_X + 260, RULER_PANEL_Y + 260, 100, 40)
AUTO_BTN = (RULER_PANEL_X + 20, RULER_PANEL_Y + 310, 340, 40)


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


def line_intersection(line1, line2):
    """Find intersection point of two lines defined by (rho, theta)"""
    rho1, theta1 = line1
    rho2, theta2 = line2

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])

    try:
        x0, y0 = np.linalg.solve(A, b)
        return int(x0[0]), int(y0[0])
    except:
        return None


def angle_difference(theta1, theta2):
    """Calculate smallest angle difference between two angles"""
    diff = abs(theta1 - theta2)
    return min(diff, np.pi - diff)


# ----------------------------------------------------------------------
# Automatic Ruler Detection
# ----------------------------------------------------------------------
def detect_ruler_automatically(frame):
    """Automatically detect ruler lines and tick marks"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Edge detection with careful thresholding
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100, apertureSize=3)

    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)

    if lines is None:
        return None, None, "No lines detected. Ensure ruler is clearly visible."

    # Group lines by angle (find dominant ruler edge)
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        rho, theta = line[0]
        # Horizontal lines (around 0 or π)
        if abs(theta) < np.pi / 4 or abs(theta - np.pi) < np.pi / 4:
            horizontal_lines.append((rho, theta))
        # Vertical lines (around π/2)
        elif abs(theta - np.pi / 2) < np.pi / 4:
            vertical_lines.append((rho, theta))

    # Use the orientation with more lines
    if len(horizontal_lines) > len(vertical_lines):
        ruler_lines = horizontal_lines
        is_horizontal = True
    else:
        ruler_lines = vertical_lines
        is_horizontal = False

    if len(ruler_lines) < 5:
        return None, None, "Not enough ruler markings detected."

    # Detect tick marks perpendicular to ruler edge
    tick_lines = vertical_lines if is_horizontal else horizontal_lines

    if len(tick_lines) < 3:
        return None, None, "Could not detect tick marks."

    # Sort tick positions
    if is_horizontal:
        tick_positions = sorted([abs(rho) for rho, theta in tick_lines])
    else:
        tick_positions = sorted([abs(rho) for rho, theta in tick_lines])

    # Find consistent spacing (look for repeated intervals)
    spacings = []
    for i in range(len(tick_positions) - 1):
        spacing = tick_positions[i + 1] - tick_positions[i]
        if 10 < spacing < 200:  # Reasonable pixel range for ruler marks
            spacings.append(spacing)

    if not spacings:
        return None, None, "Could not find consistent tick spacing."

    # Cluster spacings to find the most common interval
    spacing_counts = defaultdict(list)
    for s in spacings:
        # Group spacings within 10% tolerance
        found = False
        for key in spacing_counts.keys():
            if abs(s - key) / key < 0.1:
                spacing_counts[key].append(s)
                found = True
                break
        if not found:
            spacing_counts[s].append(s)

    # Find most common spacing
    most_common = max(spacing_counts.items(), key=lambda x: len(x[1]))
    avg_spacing_px = np.mean(most_common[1])

    # Determine if spacing is mm, cm, or inch based on typical pixel values
    # Assume camera is at reasonable distance
    if 3 < avg_spacing_px < 12:
        # Likely 1mm marks
        spacing_mm = 1.0
    elif 12 < avg_spacing_px < 80:
        # Likely 1cm or 1/8 inch marks
        # Check if it's closer to cm (10mm) or 1/8 inch (3.175mm)
        ratio_to_cm = avg_spacing_px / 10
        ratio_to_eighth_inch = avg_spacing_px / 3.175
        if abs(ratio_to_cm - round(ratio_to_cm)) < abs(ratio_to_eighth_inch - round(ratio_to_eighth_inch)):
            spacing_mm = 10.0  # 1cm
        else:
            spacing_mm = 3.175  # 1/8 inch
    elif 80 < avg_spacing_px < 200:
        # Likely 1 inch marks
        spacing_mm = 25.4
    else:
        return None, None, "Spacing outside expected range."

    # Calculate pixels per mm
    pixels_per_mm = avg_spacing_px / spacing_mm

    result = {
        'pixels_per_mm': pixels_per_mm,
        'spacing_px': avg_spacing_px,
        'spacing_mm': spacing_mm,
        'tick_count': len(tick_positions),
        'is_horizontal': is_horizontal,
        'ruler_lines': ruler_lines[:3],  # Keep first 3 for visualization
        'tick_lines': tick_lines[:20]  # Keep up to 20 ticks for visualization
    }

    return result, edges, None


# ----------------------------------------------------------------------
# Mouse Callback for Ruler Calibration
# ----------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global reference_line_start, reference_line_end, is_dragging
    global in_ruler_calib_mode, in_auto_detect_mode, PIXELS_PER_MM
    global auto_calib_result, calibration_frozen_frame

    if not in_ruler_calib_mode:
        return

    def in_rect(px, py, rect):
        rx, ry, rw, rh = rect
        return rx <= px <= rx + rw and ry <= py <= ry + rh

    # Handle button clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        if in_rect(x, y, RESET_BTN):
            reference_line_start = None
            reference_line_end = None
            is_dragging = False
            in_auto_detect_mode = False
            auto_calib_result = None
            return
        elif in_rect(x, y, APPLY_BTN):
            if auto_calib_result:
                # Apply auto-detected calibration
                PIXELS_PER_MM = auto_calib_result['pixels_per_mm']
                update_ranges()
                print(f"Auto-calibrated: {PIXELS_PER_MM:.4f} px/mm from {auto_calib_result['spacing_mm']}mm intervals")
                in_ruler_calib_mode = False
                in_auto_detect_mode = False
                auto_calib_result = None
                reference_line_start = None
                reference_line_end = None
            elif reference_line_start and reference_line_end:
                # Apply manual calibration
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
            in_auto_detect_mode = False
            auto_calib_result = None
            reference_line_start = None
            reference_line_end = None
            is_dragging = False
            return
        elif in_rect(x, y, AUTO_BTN):
            # Trigger auto-detection
            in_auto_detect_mode = True
            if calibration_frozen_frame is not None:
                result, edges, error = detect_ruler_automatically(calibration_frozen_frame)
                if result:
                    auto_calib_result = result
                    print(f"Auto-detected: {result['spacing_mm']}mm marks, {result['pixels_per_mm']:.4f} px/mm")
                else:
                    print(f"Auto-detection failed: {error}")
                    auto_calib_result = None
            return

        # Manual line drawing (only if not in panel)
        if not in_rect(x, y, (RULER_PANEL_X, RULER_PANEL_Y, RULER_PANEL_W, RULER_PANEL_H)):
            reference_line_start = (x, y)
            reference_line_end = (x, y)
            is_dragging = True
            in_auto_detect_mode = False
            auto_calib_result = None

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
# Draw Ruler Calibration Mode
# ----------------------------------------------------------------------
def draw_ruler_calibration_mode(frame):
    overlay = frame.copy()

    cv2.rectangle(overlay, (RULER_PANEL_X, RULER_PANEL_Y),
                  (RULER_PANEL_X + RULER_PANEL_W, RULER_PANEL_Y + RULER_PANEL_H),
                  (30, 30, 50), -1)
    cv2.rectangle(overlay, (RULER_PANEL_X, RULER_PANEL_Y),
                  (RULER_PANEL_X + RULER_PANEL_W, RULER_PANEL_Y + RULER_PANEL_H),
                  (100, 150, 255), 3)

    cv2.putText(overlay, "RULER CALIBRATION", (RULER_PANEL_X + 80, RULER_PANEL_Y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if auto_calib_result:
        # Show auto-detection results
        instructions = [
            f"AUTO-DETECTED:",
            f"Tick spacing: {auto_calib_result['spacing_mm']:.1f}mm",
            f"({auto_calib_result['tick_count']} marks found)",
            f"New calibration:",
            f"{auto_calib_result['pixels_per_mm']:.4f} px/mm",
            "",
            "Click APPLY to use this calibration"
        ]
    else:
        instructions = [
            "MANUAL MODE:",
            "1. Place ruler in camera view",
            "2. Click and drag a 3-inch line",
            "   OR",
            "3. Click AUTO-DETECT to scan",
            "   ruler markings automatically"
        ]

    y_offset = RULER_PANEL_Y + 60
    for instr in instructions:
        cv2.putText(overlay, instr, (RULER_PANEL_X + 20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_offset += 22

    if not auto_calib_result:
        cv2.putText(overlay, "Reference: 3 inch (76.2 mm)",
                    (RULER_PANEL_X + 60, RULER_PANEL_Y + 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)

    # Buttons
    cv2.rectangle(overlay, (RESET_BTN[0], RESET_BTN[1]),
                  (RESET_BTN[0] + RESET_BTN[2], RESET_BTN[1] + RESET_BTN[3]),
                  (50, 50, 200), -1)
    cv2.putText(overlay, "RESET", (RESET_BTN[0] + 15, RESET_BTN[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    apply_enabled = (reference_line_start and reference_line_end) or auto_calib_result
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

    # Auto-detect button
    cv2.rectangle(overlay, (AUTO_BTN[0], AUTO_BTN[1]),
                  (AUTO_BTN[0] + AUTO_BTN[2], AUTO_BTN[1] + AUTO_BTN[3]),
                  (0, 150, 200), -1)
    cv2.putText(overlay, "AUTO-DETECT RULER", (AUTO_BTN[0] + 55, AUTO_BTN[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(overlay, f"Current: {PIXELS_PER_MM:.2f} px/mm",
                (RULER_PANEL_X + 100, RULER_PANEL_Y + 245),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)

    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    # Draw auto-detected lines
    if auto_calib_result:
        # Draw detected ruler edge lines (in green)
        for rho, theta in auto_calib_result['ruler_lines']:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Draw detected tick marks (in cyan)
        for rho, theta in auto_calib_result['tick_lines']:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 500 * (-b))
            y1 = int(y0 + 500 * (a))
            x2 = int(x0 - 500 * (-b))
            y2 = int(y0 - 500 * (a))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

    # Draw manual reference line
    elif reference_line_start and reference_line_end:
        cv2.line(frame, reference_line_start, reference_line_end, (0, 255, 255), 2)

        # Crosshairs
        for pt in [reference_line_start, reference_line_end]:
            cv2.line(frame, (pt[0] - 10, pt[1]), (pt[0] + 10, pt[1]), (0, 0, 255), 2)
            cv2.line(frame, (pt[0], pt[1] - 10), (pt[0], pt[1] + 10), (0, 0, 255), 2)

        dx = reference_line_end[0] - reference_line_start[0]
        dy = reference_line_end[1] - reference_line_start[1]
        length = math.sqrt(dx ** 2 + dy ** 2)

        mid_x = (reference_line_start[0] + reference_line_end[0]) // 2
        mid_y = (reference_line_start[1] + reference_line_end[1]) // 2

        cv2.putText(frame, f"{length:.1f} px = 3 inches",
                    (mid_x - 70, mid_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


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
# Camera
# ----------------------------------------------------------------------
def get_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    return cap


# ----------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------
def main():
    global in_ruler_calib_mode, calibration_frozen_frame

    print("\nPellet Inspector with AUTO Ruler Detection")
    print("=" * 55)
    print("Press 'r' -> Calibrate (Manual or Auto-detect)")
    print("Press 'q' -> Quit")
    print("=" * 55)

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
        elif key == ord('r') and not in_ruler_calib_mode:
            in_ruler_calib_mode = True

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Shutdown complete.")


if __name__ == "__main__":
    main()