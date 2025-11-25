import cv2
import numpy as np
import time
import sys
import math
import json
import os

# ----------------------------------------------------------------------
# Global Calibration Storage (Resolution-Aware)
# ----------------------------------------------------------------------
CALIBRATION_FILE = "pellet_calibration.json"
calibrations = {}  # Key: "widthxheight", Value: pixels_per_mm

# Current resolution tracking
current_resolution = None
PIXELS_PER_MM = 6.0  # Default fallback

TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0


def load_calibrations():
    """Load saved calibrations from file."""
    global calibrations
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                calibrations = json.load(f)
            print(f"Loaded calibrations: {calibrations}")
        except Exception as e:
            print(f"Could not load calibrations: {e}")
            calibrations = {}
    else:
        calibrations = {}


def save_calibrations():
    """Save calibrations to file."""
    try:
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(calibrations, f, indent=2)
        print(f"Calibrations saved: {calibrations}")
    except Exception as e:
        print(f"Could not save calibrations: {e}")


def get_resolution_key(width, height):
    """Create a unique key for resolution."""
    return f"{int(width)}x{int(height)}"


def set_calibration_for_resolution(width, height, pixels_per_mm):
    """Store calibration for specific resolution."""
    global calibrations, PIXELS_PER_MM
    key = get_resolution_key(width, height)
    calibrations[key] = pixels_per_mm
    PIXELS_PER_MM = pixels_per_mm
    save_calibrations()
    update_ranges()
    print(f"✓ Calibration set for {key}: {pixels_per_mm:.4f} px/mm")


def get_calibration_for_resolution(width, height):
    """Retrieve calibration for specific resolution."""
    global PIXELS_PER_MM
    key = get_resolution_key(width, height)
    if key in calibrations:
        PIXELS_PER_MM = calibrations[key]
        print(f"✓ Using calibration for {key}: {PIXELS_PER_MM:.4f} px/mm")
        return True
    else:
        print(f"⚠ No calibration found for {key}. Using default: {PIXELS_PER_MM:.4f} px/mm")
        print(f"  Press 'r' to calibrate this resolution.")
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

# Common resolutions: 640x480, 1280x720, 1920x1080
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720

# ----------------------------------------------------------------------
# Ruler Calibration State
# ----------------------------------------------------------------------
in_ruler_calib_mode = False
REFERENCE_LENGTH_MM = 76.2  # 3 inches = 76.2mm
reference_line_start = None
reference_line_end = None
calibration_frozen_frame = None
is_dragging = False

# Button rectangles for ruler calibration
RULER_PANEL_X, RULER_PANEL_Y = 10, 80
RULER_PANEL_W, RULER_PANEL_H = 380, 310  # Increased height for resolution info

RESET_BTN = (RULER_PANEL_X + 20, RULER_PANEL_Y + 230, 100, 40)
APPLY_BTN = (RULER_PANEL_X + 140, RULER_PANEL_Y + 230, 100, 40)
CANCEL_BTN = (RULER_PANEL_X + 260, RULER_PANEL_Y + 230, 100, 40)


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
# Mouse Callback for Ruler Calibration
# ----------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global reference_line_start, reference_line_end, is_dragging
    global in_ruler_calib_mode, current_resolution

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
            return
        elif in_rect(x, y, APPLY_BTN):
            if reference_line_start and reference_line_end and current_resolution:
                dx = reference_line_end[0] - reference_line_start[0]
                dy = reference_line_end[1] - reference_line_start[1]
                pixel_distance = math.sqrt(dx ** 2 + dy ** 2)

                if pixel_distance > 10:
                    new_px_per_mm = pixel_distance / REFERENCE_LENGTH_MM
                    w, h = current_resolution
                    set_calibration_for_resolution(w, h, new_px_per_mm)

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

    # Show current resolution
    if current_resolution:
        res_text = f"Resolution: {get_resolution_key(*current_resolution)}"
        cv2.putText(overlay, res_text, (RULER_PANEL_X + 90, RULER_PANEL_Y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)

    instructions = [
        "1. Place a ruler in camera view",
        "2. Click and drag to match the",
        "   reference line (3 inch / 7.62 cm)",
        "3. Click APPLY when aligned"
    ]

    y_offset = RULER_PANEL_Y + 75
    for instr in instructions:
        cv2.putText(overlay, instr, (RULER_PANEL_X + 20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_offset += 20

    cv2.putText(overlay, "Reference: 3 inch (76.2 mm)",
                (RULER_PANEL_X + 60, RULER_PANEL_Y + 180),
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
                (RULER_PANEL_X + 20, RULER_PANEL_Y + 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2)

    if reference_line_start and reference_line_end:
        dx = reference_line_end[0] - reference_line_start[0]
        dy = reference_line_end[1] - reference_line_start[1]
        pixel_distance = math.sqrt(dx ** 2 + dy ** 2)
        new_px_per_mm = pixel_distance / REFERENCE_LENGTH_MM if pixel_distance > 0 else 0
        cv2.putText(overlay, f"New: {new_px_per_mm:.2f} px/mm",
                    (RULER_PANEL_X + 200, RULER_PANEL_Y + 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)

    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    # Draw reference line with markers
    if reference_line_start and reference_line_end:
        cv2.line(frame, reference_line_start, reference_line_end, (0, 255, 255), 1)

        cv2.line(frame, (reference_line_start[0] - 10, reference_line_start[1]),
                 (reference_line_start[0] + 10, reference_line_start[1]), (0, 0, 255), 1)
        cv2.line(frame, (reference_line_start[0], reference_line_start[1] - 10),
                 (reference_line_start[0], reference_line_start[1] + 10), (0, 0, 255), 1)

        cv2.line(frame, (reference_line_end[0] - 10, reference_line_end[1]),
                 (reference_line_end[0] + 10, reference_line_end[1]), (0, 0, 255), 1)
        cv2.line(frame, (reference_line_end[0], reference_line_end[1] - 10),
                 (reference_line_end[0], reference_line_end[1] + 10), (0, 0, 255), 1)

        dx = reference_line_end[0] - reference_line_start[0]
        dy = reference_line_end[1] - reference_line_start[1]
        length = math.sqrt(dx ** 2 + dy ** 2)
        angle = math.atan2(dy, dx)

        if length > 0:
            for i in range(int(REFERENCE_LENGTH_MM) + 1):
                t = i / REFERENCE_LENGTH_MM
                marker_x = int(reference_line_start[0] + dx * t)
                marker_y = int(reference_line_start[1] + dy * t)

                is_cm = (i % 10 == 0)
                tick_length = 10 if is_cm else 4
                perp_dx = int(tick_length * math.sin(angle))
                perp_dy = int(-tick_length * math.cos(angle))

                tick_start = (marker_x - perp_dx, marker_y - perp_dy)
                tick_end = (marker_x + perp_dx, marker_y + perp_dy)
                tick_color = (255, 255, 255) if is_cm else (180, 180, 180)
                cv2.line(frame, tick_start, tick_end, tick_color, 1)

                if is_cm:
                    cm_num = i // 10
                    text_offset_x = int(18 * math.sin(angle))
                    text_offset_y = int(-18 * math.cos(angle))
                    cv2.putText(frame, f"{cm_num}",
                                (marker_x + text_offset_x - 5, marker_y + text_offset_y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        mid_x = (reference_line_start[0] + reference_line_end[0]) // 2
        mid_y = (reference_line_start[1] + reference_line_end[1]) // 2
        text_offset_x = int(-20 * math.sin(angle))
        text_offset_y = int(20 * math.cos(angle))

        cv2.putText(frame, f"{length:.1f} px",
                    (mid_x + text_offset_x, mid_y + text_offset_y),
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

    # Show calibration status and resolution info
    if current_resolution:
        res_key = get_resolution_key(*current_resolution)
        is_calibrated = res_key in calibrations
        calib_status = "CALIBRATED" if is_calibrated else "NOT CALIBRATED"
        calib_color = (0, 255, 0) if is_calibrated else (0, 165, 255)

        info_text = f"{res_key} | {calib_status} | {PIXELS_PER_MM:.2f} px/mm"
        cv2.putText(frame, info_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, calib_color, 2)

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

    return cap, (actual_w, actual_h)


# ----------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------
def main():
    global in_ruler_calib_mode, calibration_frozen_frame, current_resolution

    print("\n" + "=" * 60)
    print("Pellet Inspector - Resolution-Independent Calibration")
    print("=" * 60)

    # Load saved calibrations
    load_calibrations()

    print("\nEach camera resolution needs separate calibration.")
    print("Press 'r' -> Calibrate current resolution")
    print("Press 'q' -> Quit")
    print("=" * 60 + "\n")

    cap, resolution = get_camera()
    if not cap.isOpened():
        print("Cannot open camera.")
        sys.exit(1)

    current_resolution = resolution
    get_calibration_for_resolution(*resolution)

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
            cap, new_resolution = get_camera()
            if not cap.isOpened():
                break

            # Check if resolution changed
            if new_resolution != current_resolution:
                current_resolution = new_resolution
                get_calibration_for_resolution(*new_resolution)
                print(f"⚠ Resolution changed to {get_resolution_key(*new_resolution)}")

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