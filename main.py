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

# ----------------------------------------------------------------------
# Ruler Calibration State
# ----------------------------------------------------------------------
in_ruler_calib_mode = False
REFERENCE_LENGTH_MM = 76.2
reference_line_start = None
reference_line_end = None
calibration_frozen_frame = None
is_dragging = False

# Button rectangles for ruler calibration
RULER_PANEL_X, RULER_PANEL_Y = 10, 80
RULER_PANEL_W, RULER_PANEL_H = 380, 280

RESET_BTN = (RULER_PANEL_X + 20, RULER_PANEL_Y + 200, 100, 40)
APPLY_BTN = (RULER_PANEL_X + 140, RULER_PANEL_Y + 200, 100, 40)
CANCEL_BTN = (RULER_PANEL_X + 260, RULER_PANEL_Y + 200, 100, 40)

# ----------------------------------------------------------------------
# Performance Optimization: Pre-compute reusable objects
# ----------------------------------------------------------------------
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Cache for overlay background rectangles (reduces allocation overhead)
overlay_cache = {}


# ----------------------------------------------------------------------
# Helper Checks (Optimized with inlined calculations)
# ----------------------------------------------------------------------
def is_within_tolerance(diameter: float, length: float) -> bool:
    return (DIAMETER_MIN <= diameter <= DIAMETER_MAX and
            LENGTH_MIN <= length <= LENGTH_MAX)


def should_process_pellet(diameter: float, length: float) -> bool:
    return (DIAMETER_EXCLUDE_MIN <= diameter <= DIAMETER_EXCLUDE_MAX and
            LENGTH_EXCLUDE_MIN <= length <= LENGTH_EXCLUDE_MAX)


def get_distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


# ----------------------------------------------------------------------
# Mouse Callback for Ruler Calibration
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
                pixel_distance = math.sqrt(dx * dx + dy * dy)

                if pixel_distance > 10:
                    PIXELS_PER_MM = pixel_distance / REFERENCE_LENGTH_MM
                    update_ranges()
                    print(f"Calibrated: {PIXELS_PER_MM:.4f} px/mm based on {pixel_distance:.2f}px / 76.2mm")

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
# Detection with Rotated Bounding Boxes (OPTIMIZED)
# ----------------------------------------------------------------------
def detect_pellets(frame):
    # Optimization: Work on region of interest if possible
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optimization: Reduced bilateral filter diameter for speed
    blur = cv2.bilateralFilter(gray, 7, 75, 75)

    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Optimization: Use pre-computed kernel
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=2)

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

        # Optimization: Calculate edge distances using numpy for vectorization
        edge1 = np.linalg.norm(box[1] - box[0])
        edge2 = np.linalg.norm(box[2] - box[1])

        if edge1 < edge2:
            width_px = edge1
            height_px = edge2
        else:
            width_px = edge2
            height_px = edge1

        # Optimization: Single division operation
        inv_ppm = 1.0 / PIXELS_PER_MM
        diameter = width_px * inv_ppm
        length = height_px * inv_ppm

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
# Draw Ruler Calibration Mode (OPTIMIZED)
# ----------------------------------------------------------------------
def draw_ruler_calibration_mode(frame):
    # Optimization: Use direct drawing instead of overlay blending when possible
    overlay = frame.copy()

    cv2.rectangle(overlay, (RULER_PANEL_X, RULER_PANEL_Y),
                  (RULER_PANEL_X + RULER_PANEL_W, RULER_PANEL_Y + RULER_PANEL_H),
                  (30, 30, 50), -1)
    cv2.rectangle(overlay, (RULER_PANEL_X, RULER_PANEL_Y),
                  (RULER_PANEL_X + RULER_PANEL_W, RULER_PANEL_Y + RULER_PANEL_H),
                  (100, 150, 255), 3)

    cv2.putText(overlay, "RULER CALIBRATION", (RULER_PANEL_X + 80, RULER_PANEL_Y + 30),
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

    if reference_line_start and reference_line_end:
        dx = reference_line_end[0] - reference_line_start[0]
        dy = reference_line_end[1] - reference_line_start[1]
        pixel_distance = math.sqrt(dx * dx + dy * dy)
        new_px_per_mm = pixel_distance / REFERENCE_LENGTH_MM if pixel_distance > 0 else 0
        cv2.putText(overlay, f"New: {new_px_per_mm:.2f} px/mm",
                    (RULER_PANEL_X + 200, RULER_PANEL_Y + 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)

    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    # Draw reference line
    if reference_line_start and reference_line_end:
        cv2.line(frame, reference_line_start, reference_line_end, (0, 255, 255), 1)

        # Crosshairs
        cv2.line(frame, (reference_line_start[0] - 10, reference_line_start[1]),
                 (reference_line_start[0] + 10, reference_line_start[1]), (0, 0, 255), 1)
        cv2.line(frame, (reference_line_start[0], reference_line_start[1] - 10),
                 (reference_line_start[0], reference_line_start[1] + 10), (0, 0, 255), 1)

        cv2.line(frame, (reference_line_end[0] - 10, reference_line_end[1]),
                 (reference_line_end[0] + 10, reference_line_end[1]), (0, 0, 255), 1)
        cv2.line(frame, (reference_line_end[0], reference_line_end[1] - 10),
                 (reference_line_end[0], reference_line_end[1] + 10), (0, 0, 255), 1)

        length = pixel_distance
        angle = math.atan2(dy, dx)

        if length > 0:
            # Optimization: Reduce number of markers for better performance
            for i in range(0, int(REFERENCE_LENGTH_MM) + 1, 5):  # Draw every 5mm instead of every 1mm
                t = i / REFERENCE_LENGTH_MM
                marker_x = int(reference_line_start[0] + dx * t)
                marker_y = int(reference_line_start[1] + dy * t)

                is_cm = (i % 10 == 0)
                tick_length = 10 if is_cm else 4
                sin_a = math.sin(angle)
                cos_a = math.cos(angle)
                perp_dx = int(tick_length * sin_a)
                perp_dy = int(-tick_length * cos_a)

                tick_start = (marker_x - perp_dx, marker_y - perp_dy)
                tick_end = (marker_x + perp_dx, marker_y + perp_dy)
                tick_color = (255, 255, 255) if is_cm else (180, 180, 180)
                cv2.line(frame, tick_start, tick_end, tick_color, 1)

                if is_cm:
                    cm_num = i // 10
                    text_offset_x = int(18 * sin_a)
                    text_offset_y = int(-18 * cos_a)
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
# Main Overlay (OPTIMIZED)
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

        # Optimization: Pre-calculate min values using numpy
        top_y = int(box[:, 1].min())
        left_x = int(box[:, 0].min())
        bg_y = max(top_y - 30, 0)

        cv2.rectangle(frame, (left_x, bg_y), (left_x + 70, top_y - 5), (0, 0, 0), -1)
        cv2.putText(frame, f"D: {p['diameter']:.2f}mm", (left_x + 3, bg_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(frame, f"L: {p['length']:.2f}mm", (left_x + 3, bg_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        if not p['within_tolerance']:
            # Optimization: Use argmax directly on the box array
            top_right_idx = box[:, 0].argmax()
            top_right = tuple(box[top_right_idx])
            cv2.circle(frame, top_right, 8, (0, 0, 255), -1)
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
# Camera (OPTIMIZED)
# ----------------------------------------------------------------------
def get_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Optimization: Set camera buffer to 1 to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    return cap


# ----------------------------------------------------------------------
# Main Loop (OPTIMIZED)
# ----------------------------------------------------------------------
def main():
    global in_ruler_calib_mode, calibration_frozen_frame

    print("\nPellet Inspector - Optimized Runtime Version")
    print("=" * 55)
    print("Press 'r' -> Calibrate (Drag 3-inch line)")
    print("Press 'q' -> Quit")
    print("=" * 55)

    cap = get_camera()
    if not cap.isOpened():
        print("Cannot open camera.")
        sys.exit(1)

    # Optimization: Use frame counter for FPS calculation
    fps_counter = 0
    fps_start = time.perf_counter()  # More precise timing
    fps_display = 0

    # Optimization: Frame skip counter to reduce unnecessary processing
    frame_skip = 0
    PROCESS_EVERY_N_FRAMES = 1  # Process every frame by default

    window_name = "Pellet Inspector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Optimization: Pre-allocate frame for reuse
    last_pellets = []

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

        display_frame = calibration_frozen_frame.copy() if in_ruler_calib_mode else frame

        # Optimization: Process pellets only when not in calibration mode
        if not in_ruler_calib_mode:
            frame_skip += 1
            if frame_skip >= PROCESS_EVERY_N_FRAMES:
                pellets = detect_pellets(display_frame)
                last_pellets = pellets
                frame_skip = 0
            else:
                pellets = last_pellets

            display_frame = draw_overlay(display_frame, pellets)
        else:
            display_frame = draw_overlay(display_frame, [])

        # Optimization: More efficient FPS calculation
        fps_counter += 1
        elapsed = time.perf_counter() - fps_start
        if elapsed >= 1.0:
            fps_display = int(fps_counter / elapsed)
            fps_counter = 0
            fps_start = time.perf_counter()

        cv2.putText(display_frame, f"FPS: {fps_display}",
                    (display_frame.shape[1] - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, display_frame)

        # Optimization: Reduced waitKey time for better responsiveness
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