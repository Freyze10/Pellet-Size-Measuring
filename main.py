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
MAX_CONTOUR_AREA = 10000

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
RULER_PANEL_W, RULER_PANEL_H = 380, 280

RESET_BTN = (RULER_PANEL_X + 20, RULER_PANEL_Y + 200, 100, 40)
APPLY_BTN = (RULER_PANEL_X + 140, RULER_PANEL_Y + 200, 100, 40)
CANCEL_BTN = (RULER_PANEL_X + 260, RULER_PANEL_Y + 200, 100, 40)


# ----------------------------------------------------------------------
# Helper Checks
# ----------------------------------------------------------------------
def is_within_tolerance(diameter: float, length: float) -> bool:
    return (DIAMETER_MIN <= diameter <= DIAMETER_MAX and
            LENGTH_MIN <= length <= LENGTH_MAX)


def should_process_pellet(diameter: float, length: float) -> bool:
    return (DIAMETER_EXCLUDE_MIN <= diameter <= DIAMETER_EXCLUDE_MAX and
            LENGTH_EXCLUDE_MIN <= length <= LENGTH_EXCLUDE_MAX)


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

    # Handle button clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        if in_rect(x, y, RESET_BTN):
            reference_line_start = None
            reference_line_end = None
            is_dragging = False
            return
        elif in_rect(x, y, APPLY_BTN):
            if reference_line_start and reference_line_end:
                # Calculate pixels per mm based on 3 inch reference
                dx = reference_line_end[0] - reference_line_start[0]
                dy = reference_line_end[1] - reference_line_start[1]
                pixel_distance = math.sqrt(dx ** 2 + dy ** 2)
                PIXELS_PER_MM = round(pixel_distance / REFERENCE_LENGTH_MM, 2)
                update_ranges()
                # Exit calibration mode
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

        # Start dragging the reference line
        if not in_rect(x, y, (RULER_PANEL_X, RULER_PANEL_Y, RULER_PANEL_W, RULER_PANEL_H)):
            reference_line_start = (x, y)
            reference_line_end = (x, y)
            is_dragging = True

    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        # Update end point while dragging
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

    # Enhanced preprocessing for better edge detection
    # Apply bilateral filter to reduce noise while keeping edges sharp
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

    # Use Canny edge detection for precise edges
    edges = cv2.Canny(bilateral, 50, 150)

    # Dilate slightly to connect nearby edges
    kernel_dilate = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel_dilate, iterations=1)

    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pellets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA):
            continue

        # Filter by shape - pellets should be relatively compact
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Skip very irregular shapes (circularity should be reasonable for cylindrical pellets)
        if circularity < 0.3:
            continue

        # Fit an ellipse for more accurate measurement
        if len(cnt) >= 5:  # Need at least 5 points to fit ellipse
            try:
                ellipse = cv2.fitEllipse(cnt)
                center, (width_px, height_px), angle = ellipse

                # Get the actual contour bounding box for visualization
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)

                # Use ellipse dimensions for measurement (more accurate)
                width_mm = width_px / PIXELS_PER_MM
                height_mm = height_px / PIXELS_PER_MM

                # Diameter is the smaller dimension, length is the larger
                diameter = min(width_mm, height_mm)
                length = max(width_mm, height_mm)

                if should_process_pellet(diameter, length):
                    pellets.append({
                        'contour': cnt,
                        'box': box,
                        'ellipse': ellipse,  # Store ellipse for better visualization
                        'center': center,
                        'angle': angle,
                        'width_px': width_px,
                        'height_px': height_px,
                        'diameter': diameter,
                        'length': length,
                        'within_tolerance': is_within_tolerance(diameter, length)
                    })
            except:
                # If ellipse fitting fails, fall back to minAreaRect
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                center, (width_px, height_px), angle = rect

                width_mm = width_px / PIXELS_PER_MM
                height_mm = height_px / PIXELS_PER_MM

                diameter = min(width_mm, height_mm)
                length = max(width_mm, height_mm)

                if should_process_pellet(diameter, length):
                    pellets.append({
                        'contour': cnt,
                        'box': box,
                        'ellipse': None,
                        'center': center,
                        'angle': angle,
                        'width_px': width_px,
                        'height_px': height_px,
                        'diameter': diameter,
                        'length': length,
                        'within_tolerance': is_within_tolerance(diameter, length)
                    })
        else:
            # For small contours, use minAreaRect
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            center, (width_px, height_px), angle = rect

            width_mm = width_px / PIXELS_PER_MM
            height_mm = height_px / PIXELS_PER_MM

            diameter = min(width_mm, height_mm)
            length = max(width_mm, height_mm)

            if should_process_pellet(diameter, length):
                pellets.append({
                    'contour': cnt,
                    'box': box,
                    'ellipse': None,
                    'center': center,
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

    # Semi-transparent panel background
    cv2.rectangle(overlay, (RULER_PANEL_X, RULER_PANEL_Y),
                  (RULER_PANEL_X + RULER_PANEL_W, RULER_PANEL_Y + RULER_PANEL_H),
                  (30, 30, 50), -1)
    cv2.rectangle(overlay, (RULER_PANEL_X, RULER_PANEL_Y),
                  (RULER_PANEL_X + RULER_PANEL_W, RULER_PANEL_Y + RULER_PANEL_H),
                  (100, 150, 255), 3)

    # Title
    cv2.putText(overlay, "RULER CALIBRATION", (RULER_PANEL_X + 80, RULER_PANEL_Y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Instructions
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

    # Reference measurement display
    cv2.putText(overlay, "Reference: 3 inch (76.2 mm)",
                (RULER_PANEL_X + 60, RULER_PANEL_Y + 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)

    # Buttons
    # RESET
    cv2.rectangle(overlay, (RESET_BTN[0], RESET_BTN[1]),
                  (RESET_BTN[0] + RESET_BTN[2], RESET_BTN[1] + RESET_BTN[3]),
                  (50, 50, 200), -1)
    cv2.putText(overlay, "RESET", (RESET_BTN[0] + 15, RESET_BTN[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # APPLY
    apply_enabled = reference_line_start and reference_line_end
    apply_color = (0, 200, 0) if apply_enabled else (100, 100, 100)
    cv2.rectangle(overlay, (APPLY_BTN[0], APPLY_BTN[1]),
                  (APPLY_BTN[0] + APPLY_BTN[2], APPLY_BTN[1] + APPLY_BTN[3]),
                  apply_color, -1)
    cv2.putText(overlay, "APPLY", (APPLY_BTN[0] + 15, APPLY_BTN[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # CANCEL
    cv2.rectangle(overlay, (CANCEL_BTN[0], CANCEL_BTN[1]),
                  (CANCEL_BTN[0] + CANCEL_BTN[2], CANCEL_BTN[1] + CANCEL_BTN[3]),
                  (100, 100, 100), -1)
    cv2.putText(overlay, "CANCEL", (CANCEL_BTN[0] + 10, CANCEL_BTN[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Current calibration value
    cv2.putText(overlay, f"Current: {PIXELS_PER_MM:.2f} px/mm",
                (RULER_PANEL_X + 20, RULER_PANEL_Y + 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2)

    # Show calculated value if line is drawn
    if reference_line_start and reference_line_end:
        dx = reference_line_end[0] - reference_line_start[0]
        dy = reference_line_end[1] - reference_line_start[1]
        pixel_distance = math.sqrt(dx ** 2 + dy ** 2)
        new_px_per_mm = pixel_distance / REFERENCE_LENGTH_MM if pixel_distance > 0 else 0
        cv2.putText(overlay, f"New: {new_px_per_mm:.2f} px/mm",
                    (RULER_PANEL_X + 200, RULER_PANEL_Y + 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)

    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    # Draw reference line with measurement markers
    if reference_line_start and reference_line_end:
        # Main line - thinner for precision
        cv2.line(frame, reference_line_start, reference_line_end, (0, 255, 255), 2)

        # End circles
        cv2.circle(frame, reference_line_start, 6, (0, 255, 0), -1)
        cv2.circle(frame, reference_line_start, 8, (255, 255, 255), 2)
        cv2.circle(frame, reference_line_end, 6, (0, 255, 0), -1)
        cv2.circle(frame, reference_line_end, 8, (255, 255, 255), 2)

        # Calculate line properties
        dx = reference_line_end[0] - reference_line_start[0]
        dy = reference_line_end[1] - reference_line_start[1]
        length = math.sqrt(dx ** 2 + dy ** 2)
        angle = math.atan2(dy, dx)

        # Draw mm and cm markers along the 76.2mm line
        if length > 0:
            # Draw mm markers (every 1mm)
            for i in range(int(REFERENCE_LENGTH_MM) + 1):  # 0 to 76 mm
                t = i / REFERENCE_LENGTH_MM  # Position along line
                marker_x = int(reference_line_start[0] + dx * t)
                marker_y = int(reference_line_start[1] + dy * t)

                # Perpendicular offset for tick marks
                # Larger ticks for cm, smaller for mm
                is_cm = (i % 10 == 0)
                tick_length = 10 if is_cm else 4
                tick_thickness = 1  # All lines thin for precision

                perp_dx = int(tick_length * math.sin(angle))
                perp_dy = int(-tick_length * math.cos(angle))

                # Draw tick mark
                tick_start = (marker_x - perp_dx, marker_y - perp_dy)
                tick_end = (marker_x + perp_dx, marker_y + perp_dy)
                tick_color = (255, 255, 255) if is_cm else (180, 180, 180)
                cv2.line(frame, tick_start, tick_end, tick_color, tick_thickness)

                # Draw cm numbers (every 10mm)
                if is_cm:
                    cm_num = i // 10
                    text_offset_x = int(18 * math.sin(angle))
                    text_offset_y = int(-18 * math.cos(angle))
                    cv2.putText(frame, f"{cm_num}",
                                (marker_x + text_offset_x - 5, marker_y + text_offset_y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Display pixel length below midpoint
        mid_x = (reference_line_start[0] + reference_line_end[0]) // 2
        mid_y = (reference_line_start[1] + reference_line_end[1]) // 2

        # Offset text perpendicular to line
        text_offset_x = int(-20 * math.sin(angle))
        text_offset_y = int(20 * math.cos(angle))

        cv2.putText(frame, f"{length:.1f} px",
                    (mid_x + text_offset_x, mid_y + text_offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


# ----------------------------------------------------------------------
# Main Overlay
# ----------------------------------------------------------------------
def draw_overlay(frame, pellets):
    # Status bar
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
        ellipse = p.get('ellipse')

        color = (0, 255, 0) if p['within_tolerance'] else (0, 0, 255)

        # Draw ellipse if available (more accurate representation)
        if ellipse:
            cv2.ellipse(frame, ellipse, color, 2)
        else:
            # Fall back to box
            cv2.drawContours(frame, [box], 0, color, 2)

        # Draw center point
        cv2.circle(frame, (int(center[0]), int(center[1])), 3, color, -1)

        top_y = int(min(box[:, 1]))
        left_x = int(min(box[:, 0]))

        bg_y = max(top_y - 30, 0)
        cv2.rectangle(frame, (left_x, bg_y), (left_x + 95, top_y - 5), (0, 0, 0), -1)

        cv2.putText(frame, f"D: {p['diameter']:.2f}mm", (left_x + 3, bg_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(frame, f"L: {p['length']:.2f}mm", (left_x + 3, bg_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        if not p['within_tolerance']:
            top_right = box[np.argmax(box[:, 0])]
            cv2.circle(frame, tuple(top_right), 8, (0, 0, 255), -1)
            cv2.putText(frame, "!", (top_right[0] - 4, top_right[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Calibration hint
    if not in_ruler_calib_mode:
        cv2.putText(frame, "Press 'r' for ruler calibration | 'q' to quit",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)

    # Draw ruler calibration panel
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
    return cap


# ----------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------
def main():
    global in_ruler_calib_mode, calibration_frozen_frame

    print("\nPellet Inspector with Ruler Calibration")
    print("=" * 55)
    print("Features:")
    print("  - Visual 3-inch reference line calibration")
    print("  - Drag to match your physical ruler")
    print("  - CM markers for easy alignment")
    print("=" * 55)
    print("Press 'r' → Enter ruler calibration mode")
    print("Press 'q' → Quit")
    print("=" * 55)

    cap = get_camera()
    if not cap.isOpened():
        print("Cannot open camera.")
        sys.exit(1)

    fps_counter = 0
    fps_start = time.time()
    fps_display = 0

    window_name = "Pellet Inspector - Ruler Calibration"
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

        # Freeze frame when entering calibration mode
        if in_ruler_calib_mode and calibration_frozen_frame is None:
            calibration_frozen_frame = frame.copy()
        elif not in_ruler_calib_mode:
            calibration_frozen_frame = None

        # Use frozen frame during calibration
        display_frame = calibration_frozen_frame.copy() if in_ruler_calib_mode else frame.copy()

        if not in_ruler_calib_mode:
            pellets = detect_pellets(display_frame)
            display_frame = draw_overlay(display_frame, pellets)
        else:
            display_frame = draw_overlay(display_frame, [])

        # FPS
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

        # Key handling
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