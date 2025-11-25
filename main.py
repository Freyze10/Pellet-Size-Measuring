import cv2
import numpy as np
import time
import sys
import math

# ----------------------------------------------------------------------
# Camera Configuration
# ----------------------------------------------------------------------
# Change these to your preferred resolution (e.g., 1920x1080, 1280x720, 640x480)
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720

# ----------------------------------------------------------------------
# Global Calibration
# ----------------------------------------------------------------------
# NOTE: If you change resolution, you MUST recalibrate (Press 'r')
PIXELS_PER_MM = 12.0  # This will need to change based on resolution!
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
MAX_CONTOUR_AREA = 50000  # Increased for higher resolutions

# ----------------------------------------------------------------------
# Ruler Calibration State
# ----------------------------------------------------------------------
in_ruler_calib_mode = False
REFERENCE_LENGTH_MM = 76.2  # 3 inches
reference_line_start = None
reference_line_end = None
calibration_frozen_frame = None
is_dragging = False

# UI Layout (Will be initialized in main)
RULER_PANEL_X, RULER_PANEL_Y = 10, 80
RULER_PANEL_W, RULER_PANEL_H = 380, 280
RESET_BTN = []
APPLY_BTN = []
CANCEL_BTN = []


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
# Mouse Callback
# ----------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global reference_line_start, reference_line_end, is_dragging
    global in_ruler_calib_mode, PIXELS_PER_MM

    if not in_ruler_calib_mode:
        return

    def in_rect(px, py, rect):
        if not rect: return False
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
                    print(f"Calibrated: {PIXELS_PER_MM:.4f} px/mm based on {pixel_distance:.2f}px")

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

        # If not on panel, start drawing line
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
# Detection Logic
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
            width_px, height_px = edge1, edge2
        else:
            width_px, height_px = edge2, edge1

        width_mm = width_px / PIXELS_PER_MM
        height_mm = height_px / PIXELS_PER_MM

        diameter = width_mm
        length = height_mm

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
# Drawing Functions
# ----------------------------------------------------------------------
def draw_ruler_calibration_mode(frame):
    overlay = frame.copy()

    # Draw background panel
    cv2.rectangle(overlay, (RULER_PANEL_X, RULER_PANEL_Y),
                  (RULER_PANEL_X + RULER_PANEL_W, RULER_PANEL_Y + RULER_PANEL_H),
                  (30, 30, 50), -1)
    cv2.rectangle(overlay, (RULER_PANEL_X, RULER_PANEL_Y),
                  (RULER_PANEL_X + RULER_PANEL_W, RULER_PANEL_Y + RULER_PANEL_H),
                  (100, 150, 255), 3)

    cv2.putText(overlay, "RULER CALIBRATION", (RULER_PANEL_X + 80, RULER_PANEL_Y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Instructions
    instructions = [
        "1. Place a ruler in view",
        "2. Drag line to match 3 inches",
        "3. Click APPLY"
    ]
    y_off = RULER_PANEL_Y + 70
    for ins in instructions:
        cv2.putText(overlay, ins, (RULER_PANEL_X + 20, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_off += 25

    # Buttons
    def draw_btn(btn_rect, text, color=(50, 50, 200)):
        rx, ry, rw, rh = btn_rect
        cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), color, -1)
        cv2.putText(overlay, text, (rx + 15, ry + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    draw_btn(RESET_BTN, "RESET")

    apply_active = reference_line_start and reference_line_end
    apply_col = (0, 200, 0) if apply_active else (100, 100, 100)
    draw_btn(APPLY_BTN, "APPLY", apply_col)

    draw_btn(CANCEL_BTN, "CANCEL", (100, 100, 100))

    cv2.putText(overlay, f"Current: {PIXELS_PER_MM:.2f} px/mm",
                (RULER_PANEL_X + 20, RULER_PANEL_Y + 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2)

    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    # Draw User Line
    if reference_line_start and reference_line_end:
        cv2.line(frame, reference_line_start, reference_line_end, (0, 255, 255), 2)

        # Calculate stats
        dx = reference_line_end[0] - reference_line_start[0]
        dy = reference_line_end[1] - reference_line_start[1]
        length_px = math.sqrt(dx ** 2 + dy ** 2)

        mid_x = (reference_line_start[0] + reference_line_end[0]) // 2
        mid_y = (reference_line_start[1] + reference_line_end[1]) // 2

        cv2.putText(frame, f"{length_px:.1f} px", (mid_x, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def draw_overlay(frame, pellets):
    # Status Bar (Top Left)
    total = len(pellets)
    within = sum(1 for p in pellets if p['within_tolerance'])
    out = total - within
    color = (0, 255, 0) if out == 0 else (0, 0, 255)

    cv2.rectangle(frame, (10, 10), (460, 50), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (460, 50), color, 2)
    cv2.putText(frame, f"In: {within}  Out: {out}  Total: {total}", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Draw Pellets
    for p in pellets:
        box = p['box']
        center = p['center']
        c_color = (0, 255, 0) if p['within_tolerance'] else (0, 0, 255)

        cv2.drawContours(frame, [box], 0, c_color, 2)

        # Label background
        top_y = int(min(box[:, 1]))
        left_x = int(min(box[:, 0]))
        cv2.rectangle(frame, (left_x, top_y - 40), (left_x + 80, top_y - 5), (0, 0, 0), -1)

        cv2.putText(frame, f"D:{p['diameter']:.2f}", (left_x + 2, top_y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"L:{p['length']:.2f}", (left_x + 2, top_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if not in_ruler_calib_mode:
        cv2.putText(frame, "Press 'r' to Calibrate | 'q' to Quit",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)
    else:
        draw_ruler_calibration_mode(frame)

    return frame


# ----------------------------------------------------------------------
# Camera Initialization (Dynamic)
# ----------------------------------------------------------------------
def get_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    # Request Desired Resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # Read Actual Resolution
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera initialized: Requested {DESIRED_WIDTH}x{DESIRED_HEIGHT}, Got {int(w)}x{int(h)}")
    return cap


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    global in_ruler_calib_mode, calibration_frozen_frame
    global RESET_BTN, APPLY_BTN, CANCEL_BTN

    # Initialize Buttons based on fixed panel location
    # If you want these to center on 1080p, you'd calculate them based on frame width
    RESET_BTN = (RULER_PANEL_X + 20, RULER_PANEL_Y + 200, 100, 40)
    APPLY_BTN = (RULER_PANEL_X + 140, RULER_PANEL_Y + 200, 100, 40)
    CANCEL_BTN = (RULER_PANEL_X + 260, RULER_PANEL_Y + 200, 100, 40)

    cap = get_camera()
    if not cap.isOpened():
        print("No camera found.")
        return

    cv2.namedWindow("Pellet Inspector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Pellet Inspector", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame lost.")
            break

        display_frame = frame.copy()

        # Check if user entered calibration mode
        if in_ruler_calib_mode:
            if calibration_frozen_frame is None:
                calibration_frozen_frame = frame.copy()
            display_frame = draw_overlay(calibration_frozen_frame.copy(), [])
        else:
            calibration_frozen_frame = None
            pellets = detect_pellets(frame)
            display_frame = draw_overlay(display_frame, pellets)

        cv2.imshow("Pellet Inspector", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'): in_ruler_calib_mode = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()