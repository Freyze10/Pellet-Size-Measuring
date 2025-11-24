import cv2
import numpy as np
import time
import math

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
# Start with a rough guess (will be fixed by your ruler calibration)
PIXELS_PER_MM = 12.0

# Target Dimensions (mm)
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5

# Filter Logic (Ignore dust or huge blobs)
MIN_AREA_MM = 2.0  # Ignore anything smaller than 2mm²
MAX_AREA_MM = 50.0  # Ignore anything bigger than 50mm²

# ----------------------------------------------------------------------
# STATE VARIABLES
# ----------------------------------------------------------------------
in_ruler_calib_mode = False
show_debug_mask = False
calibration_frozen_frame = None

# Ruler Calibration Variables
REFERENCE_LENGTH_MM = 76.2  # 3 inches
calib_start_pt = None
calib_end_pt = None
is_dragging = False

# UI Positioning
UI_X, UI_Y = 10, 80
UI_W, UI_H = 380, 260


# ----------------------------------------------------------------------
# MOUSE HANDLING (Ruler Calibration)
# ----------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global calib_start_pt, calib_end_pt, is_dragging, in_ruler_calib_mode, PIXELS_PER_MM

    if not in_ruler_calib_mode: return

    # Button Areas
    btn_reset = (UI_X + 20, UI_Y + 180, 100, 40)
    btn_apply = (UI_X + 140, UI_Y + 180, 100, 40)
    btn_cancel = (UI_X + 260, UI_Y + 180, 100, 40)

    def is_inside(px, py, rect):
        return rect[0] <= px <= rect[0] + rect[2] and rect[1] <= py <= rect[1] + rect[3]

    if event == cv2.EVENT_LBUTTONDOWN:
        if is_inside(x, y, btn_reset):
            calib_start_pt = None
            calib_end_pt = None
        elif is_inside(x, y, btn_apply):
            if calib_start_pt and calib_end_pt:
                dist_px = math.hypot(calib_end_pt[0] - calib_start_pt[0], calib_end_pt[1] - calib_start_pt[1])
                if dist_px > 10:
                    PIXELS_PER_MM = dist_px / REFERENCE_LENGTH_MM
                    print(f"CALIBRATION SAVED: {PIXELS_PER_MM:.2f} px/mm")
                in_ruler_calib_mode = False
                calib_start_pt = None
        elif is_inside(x, y, btn_cancel):
            in_ruler_calib_mode = False
            calib_start_pt = None
        elif not is_inside(x, y, (UI_X, UI_Y, UI_W, UI_H)):
            calib_start_pt = (x, y)
            calib_end_pt = (x, y)
            is_dragging = True

    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        calib_end_pt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging = False


# ----------------------------------------------------------------------
# CORE DETECTION ENGINE
# ----------------------------------------------------------------------
def detect_pellets(frame):
    # 1. Gray & Blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # A light blur removes digital noise without destroying edges
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Adaptive Threshold (The "First Code" method)
    # This handles shadows/gradients better than Otsu.
    # Block Size 11, C=2 is a standard robust setting.
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 3. Morphological Cleanup (Crucial for Accuracy)
    kernel = np.ones((3, 3), np.uint8)

    # Close: Fills small black holes inside the white pellet
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Open: Removes small white noise in the background
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Show what the computer sees if requested
    if show_debug_mask:
        cv2.imshow("DEBUG: Mask", thresh)

    # 4. Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        # Filter by rough pixel area to avoid processing noise
        area_px = cv2.contourArea(cnt)
        if area_px < (MIN_AREA_MM * PIXELS_PER_MM): continue

        # 5. Precision Geometry
        # minAreaRect fits the tightest possible rotated rectangle
        rect = cv2.minAreaRect(cnt)
        (center_x, center_y), (w_px, h_px), angle = rect

        # Convert to mm
        dim1 = w_px / PIXELS_PER_MM
        dim2 = h_px / PIXELS_PER_MM

        # Sort dimensions (Diameter is small side, Length is long side)
        diameter = min(dim1, dim2)
        length = max(dim1, dim2)

        # 6. Tolerance Check
        is_good_dia = (TARGET_DIAMETER - TOLERANCE) <= diameter <= (TARGET_DIAMETER + TOLERANCE)
        is_good_len = (TARGET_LENGTH - TOLERANCE) <= length <= (TARGET_LENGTH + TOLERANCE)

        # Double check area filter in mm to be safe
        if (diameter * length) > MAX_AREA_MM: continue

        box = cv2.boxPoints(rect)
        box = np.intp(box)

        results.append({
            'box': box,
            'center': (int(center_x), int(center_y)),
            'diameter': diameter,
            'length': length,
            'ok': is_good_dia and is_good_len
        })

    return results


# ----------------------------------------------------------------------
# DRAWING UTILS
# ----------------------------------------------------------------------
def draw_ui(frame, pellets):
    display = frame.copy()

    if in_ruler_calib_mode:
        # Dim background
        cv2.addWeighted(display, 0.5, np.zeros_like(display), 0.5, 0, display)

        # Panel
        cv2.rectangle(display, (UI_X, UI_Y), (UI_X + UI_W, UI_Y + UI_H), (50, 50, 50), -1)
        cv2.rectangle(display, (UI_X, UI_Y), (UI_X + UI_W, UI_Y + UI_H), (255, 255, 255), 2)

        cv2.putText(display, "RULER CALIBRATION", (UI_X + 80, UI_Y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        instr = ["1. Hold ruler in camera view",
                 "2. Drag line to measure 3 INCHES",
                 "3. Click APPLY"]
        for i, txt in enumerate(instr):
            cv2.putText(display, txt, (UI_X + 20, UI_Y + 80 + (i * 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw Buttons
        btns = [("RESET", 20, (0, 0, 150)), ("APPLY", 140, (0, 150, 0)), ("CANCEL", 260, (100, 100, 100))]
        for txt, x_off, col in btns:
            cv2.rectangle(display, (UI_X + x_off, UI_Y + 180), (UI_X + x_off + 100, UI_Y + 220), col, -1)
            cv2.putText(display, txt, (UI_X + x_off + 15, UI_Y + 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        2)

        # Draw the line if dragging
        if calib_start_pt and calib_end_pt:
            cv2.line(display, calib_start_pt, calib_end_pt, (0, 255, 255), 2)
            cv2.circle(display, calib_start_pt, 5, (0, 255, 0), -1)
            cv2.circle(display, calib_end_pt, 5, (0, 255, 0), -1)

            # Live px calculation
            px_dist = math.hypot(calib_end_pt[0] - calib_start_pt[0], calib_end_pt[1] - calib_start_pt[1])
            mid_x = (calib_start_pt[0] + calib_end_pt[0]) // 2
            cv2.putText(display, f"{px_dist:.1f} px", (mid_x, calib_start_pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    else:
        # NORMAL MODE OVERLAY
        # Top Status Bar
        cv2.rectangle(display, (0, 0), (640, 40), (0, 0, 0), -1)
        good = sum(1 for p in pellets if p['ok'])
        bad = len(pellets) - good
        status_col = (0, 255, 0) if (bad == 0 and good > 0) else (0, 0, 255)

        cv2.putText(display, f"TOTAL: {len(pellets)}  |  OK: {good}  |  BAD: {bad}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_col, 2)

        cv2.putText(display, f"CAL: {PIXELS_PER_MM:.2f} px/mm",
                    (440, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Bottom Help
        cv2.putText(display, "[R] Calibrate Ruler  |  [D] Debug Mask  |  [Q] Quit",
                    (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw Pellets
        for p in pellets:
            color = (0, 255, 0) if p['ok'] else (0, 0, 255)
            cv2.drawContours(display, [p['box']], 0, color, 2)

            # Label Background
            x, y = p['box'][1]  # Top-right corner roughly
            cv2.rectangle(display, (x, y - 35), (x + 80, y), (0, 0, 0), -1)

            cv2.putText(display, f"D:{p['diameter']:.2f}", (x + 2, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(display, f"L:{p['length']:.2f}", (x + 2, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return display


# ----------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------
def main():
    global in_ruler_calib_mode, calibration_frozen_frame, show_debug_mask

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Force resolution for consistency
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("Inspector")
    cv2.setMouseCallback("Inspector", mouse_callback)

    print("System Ready. Press 'r' to calibrate.")

    while True:
        ret, frame = cap.read()
        if not ret: continue

        # If in calibration mode, freeze the frame so user can draw line comfortably
        if in_ruler_calib_mode:
            if calibration_frozen_frame is None:
                calibration_frozen_frame = frame.copy()
            display_frame = draw_ui(calibration_frozen_frame, [])
        else:
            calibration_frozen_frame = None
            pellets = detect_pellets(frame)
            display_frame = draw_ui(frame, pellets)

        cv2.imshow("Inspector", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            in_ruler_calib_mode = not in_ruler_calib_mode
            # Reset drag state on toggle
            calib_start_pt = None
        elif key == ord('d'):
            show_debug_mask = not show_debug_mask
            if not show_debug_mask: cv2.destroyWindow("DEBUG: Mask")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()