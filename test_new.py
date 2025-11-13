import cv2
import numpy as np
import time
import sys

# ----------------------------------------------------------------------
# Global Calibration
# ----------------------------------------------------------------------
PIXELS_PER_MM = 6.0          # Default – will be changed with arrows
TARGET_DIAMETER = 3.0
TARGET_LENGTH   = 3.0
TOLERANCE       = 0.5
EXCLUSION_THRESHOLD = 1.0

def update_ranges():
    global DIAMETER_MIN, DIAMETER_MAX, LENGTH_MIN, LENGTH_MAX
    global DIAMETER_EXCLUDE_MIN, DIAMETER_EXCLUDE_MAX
    global LENGTH_EXCLUDE_MIN, LENGTH_EXCLUDE_MAX

    DIAMETER_MIN = TARGET_DIAMETER - TOLERANCE
    DIAMETER_MAX = TARGET_DIAMETER + TOLERANCE
    LENGTH_MIN   = TARGET_LENGTH - TOLERANCE
    LENGTH_MAX   = TARGET_LENGTH + TOLERANCE

    DIAMETER_EXCLUDE_MIN = TARGET_DIAMETER - EXCLUSION_THRESHOLD
    DIAMETER_EXCLUDE_MAX = TARGET_DIAMETER + EXCLUSION_THRESHOLD
    LENGTH_EXCLUDE_MIN   = TARGET_LENGTH - EXCLUSION_THRESHOLD
    LENGTH_EXCLUDE_MAX   = TARGET_LENGTH + EXCLUSION_THRESHOLD

update_ranges()

MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 10000

# ----------------------------------------------------------------------
# Calibration Panel Geometry
# ----------------------------------------------------------------------
show_calib_panel = False
CALIB_X, CALIB_Y, CALIB_W, CALIB_H = 10, 350, 220, 120
UP_RECT   = (CALIB_X + 190, CALIB_Y + 20, 25, 25)   # inside panel
DOWN_RECT = (CALIB_X + 190, CALIB_Y + 65, 25, 25)

# ----------------------------------------------------------------------
# Mouse callback – click the arrows
# ----------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global PIXELS_PER_MM, show_calib_panel

    if not show_calib_panel:
        return

    def in_rect(px, py, rect):
        rx, ry, rw, rh = rect
        return rx <= px <= rx + rw and ry <= py <= ry + rh

    if event == cv2.EVENT_LBUTTONDOWN:
        if in_rect(x, y, UP_RECT):
            PIXELS_PER_MM = round(PIXELS_PER_MM + 0.1, 2)
            update_ranges()
        elif in_rect(x, y, DOWN_RECT):
            if PIXELS_PER_MM > 0.5:
                PIXELS_PER_MM = round(PIXELS_PER_MM - 0.1, 2)
                update_ranges()

# ----------------------------------------------------------------------
# Helper checks
# ----------------------------------------------------------------------
def is_within_tolerance(diameter: float, length: float) -> bool:
    return (DIAMETER_MIN <= diameter <= DIAMETER_MAX and
            LENGTH_MIN   <= length   <= LENGTH_MAX)

def should_process_pellet(diameter: float, length: float) -> bool:
    return (DIAMETER_EXCLUDE_MIN <= diameter <= DIAMETER_EXCLUDE_MAX and
            LENGTH_EXCLUDE_MIN   <= length   <= LENGTH_EXCLUDE_MAX)

# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------
def detect_pellets(frame):
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    pellets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        width_mm  = w / PIXELS_PER_MM
        height_mm = h / PIXELS_PER_MM
        diameter = min(width_mm, height_mm)
        length   = max(width_mm, height_mm)

        if should_process_pellet(diameter, length):
            pellets.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'diameter': diameter,
                'length': length,
                'within_tolerance': is_within_tolerance(diameter, length)
            })
    return pellets

# ----------------------------------------------------------------------
# Draw Calibration Panel
# ----------------------------------------------------------------------
def draw_calib_panel(frame):
    x, y, w, h = CALIB_X, CALIB_Y, CALIB_W, CALIB_H
    overlay = frame.copy()

    # Background
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (100, 100, 255), 2)

    # Title
    cv2.putText(overlay, "CALIBRATION", (x + 10, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Current value
    cv2.putText(overlay, f"{PIXELS_PER_MM:.2f} px/mm", (x + 10, y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    # Up arrow
    cv2.arrowedLine(overlay,
                    (x + 202, y + 32), (x + 202, y + 22),
                    (0, 255, 0), 2, tipLength=0.3)
    # Down arrow
    cv2.arrowedLine(overlay,
                    (x + 202, y + 77), (x + 202, y + 87),
                    (255, 0, 0), 2, tipLength=0.3)

    # Instructions
    cv2.putText(overlay, "Click arrows or", (x + 10, y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(overlay, "use Up/Down keys", (x + 10, y + 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

# ----------------------------------------------------------------------
# Overlay
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
        x, y, w, h = p['x'], p['y'], p['w'], p['h']
        color = (0, 255, 0) if p['within_tolerance'] else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        txt_d = f"D: {p['diameter']:.2f}"
        txt_l = f"L: {p['length']:.2f}"

        bg_y = max(y - 45, 0)
        cv2.rectangle(frame, (x, bg_y), (x + 155, y - 5), (0, 0, 0), -1)
        cv2.putText(frame, txt_d, (x + 5, bg_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, txt_l, (x + 5, bg_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        if not p['within_tolerance']:
            cv2.circle(frame, (x + w - 10, y + 10), 8, (0, 0, 255), -1)
            cv2.putText(frame, "!", (x + w - 14, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if show_calib_panel:
        draw_calib_panel(frame)

    return frame

# ----------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------
def get_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    global show_calib_panel, PIXELS_PER_MM

    print("\nPellet Inspector + On-Screen Calibration")
    print("=" * 55)
    print("Press 'c' → Toggle calibration panel")
    print("Click arrows or use Up/Down keys to adjust")
    print("Press 'q' or click X to quit")
    print("=" * 55)

    cap = get_camera()
    if not cap.isOpened():
        print("Cannot open camera.")
        sys.exit(1)

    fps_counter = 0
    fps_start   = time.time()
    fps_display = 0

    window_name = "Pellet Size Measurement"
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

        display_frame = frame.copy()
        pellets = detect_pellets(display_frame)
        display_frame = draw_overlay(display_frame, pellets)

        # FPS
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_counter // int(elapsed)
            fps_counter = 0
            fps_start   = time.time()

        cv2.putText(display_frame, f"FPS: {fps_display}",
                    (display_frame.shape[1] - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, display_frame)

        # ------------------- KEY HANDLING -------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            show_calib_panel = not show_calib_panel
            print(f"Calibration panel: {'ON' if show_calib_panel else 'OFF'}")

        # Arrow keys work only when panel is visible
        if show_calib_panel:
            if key == 82:   # Up arrow
                PIXELS_PER_MM = round(PIXELS_PER_MM + 0.1, 2)
                update_ranges()
            elif key == 84: # Down arrow
                if PIXELS_PER_MM > 0.5:
                    PIXELS_PER_MM = round(PIXELS_PER_MM - 0.1, 2)
                    update_ranges()

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Shutdown complete.")


if __name__ == "__main__":
    main()