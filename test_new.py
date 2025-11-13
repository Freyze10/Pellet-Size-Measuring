import cv2
import numpy as np
import time
import sys

# ----------------------------------------------------------------------
# Configuration (unchanged)
# ----------------------------------------------------------------------
PIXEL_TO_MM = 1 / 10
TARGET_DIAMETER = 3.0
TARGET_LENGTH   = 3.0
TOLERANCE       = 0.5
EXCLUSION_THRESHOLD = 1.0

DIAMETER_MIN = TARGET_DIAMETER - TOLERANCE
DIAMETER_MAX = TARGET_DIAMETER + TOLERANCE
LENGTH_MIN   = TARGET_LENGTH - TOLERANCE
LENGTH_MAX   = TARGET_LENGTH + TOLERANCE

DIAMETER_EXCLUDE_MIN = TARGET_DIAMETER - EXCLUSION_THRESHOLD
DIAMETER_EXCLUDE_MAX = TARGET_DIAMETER + EXCLUSION_THRESHOLD
LENGTH_EXCLUDE_MIN   = TARGET_LENGTH - EXCLUSION_THRESHOLD
LENGTH_EXCLUDE_MAX   = TARGET_LENGTH + EXCLUSION_THRESHOLD

MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 10000

# ----------------------------------------------------------------------
# Helper checks (unchanged)
# ----------------------------------------------------------------------
def is_within_tolerance(diameter: float, length: float) -> bool:
    return (DIAMETER_MIN <= diameter <= DIAMETER_MAX and
            LENGTH_MIN   <= length   <= LENGTH_MAX)

def should_process_pellet(diameter: float, length: float) -> bool:
    return (DIAMETER_EXCLUDE_MIN <= diameter <= DIAMETER_EXCLUDE_MAX and
            LENGTH_EXCLUDE_MIN   <= length   <= LENGTH_EXCLUDE_MAX)

# ----------------------------------------------------------------------
# Detection (unchanged)
# ----------------------------------------------------------------------
def detect_pellets(frame):
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    pellets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        width_mm  = w * PIXEL_TO_MM
        height_mm = h * PIXEL_TO_MM
        diameter = min(width_mm, height_mm)
        length   = max(width_mm, height_mm)

        if should_process_pellet(diameter, length):
            pellets.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'diameter': diameter,
                'length':   length,
                'within_tolerance': is_within_tolerance(diameter, length)
            })
    return pellets


# ----------------------------------------------------------------------
# Overlay drawing – UPDATED STATUS BAR
# ----------------------------------------------------------------------
def draw_overlay(frame, pellets):
    # Count within and out of tolerance
    total = len(pellets)
    within = sum(1 for p in pellets if p['within_tolerance'])
    out_of = total - within

    # Status bar text
    status_text = f"Within: {within}   Out: {out_of}   Total: {total}"
    status_color = (0, 255, 0) if out_of == 0 else (0, 0, 255)  # Green if all OK, Red if any bad

    # Draw status bar (larger area to fit text)
    cv2.rectangle(frame, (10, 10), (450, 50), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (450, 50), status_color, 2)
    cv2.putText(frame, status_text, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

    # Per-pellet drawing (unchanged)
    for p in pellets:
        x, y, w, h = p['x'], p['y'], p['w'], p['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        txt_d = f"D: {p['diameter']:.2f} mm"
        txt_l = f"L: {p['length']:.2f} mm"

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

    return frame


# ----------------------------------------------------------------------
# Camera handling (unchanged)
# ----------------------------------------------------------------------
def get_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


# ----------------------------------------------------------------------
# Main loop (unchanged except window name)
# ----------------------------------------------------------------------
def main():
    print("\nPellet Size Measurement System")
    print("=" * 55)
    print(f"Target  D = {TARGET_DIAMETER} mm  (±{TOLERANCE} mm)")
    print(f"Target  L = {TARGET_LENGTH}   mm  (±{TOLERANCE} mm)")
    print(f"Calibration: {1/PIXEL_TO_MM:.1f} px = 1 mm")
    print("=" * 55)
    print("Press 'q' or click X to quit\n")

    cap = get_camera()
    if not cap.isOpened():
        print("Cannot open camera – exiting.")
        sys.exit(1)

    fps_counter = 0
    fps_start   = time.time()
    fps_display = 0

    cv2.namedWindow("Pellet Size Measurement", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost camera feed – reconnecting...")
            cap.release()
            time.sleep(1)
            cap = get_camera()
            if not cap.isOpened():
                print("Reconnect failed – exiting.")
                break
            continue

        pellets = detect_pellets(frame)
        frame   = draw_overlay(frame, pellets)

        # FPS
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_counter // int(elapsed)
            fps_counter = 0
            fps_start   = time.time()

        cv2.putText(frame, f"FPS: {fps_display}", (frame.shape[1] - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Pellets: {len(pellets)}", (frame.shape[1] - 130, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Pellet Size Measurement", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if cv2.getWindowProperty("Pellet Size Measurement", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nSystem shut down gracefully.")


if __name__ == "__main__":
    main()