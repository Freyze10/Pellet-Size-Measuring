import cv2
import numpy as np
import time
import sys
import math
import logging
from collections import deque
from datetime import datetime

# ----------------------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pellet_inspector_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Global Calibration
# ----------------------------------------------------------------------
PIXELS_PER_MM = 6.0
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0

# Performance Settings
FRAME_BUFFER_SIZE = 2  # Limit buffer to prevent memory buildup
MEASUREMENT_HISTORY_SIZE = 100  # For temporal averaging
RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 2.0


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

RULER_PANEL_X, RULER_PANEL_Y = 10, 80
RULER_PANEL_W, RULER_PANEL_H = 380, 280

RESET_BTN = (RULER_PANEL_X + 20, RULER_PANEL_Y + 200, 100, 40)
APPLY_BTN = (RULER_PANEL_X + 140, RULER_PANEL_Y + 200, 100, 40)
CANCEL_BTN = (RULER_PANEL_X + 260, RULER_PANEL_Y + 200, 100, 40)


# ----------------------------------------------------------------------
# Performance Monitoring
# ----------------------------------------------------------------------
class PerformanceMonitor:
    def __init__(self, window_size=30):
        self.fps_history = deque(maxlen=window_size)
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()
        self.total_frames = 0
        self.start_time = time.time()

    def update(self):
        current_time = time.time()
        delta = current_time - self.last_time
        if delta > 0:
            fps = 1.0 / delta
            self.fps_history.append(fps)
            self.frame_times.append(delta)
        self.last_time = current_time
        self.total_frames += 1

    def get_fps(self):
        if len(self.fps_history) > 0:
            return int(sum(self.fps_history) / len(self.fps_history))
        return 0

    def get_avg_frame_time(self):
        if len(self.frame_times) > 0:
            return sum(self.frame_times) / len(self.frame_times)
        return 0

    def get_runtime(self):
        return time.time() - self.start_time

    def log_stats(self):
        runtime = self.get_runtime()
        avg_fps = self.total_frames / runtime if runtime > 0 else 0
        logger.info(f"Runtime: {runtime / 3600:.2f}h | Frames: {self.total_frames} | Avg FPS: {avg_fps:.1f}")


# ----------------------------------------------------------------------
# Measurement History for Temporal Smoothing
# ----------------------------------------------------------------------
class MeasurementTracker:
    def __init__(self, max_size=100):
        self.history = deque(maxlen=max_size)

    def add_measurement(self, diameter, length):
        self.history.append({'diameter': diameter, 'length': length, 'time': time.time()})

    def get_smoothed_measurement(self, diameter, length, window=5):
        """Apply temporal smoothing to reduce jitter"""
        if len(self.history) < 2:
            return diameter, length

        # Get recent similar measurements
        recent = list(self.history)[-window:]
        similar = [m for m in recent if abs(m['diameter'] - diameter) < 0.3 and abs(m['length'] - length) < 0.3]

        if len(similar) >= 2:
            avg_d = sum(m['diameter'] for m in similar) / len(similar)
            avg_l = sum(m['length'] for m in similar) / len(similar)
            # Blend with current measurement (70% history, 30% current)
            return avg_d * 0.7 + diameter * 0.3, avg_l * 0.7 + length * 0.3

        return diameter, length


measurement_tracker = MeasurementTracker()
perf_monitor = PerformanceMonitor()


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
# Mouse Callback
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
                pixel_distance = math.sqrt(dx ** 2 + dy ** 2)

                if pixel_distance > 10:
                    PIXELS_PER_MM = pixel_distance / REFERENCE_LENGTH_MM
                    update_ranges()
                    logger.info(f"Calibrated: {PIXELS_PER_MM:.4f} px/mm")

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
# Enhanced Detection with Preprocessing Options
# ----------------------------------------------------------------------
def detect_pellets(frame, use_clahe=True):
    """Enhanced detection with optional CLAHE for better lighting handling"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for better contrast in variable lighting
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Bilateral filter for edge-preserving smoothing
    blur = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pellets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA):
            continue

        # Shape filtering: check circularity/solidity
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter ** 2)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Filter out non-pellet shapes (too irregular)
        if circularity < 0.3 or solidity < 0.7:
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
            # Apply temporal smoothing
            diameter_smooth, length_smooth = measurement_tracker.get_smoothed_measurement(diameter, length)
            measurement_tracker.add_measurement(diameter, length)

            pellets.append({
                'contour': cnt,
                'box': box,
                'center': (center_x, center_y),
                'angle': angle,
                'width_px': width_px,
                'height_px': height_px,
                'diameter': diameter_smooth,
                'length': length_smooth,
                'within_tolerance': is_within_tolerance(diameter_smooth, length_smooth),
                'circularity': circularity,
                'solidity': solidity
            })

    return pellets


# ----------------------------------------------------------------------
# Drawing Functions
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
        pixel_distance = math.sqrt(dx ** 2 + dy ** 2)
        new_px_per_mm = pixel_distance / REFERENCE_LENGTH_MM if pixel_distance > 0 else 0
        cv2.putText(overlay, f"New: {new_px_per_mm:.2f} px/mm",
                    (RULER_PANEL_X + 200, RULER_PANEL_Y + 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)

    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    if reference_line_start and reference_line_end:
        cv2.line(frame, reference_line_start, reference_line_end, (0, 255, 255), 2)

        # Crosshairs
        cv2.line(frame, (reference_line_start[0] - 10, reference_line_start[1]),
                 (reference_line_start[0] + 10, reference_line_start[1]), (0, 0, 255), 2)
        cv2.line(frame, (reference_line_start[0], reference_line_start[1] - 10),
                 (reference_line_start[0], reference_line_start[1] + 10), (0, 0, 255), 2)

        cv2.line(frame, (reference_line_end[0] - 10, reference_line_end[1]),
                 (reference_line_end[0] + 10, reference_line_end[1]), (0, 0, 255), 2)
        cv2.line(frame, (reference_line_end[0], reference_line_end[1] - 10),
                 (reference_line_end[0], reference_line_end[1] + 10), (0, 0, 255), 2)


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

        cv2.drawContours(frame, [box], 0, color, 2)
        cv2.circle(frame, (int(center[0]), int(center[1])), 3, color, -1)

        top_y = int(min(box[:, 1]))
        left_x = int(min(box[:, 0]))
        bg_y = max(top_y - 35, 0)

        cv2.rectangle(frame, (left_x, bg_y), (left_x + 85, top_y - 5), (0, 0, 0), -1)
        cv2.putText(frame, f"D: {p['diameter']:.2f}mm", (left_x + 3, bg_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"L: {p['length']:.2f}mm", (left_x + 3, bg_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if not p['within_tolerance']:
            top_right = box[np.argmax(box[:, 0])]
            cv2.circle(frame, tuple(top_right), 10, (0, 0, 255), -1)
            cv2.putText(frame, "!", (top_right[0] - 5, top_right[1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if not in_ruler_calib_mode:
        cv2.putText(frame, "Press 'r' = calibrate | 's' = stats | 'q' = quit",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 2)

    if in_ruler_calib_mode:
        draw_ruler_calibration_mode(frame)

    return frame


# ----------------------------------------------------------------------
# Enhanced Camera with Reconnection Logic
# ----------------------------------------------------------------------
def get_camera(attempt=0):
    """Get camera with retry logic and proper error handling"""
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            raise Exception("Failed to open camera")

        # Set properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, FRAME_BUFFER_SIZE)  # Limit buffer

        # Verify camera is actually working
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            cap.release()
            raise Exception("Camera opened but cannot read frames")

        logger.info(f"Camera initialized successfully (attempt {attempt + 1})")
        return cap

    except Exception as e:
        logger.error(f"Camera initialization failed (attempt {attempt + 1}): {e}")
        if attempt < RECONNECT_ATTEMPTS - 1:
            logger.info(f"Retrying in {RECONNECT_DELAY} seconds...")
            time.sleep(RECONNECT_DELAY)
            return get_camera(attempt + 1)
        else:
            logger.critical("Failed to initialize camera after all attempts")
            return None


# ----------------------------------------------------------------------
# Main Loop with Enhanced Error Handling
# ----------------------------------------------------------------------
def main():
    global in_ruler_calib_mode, calibration_frozen_frame

    logger.info("=" * 60)
    logger.info("Pellet Inspector - Long-Running Edition")
    logger.info("=" * 60)
    logger.info("Press 'r' -> Calibrate (Drag 3-inch line)")
    logger.info("Press 's' -> Show statistics")
    logger.info("Press 'q' -> Quit")
    logger.info("=" * 60)

    cap = get_camera()
    if cap is None:
        logger.critical("Cannot start without camera. Exiting.")
        sys.exit(1)

    window_name = "Pellet Inspector - Long Run"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    consecutive_read_failures = 0
    max_read_failures = 30
    stats_log_interval = 300  # Log stats every 5 minutes
    last_stats_log = time.time()

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                consecutive_read_failures += 1
                logger.warning(f"Frame read failed ({consecutive_read_failures}/{max_read_failures})")

                if consecutive_read_failures >= max_read_failures:
                    logger.error("Too many consecutive read failures. Attempting reconnection...")
                    cap.release()
                    time.sleep(1)
                    cap = get_camera()

                    if cap is None:
                        logger.critical("Reconnection failed. Exiting.")
                        break

                    consecutive_read_failures = 0
                continue

            # Reset failure counter on successful read
            consecutive_read_failures = 0

            # Handle calibration mode frame freezing
            if in_ruler_calib_mode and calibration_frozen_frame is None:
                calibration_frozen_frame = frame.copy()
            elif not in_ruler_calib_mode:
                calibration_frozen_frame = None

            display_frame = calibration_frozen_frame.copy() if in_ruler_calib_mode else frame.copy()

            # Process pellets only when not calibrating
            if not in_ruler_calib_mode:
                try:
                    pellets = detect_pellets(display_frame)
                    display_frame = draw_overlay(display_frame, pellets)
                except Exception as e:
                    logger.error(f"Detection error: {e}")
                    pellets = []
            else:
                display_frame = draw_overlay(display_frame, [])

            # Update performance monitoring
            perf_monitor.update()
            fps = perf_monitor.get_fps()

            # Display FPS and runtime
            runtime_hours = perf_monitor.get_runtime() / 3600
            cv2.putText(display_frame, f"FPS: {fps} | Runtime: {runtime_hours:.1f}h",
                        (display_frame.shape[1] - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Show window
            cv2.imshow(window_name, display_frame)

            # Periodic stats logging
            if time.time() - last_stats_log > stats_log_interval:
                perf_monitor.log_stats()
                last_stats_log = time.time()

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit requested by user")
                break
            elif key == ord('r') and not in_ruler_calib_mode:
                in_ruler_calib_mode = True
                logger.info("Entering calibration mode")
            elif key == ord('s'):
                perf_monitor.log_stats()

            # Check if window was closed
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Window closed by user")
                    break
            except cv2.error:
                logger.warning("Window property check failed, continuing...")

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.critical(f"Unexpected error in main loop: {e}", exc_info=True)
    finally:
        # Cleanup
        logger.info("Shutting down...")
        perf_monitor.log_stats()

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()