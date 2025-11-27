import cv2
import numpy as np
import time
import sys
import math
from ultralytics import YOLO
from scipy.stats import mode  # For robust tick spacing detection

# ----------------------------------------------------------------------
# Global Calibration
# ----------------------------------------------------------------------
PIXELS_PER_MM = 6.0
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0

# Auto-calibration settings (more forgiving for partial rulers)
AUTO_CALIBRATION_ENABLED = True
CALIBRATION_CONFIDENCE_THRESHOLD = 0.25   # Lowered!
CALIBRATION_SAMPLES = []
MAX_CALIBRATION_SAMPLES = 10
STABLE_CALIBRATION_THRESHOLD = 0.12       # Slightly relaxed

# Load YOLO model for ruler detection
try:
    ruler_model = YOLO('yolo/best.pt')
    print("✓ YOLO ruler detection model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load YOLO model: {e}")
    print("   Auto-calibration will be disabled. Place 'best.pt' in the same directory.")
    AUTO_CALIBRATION_ENABLED = False
    ruler_model = None

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

# Manual calibration fallback
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
# IMPROVED: Detect ruler even if partially visible
# ----------------------------------------------------------------------
def detect_ruler_region(frame):
    if not AUTO_CALIBRATION_ENABLED or ruler_model is None:
        return None

    try:
        # Lower confidence + imgsz for speed
        results = ruler_model(frame, verbose=False, conf=0.25, imgsz=640)
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()

        best_region = None
        best_score = 0

        for (x1, y1, x2, y2), conf in zip(boxes, confidences):
            if conf < 0.25:
                continue
            w = x2 - x1
            h = y2 - y1
            if w < 50 or h < 30:  # Too tiny
                continue

            area = w * h
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Score: confidence × size (favors larger visible parts)
            score = conf * (area ** 0.5)

            if score > best_score:
                best_score = score
                label = "Ruler"
                if w < 200:
                    label += " [PARTIAL]"
                best_region = {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(conf),
                    'roi': roi.copy(),
                    'area': area,
                    'label': label
                }

        return best_region

    except Exception as e:
        print(f"YOLO Error: {e}")
        return None

# ----------------------------------------------------------------------
# IMPROVED: Tick detection works on small/cropped rulers
# ----------------------------------------------------------------------
def detect_tick_marks(roi):
    if roi is None or roi.shape[0] < 30 or roi.shape[1] < 50:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 30, 120, apertureSize=3)

    h, w = roi.shape[:2]
    min_length = max(8, h // 20)
    max_gap = max(5, w // 30)
    threshold = max(10, w // 10)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=threshold,
                            minLineLength=min_length,
                            maxLineGap=max_gap)

    if lines is None:
        return []

    tick_marks = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 5:
            continue

        angle = abs(math.degrees(math.atan2(dy, dx)))
        # Vertical ticks (main) or short horizontal (minor)
        if (70 <= angle <= 110) or (length < 20 and (angle < 25 or angle > 155)):
            mid_x = (x1 + x2) // 2
            tick_marks.append({
                'x': mid_x,
                'y1': min(y1, y2),
                'y2': max(y1, y2),
                'length': length,
                'line': (x1, y1, x2, y2)
            })

    tick_marks.sort(key=lambda t: t['x'])
    return tick_marks

# ----------------------------------------------------------------------
# IMPROVED: Works with few ticks & guesses correct mm spacing
# ----------------------------------------------------------------------
def calculate_pixel_per_mm_from_ticks(tick_marks, roi_width):
    if len(tick_marks) < 2:
        return None

    x_positions = [t['x'] for t in tick_marks]
    distances = np.diff(x_positions)
    distances = distances[(distances > 12) & (distances < roi_width * 0.6)]

    if len(distances) == 0:
        return None

    # Use mode (most frequent spacing) → robust!
    try:
        most_common = mode(distances, keepdims=False).mode
        if np.isnan(most_common):
            raise ValueError
    except:
        most_common = np.median(distances)

    # Try 1mm, 5mm, 10mm spacing — pick most plausible
    candidates = []
    for mm_spacing in [1.0, 5.0, 10.0]:
        px_per_mm = most_common / mm_spacing
        if 2.0 <= px_per_mm <= 30.0:
            candidates.append((abs(px_per_mm - PIXELS_PER_MM) if PIXELS_PER_MM > 1 else 0, px_per_mm))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    return most_common / 10.0  # fallback

# ----------------------------------------------------------------------
# Calibration update
# ----------------------------------------------------------------------
def update_calibration_with_sample(new_px_per_mm):
    global PIXELS_PER_MM, CALIBRATION_SAMPLES
    CALIBRATION_SAMPLES.append(new_px_per_mm)
    if len(CALIBRATION_SAMPLES) > MAX_CALIBRATION_SAMPLES:
        CALIBRATION_SAMPLES.pop(0)

    if len(CALIBRATION_SAMPLES) >= 5:
        mean_val = np.mean(CALIBRATION_SAMPLES)
        std_val = np.std(CALIBRATION_SAMPLES)
        if std_val / mean_val < STABLE_CALIBRATION_THRESHOLD:
            old_val = PIXELS_PER_MM
            PIXELS_PER_MM = mean_val
            update_ranges()
            if abs(old_val - mean_val) > 0.15:
                print(f"Auto-calibrated → {PIXELS_PER_MM:.4f} px/mm (±{std_val:.4f})")
            return True
    return False

def auto_calibrate_from_frame(frame):
    ruler_data = detect_ruler_region(frame)
    if ruler_data is None:
        return None, None

    tick_marks = detect_tick_marks(ruler_data['roi'])
    if len(tick_marks) >= 2:
        px_per_mm = calculate_pixel_per_mm_from_ticks(tick_marks, ruler_data['roi'].shape[1])
        if px_per_mm is not None:
            update_calibration_with_sample(px_per_mm)
            print(f"Partial ruler used ({ruler_data['roi'].shape[1]}px wide, {len(tick_marks)} ticks)")
    return ruler_data, tick_marks

# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def draw_ruler_detection(frame, ruler_data, tick_marks):
    if ruler_data is None:
        return
    x1, y1, x2, y2 = ruler_data['bbox']
    label = ruler_data.get('label', f"Ruler {ruler_data['confidence']:.2f}")
    color = (0, 255, 255) if 'PARTIAL' not in label else (0, 255, 100)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - h - 15), (x1 + w + 15, y1), color, -1)
    cv2.putText(frame, label, (x1 + 8, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if tick_marks:
        for t in tick_marks:
            tx1, ty1, tx2, ty2 = t['line']
            cv2.line(frame, (x1 + tx1, y1 + ty1), (x1 + tx2, y1 + ty2),
                     (255, 100, 255), 2)

# ----------------------------------------------------------------------
# Rest of your original functions (unchanged or slightly cleaned)
# ----------------------------------------------------------------------
def is_within_tolerance(diameter: float, length: float) -> bool:
    return (DIAMETER_MIN <= diameter <= DIAMETER_MAX and
            LENGTH_MIN <= length <= LENGTH_MAX)

def should_process_pellet(diameter: float, length: float) -> bool:
    return (DIAMETER_EXCLUDE_MIN <= diameter <= DIAMETER_EXCLUDE_MAX and
            LENGTH_EXCLUDE_MIN <= length <= LENGTH_EXCLUDE_MAX)

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def mouse_callback(event, x, y, flags, param):
    global reference_line_start, reference_line_end, is_dragging, in_ruler_calib_mode, PIXELS_PER_MM
    if not in_ruler_calib_mode:
        return

    def in_rect(px, py, rect):
        rx, ry, rw, rh = rect
        return rx <= px < rx + rw and ry <= py < ry + rh

    if event == cv2.EVENT_LBUTTONDOWN:
        if in_rect(x, y, RESET_BTN):
            reference_line_start = reference_line_end = None
            is_dragging = False
        elif in_rect(x, y, APPLY_BTN):
            if reference_line_start and reference_line_end:
                d = get_distance(reference_line_start, reference_line_end)
                if d > 20:
                    PIXELS_PER_MM = d / REFERENCE_LENGTH_MM
                    update_ranges()
                    print(f"Manual calibration applied: {PIXELS_PER_MM:.3f} px/mm")
                    in_ruler_calib_mode = False
        elif in_rect(x, y, CANCEL_BTN):
            in_ruler_calib_mode = False
        else:
            if not in_rect(x, y, (RULER_PANEL_X, RULER_PANEL_Y, RULER_PANEL_W, RULER_PANEL_H)):
                reference_line_start = (x, y)
                reference_line_end = (x, y)
                is_dragging = True
    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        reference_line_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and is_dragging:
        reference_line_end = (x, y)
        is_dragging = False

def detect_pellets(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
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
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        (cx, cy), (w, h), angle = rect

        edge1 = get_distance(box[0], box[1])
        edge2 = get_distance(box[1], box[2])
        width_px = min(edge1, edge2)
        height_px = max(edge1, edge2)

        diameter = width_px / PIXELS_PER_MM
        length = height_px / PIXELS_PER_MM

        if should_process_pellet(diameter, length):
            pellets.append({
                'contour': cnt,
                'box': box,
                'center': (cx, cy),
                'angle': angle,
                'width_px': width_px,
                'height_px': height_px,
                'diameter': round(diameter, 3),
                'length': round(length, 3),
                'within_tolerance': is_within_tolerance(diameter, length)
            })
    return pellets

def draw_ruler_calibration_mode(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (RULER_PANEL_X, RULER_PANEL_Y),
                  (RULER_PANEL_X + RULER_PANEL_W, RULER_PANEL_Y + RULER_PANEL_H),
                  (30, 30, 50), -1)
    cv2.rectangle(overlay, (RULER_PANEL_X, RULER_PANEL_Y),
                  (RULER_PANEL_X + RULER_PANEL_W, RULER_PANEL_Y + RULER_PANEL_H),
                  (100, 150, 255), 3)
    cv2.putText(overlay, "MANUAL RULER CALIBRATION", (RULER_PANEL_X + 30, RULER_PANEL_Y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    lines = ["Click and drag to draw a line over",
             "exactly 76.2 mm (3 inches) on ruler",
             "", "Then click APPLY"]
    for i, txt in enumerate(lines):
        cv2.putText(overlay, txt, (RULER_PANEL_X + 20, RULER_PANEL_Y + 70 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    cv2.putText(overlay, f"Current: {PIXELS_PER_MM:.3f} px/mm", (RULER_PANEL_X + 20, RULER_PANEL_Y + 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    # Buttons
    for btn, text, color in [(RESET_BTN, "RESET", (50,50,200)), (APPLY_BTN, "APPLY", (0,180,0) if reference_line_start else (80,80,80)), (CANCEL_BTN, "CANCEL", (100,100,100))]:
        cv2.rectangle(overlay, (btn[0], btn[1]), (btn[0]+btn[2], btn[1]+btn[3]), color, -1)
        cv2.putText(overlay, text, (btn[0]+15, btn[1]+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    if reference_line_start and reference_line_end:
        cv2.line(frame, reference_line_start, reference_line_end, (0, 255, 255), 3)
        cv2.circle(frame, reference_line_start, 6, (0,0,255), -1)
        cv2.circle(frame, reference_line_end, 6, (0,0,255), -1)

def draw_overlay(frame, pellets, ruler_data=None, tick_marks=None):
    h, w = frame.shape[:2]
    total = len(pellets)
    good = sum(1 for p in pellets if p['within_tolerance'])
    bad = total - good
    status = f"In: {good}  Out: {bad}  Total: {total}"
    color = (0, 255, 0) if bad == 0 else (0, 0, 255)
    cv2.rectangle(frame, (10, 10), (500, 55), (0, 0, 0), -1)
    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    calib_text = f"Scale: {PIXELS_PER_MM:.3f} px/mm"
    calib_text += " [AUTO]" if AUTO_CALIBRATION_ENABLED and ruler_data else " [FIXED]"
    cv2.putText(frame, calib_text, (510, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    if AUTO_CALIBRATION_ENABLED and ruler_data:
        draw_ruler_detection(frame, ruler_data, tick_marks)

    for p in pellets:
        color = (0, 255, 0) if p['within_tolerance'] else (0, 0, 255)
        cv2.drawContours(frame, [p['box']], 0, color, 2)
        cv2.circle(frame, (int(p['center'][0]), int(p['center'][1])), 4, color, -1)

        x = min(p['box'][:, 0]) - 5
        y = min(p['box'][:, 1]) - 10
        cv2.putText(frame, f"D:{p['diameter']} L:{p['length']}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    help_text = "Auto-calibration ACTIVE | 'r' = manual calib | 'q' = quit"
    cv2.putText(frame, help_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1)

    if in_ruler_calib_mode:
        draw_ruler_calibration_mode(frame)

    return frame

def get_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"Camera opened: {int(cap.get(3))}x{int(cap.get(4))}")
    return cap

# ----------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------
def main():
    global in_ruler_calib_mode, calibration_frozen_frame
    print("\n" + "="*65)
    print("   PELLET INSPECTOR – PARTIAL RULER AUTO-CALIBRATION READY")
    print("="*65)
    print("   • Works with 20–100% of ruler visible!")
    print("   • Robust tick detection + smart spacing guess")
    print("   • Press 'r' for manual fallback, 'q' to quit")
    print("="*65 + "\n")

    cap = get_camera()
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    cv2.namedWindow("Pellet Inspector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Pellet Inspector", mouse_callback)

    fps_counter = 0
    fps_start = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera disconnected")
            break

        display_frame = frame.copy()
        if in_ruler_calib_mode:
            if calibration_frozen_frame is None:
                calibration_frozen_frame = frame.copy()
            display_frame = calibration_frozen_frame.copy()

        ruler_data = None
        tick_marks = None
        if not in_ruler_calib_mode and AUTO_CALIBRATION_ENABLED and frame_count % 4 == 0:
            ruler_data, tick_marks = auto_calibrate_from_frame(frame)

        if not in_ruler_calib_mode:
            pellets = detect_pellets(display_frame)
            display_frame = draw_overlay(display_frame, pellets, ruler_data, tick_marks)
        else:
            display_frame = draw_overlay(display_frame, [])

        # FPS
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            cv2.putText(display_frame, f"FPS: {fps_counter}", (display_frame.shape[1]-140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            fps_counter = 0
            fps_start = time.time()

        cv2.imshow("Pellet Inspector", display_frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            in_ruler_calib_mode = not in_ruler_calib_mode
            if not in_ruler_calib_mode:
                calibration_frozen_frame = None
                reference_line_start = reference_line_end = None

    cap.release()
    cv2.destroyAllWindows()
    print("Shutdown complete.")

if __name__ == "__main__":
    main()