import cv2
import numpy as np
import time
import sys
import math
from ultralytics import YOLO
from collections import defaultdict

# ----------------------------------------------------------------------
# Global Calibration
# ----------------------------------------------------------------------
PIXELS_PER_MM = 6.0
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0

# YOLO Model
yolo_model = None
last_calibration_time = 0
CALIBRATION_INTERVAL = 2.0  # Recalibrate every 2 seconds

def load_yolo_model():
    global yolo_model
    try:
        yolo_model = YOLO("yolo/best.pt")
        print("✓ YOLO model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
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

DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720

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
# YOLO Ruler Detection
# ----------------------------------------------------------------------
def detect_ruler_yolo(frame):
    global yolo_model
    if yolo_model is None:
        return None
    try:
        results = yolo_model(frame, conf=0.5, verbose=False)
        if len(results) > 0 and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            return {
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(conf)
            }
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
    return None

# ----------------------------------------------------------------------
# IMPROVED Ruler Tick Detection and Auto-Calibration
# ----------------------------------------------------------------------
def analyze_ruler_region(frame, bbox):
    x1, y1, x2, y2 = bbox
    padding = 20
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame.shape[1], x2 + padding)
    y2 = min(frame.shape[0], y2 + padding)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=25,
                            minLineLength=10, maxLineGap=8)

    if lines is None or len(lines) < 10:
        return None

    center_x = roi.shape[1] // 2
    left_ticks = []
    right_ticks = []

    for line in lines:
        x1_l, y1_l, x2_l, y2_l = line[0]
        length = np.hypot(x2_l - x1_l, y2_l - y1_l)
        if length < 10:
            continue
        mid_x = (x1_l + x2_l) / 2
        mid_y = (y1_l + y2_l) / 2
        angle = np.degrees(np.arctan2(y2_l - y1_l, x2_l - x1_l))
        angle = (angle + 180) % 180

        tick = {
            'mid_x': mid_x, 'mid_y': mid_y,
            'length': length, 'angle': angle
        }
        if abs(angle - 90) < 40:  # near-vertical ticks
            if mid_x < center_x:
                left_ticks.append(tick)
            else:
                right_ticks.append(tick)

    # Take longest ticks on each side
    left_ticks = sorted(left_ticks, key=lambda t: t['length'], reverse=True)[:30]
    right_ticks = sorted(right_ticks, key=lambda t: t['length'], reverse=True)[:30]

    sides = [("LEFT", left_ticks), ("RIGHT", right_ticks)]
    best_px_per_mm = None
    best_confidence = 0
    best_details = None

    for side_name, ticks in sides:
        if len(ticks) < 3:
            continue

        # Sort by vertical position
        ticks_sorted = sorted(ticks, key=lambda t: t['mid_y'])
        positions = [t['mid_y'] for t in ticks_sorted]
        lengths = [t['length'] for t in ticks_sorted]

        # Find intervals
        intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        if len(intervals) < 2:
            continue

        median_interval = np.median(intervals)
        valid_intervals = [iv for iv in intervals if abs(iv - median_interval) < median_interval * 0.4]
        if len(valid_intervals) < 2:
            continue

        avg_interval_px = np.mean(valid_intervals)
        avg_tick_length = np.mean([lengths[i] for i in range(len(positions)-1)
                                   if abs(intervals[i] - median_interval) < median_interval * 0.4])

        # Decide unit based on tick length
        confidence = 50
        if avg_tick_length > 40:
            unit = "inch"; mm_per_unit = 25.4; confidence += 50
        elif avg_tick_length > 22:
            unit = "cm"; mm_per_unit = 10.0; confidence += 30
        else:
            unit = "mm"; mm_per_unit = 1.0; confidence += 10

        px_per_mm = avg_interval_px / mm_per_unit
        if not (2.0 < px_per_mm < 12.0):
            confidence -= 40

        if confidence > best_confidence:
            best_confidence = confidence
            best_px_per_mm = px_per_mm
            best_details = {
                "side": side_name,
                "detected_unit": unit,
                "avg_interval_px": avg_interval_px,
                "avg_tick_length": avg_tick_length,
                "num_intervals": len(valid_intervals),
                "px_per_mm": px_per_mm,
                "confidence": confidence,
                "ticks": ticks_sorted
            }

    if best_details is None or best_confidence < 40:
        return None

    result = {
        "pixels_per_mm": best_px_per_mm,
        "detected_unit": best_details["detected_unit"],
        "unit_confidence": min(99, best_confidence),
        "calibration_side": best_details["side"],
        "avg_interval_px": best_details["avg_interval_px"],
        "num_ticks": len(best_details["ticks"]),
        "num_intervals": best_details["num_intervals"],
        "roi_offset": (x1, y1),
        "bbox": bbox,
        "tick_positions": [(t['mid_x'] + x1, t['mid_y'] + y1) for t in best_details["ticks"]],
        "total_length_mm": best_details["num_intervals"] *
                           (25.4 if best_details["detected_unit"] == "inch" else
                            10 if best_details["detected_unit"] == "cm" else 1)
    }
    return result

# ----------------------------------------------------------------------
# Pellet Detection
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
        width_px = min(edge1, edge2)
        height_px = max(edge1, edge2)

        diameter = width_px / PIXELS_PER_MM
        length = height_px / PIXELS_PER_MM

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
# Draw Overlay
# ----------------------------------------------------------------------
def draw_overlay(frame, pellets, ruler_info=None):
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
        color = (0, 255, 0) if p['within_tolerance'] else (0, 0, 255)
        cv2.drawContours(frame, [box], 0, color, 2)
        cv2.circle(frame, (int(center[0]), int(center[1])), 3, color, -1)

        left_x = int(min(box[:, 0]))
        top_y = int(min(box[:, 1]))
        bg_y = max(top_y - 35, 0)
        cv2.rectangle(frame, (left_x, bg_y), (left_x + 75, top_y - 5), (0, 0, 0), -1)
        cv2.putText(frame, f"D:{p['diameter']:.2f}mm", (left_x + 3, bg_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(frame, f"L:{p['length']:.2f}mm", (left_x + 3, bg_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        if not p['within_tolerance']:
            top_right = box[np.argmax(box[:, 0])]
            cv2.circle(frame, tuple(top_right), 8, (0, 0, 255), -1)
            cv2.putText(frame, "!", (top_right[0] - 6, top_right[1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Ruler overlay
    if ruler_info and 'detected_unit' in ruler_info:
        bbox = ruler_info['bbox']
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 3)

        # Calibration panel
        panel_x, panel_y = 10, 60
        panel_w, panel_h = 400, 130
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                      (30, 30, 50), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                      (255, 255, 0), 2)

        cv2.putText(frame, "AUTO-CALIBRATION ACTIVE", (panel_x + 10, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"Using {ruler_info['calibration_side']} side: {ruler_info['detected_unit'].upper()}",
                    (panel_x + 10, panel_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {ruler_info['unit_confidence']}%",
                    (panel_x + 10, panel_y + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
        cv2.putText(frame, f"Scale: {ruler_info['pixels_per_mm']:.3f} px/mm",
                    (panel_x + 10, panel_y + 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
        cv2.putText(frame, f"Ticks: {ruler_info['num_ticks']}  |  Length: {ruler_info['total_length_mm']:.1f}mm",
                    (panel_x + 10, panel_y + 116),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)

        # Draw used ticks (cyan circles)
        for tx, ty in ruler_info['tick_positions'][:20]:
            cv2.circle(frame, (int(tx), int(ty)), 5, (255, 255, 0), -1)

    elif ruler_info:
        bbox = ruler_info['bbox']
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
        cv2.putText(frame, f"RULER DETECTED ({ruler_info['confidence']:.0%})",
                    (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "Searching for ruler...", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.putText(frame, "Press 'q' to quit | Auto-calibration: ON",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)

    return frame

# ----------------------------------------------------------------------
# Camera Setup
# ----------------------------------------------------------------------
def get_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    print(f"Camera resolution: {int(cap.get(3))}x{int(cap.get(4))}")
    return cap

# ----------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------
def main():
    global PIXELS_PER_MM, last_calibration_time

    print("\n" + "="*60)
    print("   Pellet Inspector with Smart YOLO Auto-Calibration")
    print("="*60)
    print("Features:")
    print("  • Ruler detection with YOLO")
    print("  • Detects inches (longest ticks) vs cm automatically")
    print("  • Uses only the inch side for best accuracy")
    print("  • Real-time pellet measurement & counting")
    print("="*60 + "\n")

    if not load_yolo_model():
        sys.exit(1)

    cap = get_camera()
    if not cap.isOpened():
        print("Cannot open camera.")
        sys.exit(1)

    fps_counter = 0
    fps_start = time.time()
    fps_display = 0
    window_name = "Pellet Inspector - Smart Auto-Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    ruler_info = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera lost – reconnecting...")
            cap = get_camera()
            continue

        current_time = time.time()

        ruler_detection = detect_ruler_yolo(frame)

        if ruler_detection and (current_time - last_calibration_time) > CALIBRATION_INTERVAL:
            calibration_result = analyze_ruler_region(frame, ruler_detection['bbox'])
            if calibration_result:
                PIXELS_PER_MM = calibration_result['pixels_per_mm']
                update_ranges()
                last_calibration_time = current_time
                ruler_info = calibration_result
                ruler_info['confidence'] = ruler_detection['confidence']

                print(f"\nAUTO-CALIBRATION SUCCESS!")
                print(f"Using {calibration_result['calibration_side']} side → {calibration_result['detected_unit'].upper()}")
                print(f"Scale: {PIXELS_PER_MM:.3f} px/mm  |  Confidence: {calibration_result['unit_confidence']}%")
                print(f"Ruler length: {calibration_result['total_length_mm']:.1f} mm\n")
            elif ruler_info is None:
                ruler_info = ruler_detection
        elif not ruler_detection:
            if ruler_info and 'detected_unit' not in ruler_info:
                ruler_info = None

        pellets = detect_pellets(frame)
        frame = draw_overlay(frame, pellets, ruler_info)

        # FPS
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            fps_display = fps_counter // int(time.time() - fps_start)
            fps_counter = 0
            fps_start = time.time()
        cv2.putText(frame, f"FPS: {fps_display}", (frame.shape[1]-140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nShutdown complete.")

if __name__ == "__main__":
    main()