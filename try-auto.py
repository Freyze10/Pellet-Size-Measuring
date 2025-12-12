import cv2
import numpy as np
import time
import math
from collections import deque
from ultralytics import YOLO

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
YOLO_MODEL_PATH = "yolo/best.pt"

# Initial Calibration Defaults
PIXELS_PER_MM = 7.0
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0
RULER_HEIGHT_ADJUSTMENT = 1.0

# Calibration Modes
CALIBRATION_MODE = 'CM'
MIN_LINES_CM = 5
DIVISOR_CM = 10.0
MIN_LINES_INCH = 4
DIVISOR_INCH = 25.4

# Detection Settings
ASPECT_RATIO_MIN = 2.0
HEIGHT_RATIO_STRICT = 0.85
MAX_GAP_VARIANCE = 0.08
EDGE_MARGIN_PERCENT = 0.30

# System State
yolo_model = None
is_calibrated = False
calibration_error_msg = ""

# NEW: Capture Mode State
captured_frame = None
in_capture_mode = False
calibration_mode_active = False
current_tick_data = None
current_pellets = []
current_yolo_objects = []

# Manual Calibration
in_manual_calib_mode = False
MANUAL_REFERENCE_LENGTH_MM = 76.2
manual_line_start = None
manual_line_end = None
is_dragging = False

# UI Elements
MANUAL_PANEL_X, MANUAL_PANEL_Y = 10, 80
MANUAL_PANEL_W, MANUAL_PANEL_H = 380, 260
RESET_BTN = (MANUAL_PANEL_X + 20, MANUAL_PANEL_Y + 180, 100, 40)
APPLY_BTN = (MANUAL_PANEL_X + 140, MANUAL_PANEL_Y + 180, 100, 40)
CANCEL_BTN = (MANUAL_PANEL_X + 260, MANUAL_PANEL_Y + 180, 100, 40)

# Camera Settings
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 20000


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


def load_yolo_model():
    global yolo_model
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"✓ YOLO model loaded: {YOLO_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        return False


def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def mouse_callback(event, x, y, flags, param):
    global manual_line_start, manual_line_end, is_dragging
    global in_manual_calib_mode, PIXELS_PER_MM, is_calibrated

    if not in_manual_calib_mode:
        return

    def in_rect(px, py, rect):
        rx, ry, rw, rh = rect
        return rx <= px <= rx + rw and ry <= py <= ry + rh

    if event == cv2.EVENT_LBUTTONDOWN:
        if in_rect(x, y, RESET_BTN):
            manual_line_start = None
            manual_line_end = None
            is_dragging = False
        elif in_rect(x, y, APPLY_BTN):
            if manual_line_start and manual_line_end:
                dx = manual_line_end[0] - manual_line_start[0]
                dy = manual_line_end[1] - manual_line_start[1]
                pixel_distance = math.sqrt(dx ** 2 + dy ** 2)
                if pixel_distance > 10:
                    PIXELS_PER_MM = (pixel_distance / MANUAL_REFERENCE_LENGTH_MM) + 0.35
                    is_calibrated = True
                    update_ranges()
                    print(f"✓ Manual Calibration: {PIXELS_PER_MM:.2f} px/mm")
                in_manual_calib_mode = False
                manual_line_start = None
                manual_line_end = None
        elif in_rect(x, y, CANCEL_BTN):
            in_manual_calib_mode = False
            manual_line_start = None
            manual_line_end = None
        elif not in_rect(x, y, (MANUAL_PANEL_X, MANUAL_PANEL_Y, MANUAL_PANEL_W, MANUAL_PANEL_H)):
            manual_line_start = (x, y)
            manual_line_end = (x, y)
            is_dragging = True

    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        manual_line_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP and is_dragging:
        manual_line_end = (x, y)
        is_dragging = False


def is_within_tolerance(d, l):
    return DIAMETER_MIN <= d <= DIAMETER_MAX and LENGTH_MIN <= l <= LENGTH_MAX


def should_process_pellet(d, l):
    if d < 0.5 or l < 0.5: return False
    return (DIAMETER_EXCLUDE_MIN <= d <= DIAMETER_EXCLUDE_MAX and
            LENGTH_EXCLUDE_MIN <= l <= LENGTH_EXCLUDE_MAX)


def remove_outliers_iqr(data, factor=1.5):
    if len(data) < 4: return data
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - (factor * iqr)
    upper = q3 + (factor * iqr)
    return [x for x in data if lower <= x <= upper]


def calculate_consistent_gaps(ticks):
    if len(ticks) < 2: return None, 0
    sorted_ticks = sorted(ticks, key=lambda x: x['pos'])
    gaps = [sorted_ticks[i + 1]['pos'] - sorted_ticks[i]['pos'] for i in range(len(sorted_ticks) - 1)]
    if not gaps: return None, 0
    clean = remove_outliers_iqr(gaps)
    if len(clean) < max(2, len(gaps) * 0.6): return None, 0
    mean = np.mean(clean)
    std = np.std(clean)
    cv = std / mean if mean > 0 else 1.0
    return mean, cv


def run_yolo_detection(frame):
    if yolo_model is None: return [], None
    results = yolo_model(frame, conf=0.35, verbose=False)
    best_map = {}
    best_zone = None
    best_conf = 0

    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = yolo_model.names[cls_id]
            if name not in best_map or conf > best_map[name]['conf']:
                best_map[name] = {'box': (x1, y1, x2, y2), 'name': name, 'conf': conf}

    all_dets = list(best_map.values())
    for det in all_dets:
        is_pref = "mm" in det['name'].lower() or "zone" in det['name'].lower()
        if is_pref:
            if best_zone is None or det['conf'] > best_conf:
                best_conf = det['conf']
                best_zone = det['box']
        elif best_zone is None and "ruler" in det['name'].lower():
            best_zone = det['box']
    return all_dets, best_zone


def analyze_structure(frame, bbox):
    global calibration_error_msg
    MIN_LINES = MIN_LINES_CM if CALIBRATION_MODE == 'CM' else MIN_LINES_INCH
    DIVISOR = DIVISOR_CM if CALIBRATION_MODE == 'CM' else DIVISOR_INCH

    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    pad = 5
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
    roi = frame[cy1:cy2, cx1:cx2]
    if roi.size == 0: return None

    is_horiz = (cx2 - cx1) > (cy2 - cy1)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 5)

    if is_horiz:
        k_h = max(5, roi.shape[0] // 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_h))
    else:
        k_w = max(5, roi.shape[1] // 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))

    lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    lines = cv2.dilate(lines, None, iterations=1)
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ticks = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 5: continue
        tx, ty, tw, th = cv2.boundingRect(cnt)
        if is_horiz:
            ratio = th / float(tw) if tw > 0 else 0
            if ratio > ASPECT_RATIO_MIN:
                ticks.append({'pos': tx + tw / 2.0, 'len': th, 'rect': (tx, ty, tw, th)})
        else:
            ratio = tw / float(th) if th > 0 else 0
            if ratio > ASPECT_RATIO_MIN:
                ticks.append({'pos': ty + th / 2.0, 'len': tw, 'rect': (tx, ty, tw, th)})

    if len(ticks) < MIN_LINES:
        calibration_error_msg = f"Need {MIN_LINES}+ lines (found {len(ticks)})"
        return None

    roi_sz = (cx2 - cx1) if is_horiz else (cy2 - cy1)
    edge = roi_sz * EDGE_MARGIN_PERCENT
    max_len = max(t['len'] for t in ticks)
    thresh = max_len * HEIGHT_RATIO_STRICT

    majors = []
    for t in ticks:
        if is_horiz:
            ty = t['rect'][1]
            at_edge = (ty < edge) or ((ty + t['rect'][3]) > (roi.shape[0] - edge))
        else:
            tx = t['rect'][0]
            at_edge = (tx < edge) or ((tx + t['rect'][2]) > (roi.shape[1] - edge))

        if t['len'] >= thresh and at_edge:
            t['type'] = 'MAJOR'
            majors.append(t)
        elif t['len'] > (max_len * 0.60):
            t['type'] = 'MEDIUM'
        else:
            t['type'] = 'MINOR'

    if len(majors) < MIN_LINES:
        calibration_error_msg = f"Need {MIN_LINES} {CALIBRATION_MODE} lines"
        return {"px_per_mm": 0, "ticks": ticks, "roi_offset": (cx1, cy1)}

    gap, cv = calculate_consistent_gaps(majors)
    if gap is None or cv > MAX_GAP_VARIANCE:
        calibration_error_msg = f"Spacing uneven (CV: {cv * 100:.1f}%)"
        return {"px_per_mm": 0, "ticks": ticks, "roi_offset": (cx1, cy1)}

    px_mm = (gap / DIVISOR) * RULER_HEIGHT_ADJUSTMENT
    if px_mm < 2 or px_mm > 150:
        calibration_error_msg = f"Invalid scale ({px_mm:.2f})"
        return None

    calibration_error_msg = ""
    return {
        "px_per_mm": px_mm,
        "ticks": ticks,
        "roi_offset": (cx1, cy1),
        "major_count": len(majors),
        "spacing_variance": cv
    }


def detect_pellets(frame, excluded):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pellets = []
    for cnt in contours:
        if not (MIN_CONTOUR_AREA <= cv2.contourArea(cnt) <= MAX_CONTOUR_AREA): continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        box = np.intp(cv2.boxPoints(rect))

        skip = False
        for ex in excluded:
            bx1, by1, bx2, by2 = ex['box']
            if (bx1 - 5) <= cx <= (bx2 + 5) and (by1 - 5) <= cy <= (by2 + 5):
                skip = True
                break
        if skip: continue

        d1 = get_distance(box[0], box[1])
        d2 = get_distance(box[1], box[2])
        w_px, h_px = min(d1, d2), max(d1, d2)

        d_mm = w_px / PIXELS_PER_MM
        l_mm = h_px / PIXELS_PER_MM

        if should_process_pellet(d_mm, l_mm):
            pellets.append({
                'box': box,
                'diameter': d_mm,
                'length': l_mm,
                'is_good': is_within_tolerance(d_mm, l_mm)
            })

    return pellets


def draw_manual_calib(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (MANUAL_PANEL_X, MANUAL_PANEL_Y),
                  (MANUAL_PANEL_X + MANUAL_PANEL_W, MANUAL_PANEL_Y + MANUAL_PANEL_H),
                  (30, 30, 50), -1)
    cv2.rectangle(overlay, (MANUAL_PANEL_X, MANUAL_PANEL_Y),
                  (MANUAL_PANEL_X + MANUAL_PANEL_W, MANUAL_PANEL_Y + MANUAL_PANEL_H),
                  (100, 150, 255), 3)

    cv2.putText(overlay, "MANUAL CALIBRATION", (MANUAL_PANEL_X + 65, MANUAL_PANEL_Y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    instructions = [
        "1. Click and drag to match",
        "   reference (3 inch / 76.2 mm)",
        "2. Click APPLY"
    ]

    y = MANUAL_PANEL_Y + 60
    for inst in instructions:
        cv2.putText(overlay, inst, (MANUAL_PANEL_X + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += 20

    cv2.putText(overlay, "Reference: 3 inch (76.2 mm)",
                (MANUAL_PANEL_X + 60, MANUAL_PANEL_Y + 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)

    # Buttons
    cv2.rectangle(overlay, (RESET_BTN[0], RESET_BTN[1]),
                  (RESET_BTN[0] + RESET_BTN[2], RESET_BTN[1] + RESET_BTN[3]),
                  (50, 50, 200), -1)
    cv2.putText(overlay, "RESET", (RESET_BTN[0] + 15, RESET_BTN[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    enabled = manual_line_start and manual_line_end
    color = (0, 200, 0) if enabled else (100, 100, 100)
    cv2.rectangle(overlay, (APPLY_BTN[0], APPLY_BTN[1]),
                  (APPLY_BTN[0] + APPLY_BTN[2], APPLY_BTN[1] + APPLY_BTN[3]),
                  color, -1)
    cv2.putText(overlay, "APPLY", (APPLY_BTN[0] + 15, APPLY_BTN[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.rectangle(overlay, (CANCEL_BTN[0], CANCEL_BTN[1]),
                  (CANCEL_BTN[0] + CANCEL_BTN[2], CANCEL_BTN[1] + CANCEL_BTN[3]),
                  (100, 100, 100), -1)
    cv2.putText(overlay, "CANCEL", (CANCEL_BTN[0] + 10, CANCEL_BTN[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(overlay, f"Current: {PIXELS_PER_MM:.2f} px/mm",
                (MANUAL_PANEL_X + 20, MANUAL_PANEL_Y + 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2)

    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    if manual_line_start and manual_line_end:
        cv2.line(frame, manual_line_start, manual_line_end, (0, 255, 255), 2)
        for pt in [manual_line_start, manual_line_end]:
            cv2.line(frame, (pt[0] - 10, pt[1]), (pt[0] + 10, pt[1]), (0, 0, 255), 2)
            cv2.line(frame, (pt[0], pt[1] - 10), (pt[0], pt[1] + 10), (0, 0, 255), 2)

        dx = manual_line_end[0] - manual_line_start[0]
        dy = manual_line_end[1] - manual_line_start[1]
        length = math.sqrt(dx ** 2 + dy ** 2)

        if length > 10:
            angle = math.atan2(dy, dx)

            def tick(dist_mm, tick_len, thick=1, col=(180, 180, 180)):
                t = dist_mm / MANUAL_REFERENCE_LENGTH_MM
                x = manual_line_start[0] + dx * t
                y = manual_line_start[1] + dy * t
                px = int(tick_len * math.sin(angle))
                py = int(-tick_len * math.cos(angle))
                cv2.line(frame, (int(x - px), int(y - py)), (int(x + px), int(y + py)), col, thick)

            for cm in range(1, 8):
                tick(cm * 10, 12, 2, (255, 255, 255))
                t = (cm * 10) / MANUAL_REFERENCE_LENGTH_MM
                x = manual_line_start[0] + dx * t
                y = manual_line_start[1] + dy * t
                lx = int(x + 25 * math.sin(angle))
                ly = int(y - 25 * math.cos(angle))
                cv2.putText(frame, str(cm), (lx - 8, ly + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            for half in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]:
                tick(half * 10, 10, 1, (200, 200, 200))

        mx = (manual_line_start[0] + manual_line_end[0]) // 2
        my = (manual_line_start[1] + manual_line_end[1]) // 2
        tx = int(-20 * math.sin(angle))
        ty = int(20 * math.cos(angle))
        cv2.putText(frame, f"{length:.1f} px", (mx + tx, my + ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def draw_ui(frame, yolo_objs, zone, pellets, ticks):
    # Draw ruler detection (calibration mode only)
    if calibration_mode_active:
        for obj in yolo_objs:
            bx1, by1, bx2, by2 = obj['box']
            col = (0, 255, 0) if (zone and obj['box'] == zone) else (100, 100, 100)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, 2)
            cv2.putText(frame, obj['name'], (bx1, by1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    # Draw ticks (calibration mode only)
    if calibration_mode_active and ticks:
        ox, oy = ticks['roi_offset']
        for t in ticks['ticks']:
            rx, ry, rw, rh = t['rect']
            cx = int(ox + rx + rw / 2)
            cy = int(oy + ry + rh / 2)
            is_vert = rh > rw

            if t['type'] == 'MAJOR':
                ln, col, th = 25, (0, 0, 255), 2
            else:
                ln, col, th = 12, (255, 255, 0), 1

            if is_vert:
                pt1 = (cx, cy - ln // 2)
                pt2 = (cx, cy + ln // 2)
            else:
                pt1 = (cx - ln // 2, cy)
                pt2 = (cx + ln // 2, cy)

            cv2.line(frame, pt1, pt2, col, th)

    # Draw pellets (measurement mode only)
    if not calibration_mode_active:
        for p in pellets:
            box = p['box']
            col = (0, 255, 0) if p['is_good'] else (0, 0, 255)
            cv2.drawContours(frame, [box], 0, col, 1)

            M = cv2.moments(box)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = box[0]

            txt = f"{p['diameter']:.2f}x{p['length']:.2f}"
            cv2.putText(frame, txt, (cx - 30, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(frame, txt, (cx - 30, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if not p['is_good']:
                cv2.putText(frame, "!", (cx - 45, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Status bar
    cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, 90), (20, 20, 20), -1)

    if calibration_mode_active:
        if is_calibrated:
            msg = f"CALIBRATION COMPLETE: {PIXELS_PER_MM:.2f} px/mm"
            col = (0, 255, 0)
        else:
            msg = f"CALIBRATION MODE: {calibration_error_msg}" if calibration_error_msg else "Ruler detected"
            col = (0, 165, 255)
        cv2.putText(frame, msg, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        cv2.putText(frame, f"MODE: {CALIBRATION_MODE} | 'u' to switch | 'Enter' to confirm",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    else:
        if is_calibrated:
            msg = f"CALIBRATED: {PIXELS_PER_MM:.2f} px/mm"
            col = (0, 255, 0)
        else:
            msg = "NOT CALIBRATED - Press 'a' or 'm'"
            col = (0, 0, 255)
        cv2.putText(frame, msg, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        if in_capture_mode:
            cv2.putText(frame, "CAPTURE MODE | 'ESC'=live | 'a'=auto | 'm'=manual | 'Space'=new | 'q'=quit",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "LIVE VIEW | Press 'Space' to capture",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Pellet count
        if pellets:
            good = sum(1 for p in pellets if p['is_good'])
            bad = len(pellets) - good
            cx = DESIRED_WIDTH - 350
            cv2.putText(frame, f"IN: {good}", (cx, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"OUT: {bad}", (cx, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame


def main():
    global captured_frame, in_capture_mode, calibration_mode_active
    global PIXELS_PER_MM, is_calibrated, CALIBRATION_MODE
    global in_manual_calib_mode, current_tick_data
    global current_pellets, current_yolo_objects

    if not load_yolo_model(): return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    window = "Pellet Inspector - Capture Mode"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse_callback)

    print("=" * 60)
    print("CAPTURE-BASED PELLET INSPECTOR")
    print("=" * 60)
    print("CONTROLS:")
    print("  SPACE    - Capture frame")
    print("  ESC      - Return to live view")
    print("  a        - Auto calibration (on captured image)")
    print("  m        - Manual calibration (on captured image)")
    print("  u        - Switch CM/INCH (during calibration)")
    print("  Enter    - Confirm calibration")
    print("  q        - Quit")
    print("=" * 60)

    while True:
        # Get the frame to display
        if in_capture_mode:
            frame = captured_frame.copy()
        else:
            ret, frame = cap.read()
            if not ret: break

        # Process based on mode
        if in_manual_calib_mode:
            draw_manual_calib(frame)
        elif calibration_mode_active and in_capture_mode:
            # Auto calibration on captured frame
            yolo_objs, zone = run_yolo_detection(frame)
            current_yolo_objects = yolo_objs

            if zone:
                result = analyze_structure(frame, zone)
                current_tick_data = result
                if result and result['px_per_mm'] > 0:
                    PIXELS_PER_MM = result['px_per_mm'] + 0.35
                    is_calibrated = True
                    update_ranges()
            else:
                current_tick_data = None

            frame = draw_ui(frame, current_yolo_objects, zone, [], current_tick_data)

        elif in_capture_mode and not calibration_mode_active:
            # Measurement mode on captured frame
            if is_calibrated:
                current_pellets = detect_pellets(frame, current_yolo_objects)
            else:
                current_pellets = []
            frame = draw_ui(frame, [], None, current_pellets, None)

        else:
            # Live preview mode
            frame = draw_ui(frame, [], None, [], None)

        cv2.imshow(window, frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == 27:  # ESC key - return to live view
            if in_capture_mode:
                in_capture_mode = False
                calibration_mode_active = False
                in_manual_calib_mode = False
                current_yolo_objects = []
                current_tick_data = None
                current_pellets = []
                manual_line_start = None
                manual_line_end = None
                print("\n📹 Returned to live view (calibration preserved)")

        elif key == ord(' '):  # Spacebar - capture
            if not in_capture_mode:
                ret, capture = cap.read()
                if ret:
                    captured_frame = capture.copy()
                    in_capture_mode = True
                    calibration_mode_active = False
                    current_yolo_objects = []
                    current_tick_data = None
                    current_pellets = []
                    print("\n✓ Frame captured! Press 'a' for auto calib or 'm' for manual")
            else:
                # New capture
                ret, capture = cap.read()
                if ret:
                    captured_frame = capture.copy()
                    calibration_mode_active = False
                    current_yolo_objects = []
                    current_tick_data = None
                    current_pellets = []
                    print("\n✓ New frame captured!")

        elif key == ord('a') and in_capture_mode and not in_manual_calib_mode:
            # Start auto calibration
            calibration_mode_active = True
            print("\n🔧 Auto-calibration mode activated...")

        elif key == ord('m') and in_capture_mode and not calibration_mode_active:
            # Start manual calibration
            in_manual_calib_mode = True
            manual_line_start = None
            manual_line_end = None
            print("\n📏 Manual calibration mode activated...")

        elif key == ord('u') and calibration_mode_active:
            # Switch calibration mode
            CALIBRATION_MODE = 'INCH' if CALIBRATION_MODE == 'CM' else 'CM'
            print(f"\n🔄 Switched to {CALIBRATION_MODE} mode")

        elif key == 13 and calibration_mode_active:  # Enter key
            # Confirm calibration
            if is_calibrated:
                calibration_mode_active = False
                print(f"\n✓ Calibration confirmed: {PIXELS_PER_MM:.2f} px/mm")
            else:
                print("\n✗ No valid calibration to confirm")

        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()