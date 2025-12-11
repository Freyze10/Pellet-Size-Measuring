import cv2
import numpy as np
import time
from ultralytics import YOLO

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
YOLO_MODEL_PATH = "yolo/best.pt"

# Measurement Targets
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5

# Ruler Settings
# Does your ruler mark every 1mm? (Standard CM ruler = 1.0)
RULER_TICK_SPACING_MM = 1.0
# Min confidence to trust the YOLO detection of the ruler
RULER_CONFIDENCE = 0.4

# Pellet Detection Settings
MIN_AREA_PX = 300
MAX_AREA_PX = 15000
MAX_ASPECT_RATIO = 4.0  # Filters out long objects (like the ruler itself)


# ----------------------------------------------------------------------
# SYSTEM STATE
# ----------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.mode = 'LIVE'  # LIVE, FROZEN
        self.frame = None
        self.display_image = None
        self.pixels_per_mm = None
        self.calibration_status = "Not Calibrated"

        # Manual Fallback
        self.calib_start = None
        self.calib_end = None
        self.is_dragging = False

        self.pellets = []


state = AppState()

# Load YOLO
print("Loading YOLO model...")
try:
    model = YOLO(YOLO_MODEL_PATH)
    print("✓ Model loaded")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# ----------------------------------------------------------------------
# 1. AUTO-CALIBRATION LOGIC (YOLO + Line Processing)
# ----------------------------------------------------------------------
def analyze_ruler_ticks(ruler_roi):
    """
    Analyzes the cut-out image of the ruler to find pixel spacing.
    """
    # 1. Gray & Threshold
    gray = cv2.cvtColor(ruler_roi, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold to handle glare/shadows on the ruler
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 5)

    # 2. Extract Vertical Lines (Tick Marks)
    # This kernel destroys horizontal lines (numbers/edges) and keeps vertical ticks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    ticks_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. Find Contours of the ticks
    contours, _ = cv2.findContours(ticks_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tick_centers = []
    h, w = ruler_roi.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5: continue  # Ignore noise

        x, y, cw, ch = cv2.boundingRect(cnt)

        # Filter: Ticks should be somewhat tall relative to width
        aspect = ch / float(cw)
        if aspect < 1.5: continue

        # Store the center X position
        tick_centers.append(x + cw / 2)

    if len(tick_centers) < 5:
        return None  # Not enough ticks found to be sure

    # 4. Calculate Spacing
    tick_centers.sort()

    # Calculate distance between adjacent ticks
    diffs = np.diff(tick_centers)

    # Remove outliers (e.g., gaps between big inch marks vs mm marks)
    # We use Interquartile Range (IQR) to find the "consistent" gap
    q1 = np.percentile(diffs, 25)
    q3 = np.percentile(diffs, 75)
    iqr = q3 - q1
    valid_diffs = [d for d in diffs if (q1 - 1.5 * iqr) <= d <= (q3 + 1.5 * iqr)]

    if not valid_diffs:
        return None

    avg_gap_px = np.mean(valid_diffs)

    # Result: Pixels per 1 MM (assuming ticks are 1mm apart)
    return avg_gap_px / RULER_TICK_SPACING_MM


def run_auto_calibration(image):
    """
    Uses YOLO to find the ruler, then calls analyze_ruler_ticks
    """
    print("Attempting Auto-Calibration...")
    results = model(image, verbose=False)

    best_conf = 0
    ruler_box = None

    # Find the best ruler detection
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id].lower()
        conf = float(box.conf[0])

        # Adjust these names based on what your model actually outputs
        # Common names: "ruler", "scale", "cm", "mm"
        if ("ruler" in name or "cm" in name or "inch" in name) and conf > RULER_CONFIDENCE:
            if conf > best_conf:
                best_conf = conf
                ruler_box = map(int, box.xyxy[0].cpu().numpy())

    if ruler_box:
        x1, y1, x2, y2 = ruler_box
        # Crop the ruler
        roi = image[y1:y2, x1:x2]

        # Analyze ticks inside the crop
        px_per_mm = analyze_ruler_ticks(roi)

        if px_per_mm:
            state.pixels_per_mm = px_per_mm
            state.calibration_status = f"AUTO SUCCESS ({px_per_mm:.2f} px/mm)"
            print(state.calibration_status)
            return True
        else:
            state.calibration_status = "Found Ruler, but tick marks unclear."
            return False
    else:
        state.calibration_status = "Auto-Calibrate Failed: No ruler found."
        return False


# ----------------------------------------------------------------------
# 2. PELLET MEASUREMENT (OpenCV Blobs)
# ----------------------------------------------------------------------
def detect_pellets(image):
    if state.pixels_per_mm is None: return

    state.pellets = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # Adaptive Threshold to handle lighting
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_PX or area > MAX_AREA_PX: continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect

        dim1, dim2 = min(w, h), max(w, h)

        # Ruler Rejection: Rulers are long and thin
        if dim1 > 0:
            aspect = dim2 / dim1
            if aspect > MAX_ASPECT_RATIO: continue

            # Convert to MM
        mm_d = dim1 / state.pixels_per_mm
        mm_l = dim2 / state.pixels_per_mm

        if mm_d < 0.5: continue

        is_good = (
                (TARGET_DIAMETER - TOLERANCE <= mm_d <= TARGET_DIAMETER + TOLERANCE) and
                (TARGET_LENGTH - TOLERANCE <= mm_l <= TARGET_LENGTH + TOLERANCE)
        )

        box = np.intp(cv2.boxPoints(rect))
        state.pellets.append({
            'box': box, 'cx': int(cx), 'cy': int(cy),
            'd_mm': mm_d, 'l_mm': mm_l, 'is_good': is_good
        })


# ----------------------------------------------------------------------
# UI & HELPERS
# ----------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    if state.mode != 'FROZEN': return
    # Manual Override Logic
    if event == cv2.EVENT_LBUTTONDOWN:
        state.calib_start = (x, y)
        state.is_dragging = True
    elif event == cv2.EVENT_MOUSEMOVE and state.is_dragging:
        state.calib_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        state.is_dragging = False
        state.calib_end = (x, y)
        p1, p2 = np.array(state.calib_start), np.array(state.calib_end)
        dist = np.linalg.norm(p1 - p2)
        if dist > 10:
            # Assume manual drag is exactly 10mm (1cm)
            state.pixels_per_mm = dist / 10.0
            state.calibration_status = f"MANUAL ({state.pixels_per_mm:.2f} px/mm)"
            detect_pellets(state.frame)


def draw_ui(img):
    h, w = img.shape[:2]

    if state.mode == 'LIVE':
        cv2.putText(img, "LIVE VIEW", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, "Ensure Ruler is visible & flat.", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
        cv2.putText(img, "Press SPACE to Capture & Auto-Calibrate", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
    else:
        # Frozen Mode
        cv2.putText(img, "ANALYSIS MODE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        # Calibration Status
        color = (0, 255, 0) if "SUCCESS" in state.calibration_status or "MANUAL" in state.calibration_status else (
        0, 0, 255)
        cv2.putText(img, f"Calib: {state.calibration_status}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if "Failed" in state.calibration_status:
            cv2.putText(img, "Draw 10mm line on ruler to calibrate manually.", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 1)

        # Draw Manual Line if dragging
        if state.calib_start and state.calib_end:
            cv2.line(img, state.calib_start, state.calib_end, (0, 255, 255), 2)

        # Draw Pellets
        good, bad = 0, 0
        for p in state.pellets:
            col = (0, 255, 0) if p['is_good'] else (0, 0, 255)
            cv2.drawContours(img, [p['box']], 0, col, 2)
            lbl = f"{p['d_mm']:.2f}x{p['l_mm']:.2f}"
            cv2.putText(img, lbl, (p['cx'] - 30, p['cy'] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
            cv2.putText(img, lbl, (p['cx'] - 30, p['cy'] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if p['is_good']:
                good += 1
            else:
                bad += 1

        # Stats
        if state.pixels_per_mm:
            cv2.rectangle(img, (20, h - 80), (200, h - 20), (30, 30, 30), -1)
            cv2.putText(img, f"GOOD: {good}", (30, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"BAD:  {bad}", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return img


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    cv2.namedWindow("Inspector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Inspector", mouse_callback)

    while True:
        if state.mode == 'LIVE':
            ret, frame = cap.read()
            if not ret: break
            state.frame = frame.copy()
            state.display_image = frame.copy()
        elif state.mode == 'FROZEN':
            state.display_image = state.frame.copy()

        final_img = draw_ui(state.display_image)
        cv2.imshow("Inspector", final_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # SPACE
            if state.mode == 'LIVE':
                state.mode = 'FROZEN'
                # 1. Try Auto-Calibration
                success = run_auto_calibration(state.frame)
                # 2. If success, Measure Pellets
                if success:
                    detect_pellets(state.frame)
                else:
                    print("Auto-Calib failed. Waiting for manual input.")
            else:
                state.mode = 'LIVE'  # Unfreeze
                state.calibration_status = "Not Calibrated"
                state.pixels_per_mm = None
                state.pellets = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()