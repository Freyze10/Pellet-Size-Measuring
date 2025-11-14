import cv2
import numpy as np
import time
import sys
import json
import os
from pathlib import Path

# ----------------------------------------------------------------------
# Global Calibration
# ----------------------------------------------------------------------
PIXELS_PER_MM = 6.0
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 200.0

# ----------------------------------------------------------------------
# COCO Annotations & Feature Storage
# ----------------------------------------------------------------------
coco_annotations = []
coco_images = {}
reference_features = []  # Store Hu moments + area + aspect ratio for each labeled pellet
training_images_dir = "training_images"

# ----------------------------------------------------------------------
# Load COCO + Extract Features from Training Images
# ----------------------------------------------------------------------
def load_coco_annotations_and_features(json_path="pellets_label.json"):
    """Load COCO annotations and extract visual features from training images."""
    global coco_annotations, coco_images, reference_features

    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Running without reference features.")
        return False

    try:
        with open(json_path, 'r') as f:
            coco_data = json.load(f)

        # Store training_images info
        for img in coco_data.get('training_images', []):
            img_id = img['id']
            file_name = img.get('file_name', '')
            if not file_name:
                continue
            img_path = os.path.join(training_images_dir, file_name)
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} not found. Skipping.")
                continue
            coco_images[img_id] = {
                'file_name': file_name,
                'path': img_path,
                'width': img.get('width', 0),
                'height': img.get('height', 0)
            }

        # Store annotations
        coco_annotations = coco_data.get('annotations', [])

        # Extract features from each annotated pellet in training images
        for ann in coco_annotations:
            image_id = ann.get('image_id')
            if image_id not in coco_images:
                continue

            img_info = coco_images[image_id]
            img_path = img_info['path']
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Extract mask from segmentation
            if 'segmentation' not in ann or not ann['segmentation']:
                continue

            # Create binary mask from polygon
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            for seg in ann['segmentation']:
                poly = np.array(seg, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [poly], 255)

            # Find contours in mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = contours[0]

            # Compute features
            area = cv2.contourArea(cnt)
            if area < 50:
                continue

            # Hu moments (rotation, scale, translation invariant)
            moments = cv2.moments(cnt)
            if moments["m00"] == 0:
                continue
            hu_moments = cv2.HuMoments(moments).flatten()
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-8)  # Stabilize

            # Bounding box aspect ratio
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h != 0 else 1.0

            # Perimeter
            perimeter = cv2.arcLength(cnt, True)

            # Compactness
            compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0

            # Store feature vector
            feature_vec = np.concatenate([
                hu_moments,  # 7 values
                [area, aspect_ratio, compactness]  # 3 more
            ])

            reference_features.append({
                'features': feature_vec,
                'area': area,
                'bbox': [x, y, w, h],
                'image_id': image_id,
                'file_name': img_info['file_name']
            })

        print(f"Loaded {len(reference_features)} reference pellet features from {len(coco_images)} training images.")
        return True

    except Exception as e:
        print(f"Error loading COCO annotations or features: {e}")
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
MAX_CONTOUR_AREA = 10000

# ----------------------------------------------------------------------
# Calibration Panel State
# ----------------------------------------------------------------------
in_calib_mode = False

# Panel layout
PANEL_X, PANEL_Y = 10, 300
PANEL_W, PANEL_H = 300, 180

# Button rects
UP_BTN = (PANEL_X + 230, PANEL_Y + 40, 50, 40)
DOWN_BTN = (PANEL_X + 230, PANEL_Y + 90, 50, 40)
BACK_BTN = (PANEL_X + 20, PANEL_Y + 130, 80, 35)


# ----------------------------------------------------------------------
# Mouse Callback
# ----------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global PIXELS_PER_MM, in_calib_mode

    if not in_calib_mode:
        return

    def in_rect(px, py, rect):
        rx, ry, rw, rh = rect
        return rx <= px <= rx + rw and ry <= py <= ry + rh

    if event == cv2.EVENT_LBUTTONDOWN:
        if in_rect(x, y, UP_BTN):
            PIXELS_PER_MM = round(PIXELS_PER_MM + 0.1, 2)
            update_ranges()
        elif in_rect(x, y, DOWN_BTN):
            if PIXELS_PER_MM > 0.5:
                PIXELS_PER_MM = round(PIXELS_PER_MM - 0.1, 2)
                update_ranges()
        elif in_rect(x, y, BACK_BTN):
            in_calib_mode = False


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
# Feature Matching - Match detected contour to trained pellet features
# ----------------------------------------------------------------------
def extract_features(contour):
    """Extract same features as in training."""
    area = cv2.contourArea(contour)
    if area < 50:
        return None

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-8)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h != 0 else 1.0
    perimeter = cv2.arcLength(contour, True)
    compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0

    return np.concatenate([hu_moments, [area, aspect_ratio, compactness]])


def match_to_reference_feature(contour):
    """Match contour to closest trained pellet using feature distance."""
    if not reference_features:
        return None

    features = extract_features(contour)
    if features is None:
        return None

    # Normalize features (same scale as training)
    ref_areas = np.array([rf['area'] for rf in reference_features])
    if len(ref_areas) == 0:
        return None
    area_mean, area_std = ref_areas.mean(), ref_areas.std()
    if area_std == 0:
        area_std = 1.0

    # Normalize area
    features[-3] = (features[-3] - area_mean) / area_std

    min_dist = float('inf')
    best_match = None

    for ref in reference_features:
        ref_feat = ref['features'].copy()
        ref_feat[-3] = (ref_feat[-3] - area_mean) / area_std  # Normalize same way

        # Weighted Euclidean distance (Hu moments more important)
        diff = features - ref_feat
        weights = np.array([1.0]*7 + [0.5, 0.3, 0.2])  # Hu > area > aspect > compactness
        dist = np.sqrt(np.sum(weights * (diff ** 2)))

        if dist < min_dist:
            min_dist = dist
            best_match = ref

    # Threshold: only accept good matches
    if min_dist < 2.5:  # Tuned threshold
        return best_match

    return None


# ----------------------------------------------------------------------
# Detection with Feature-Based Recognition
# ----------------------------------------------------------------------
def detect_pellets(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
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
        width_mm = w / PIXELS_PER_MM
        height_mm = h / PIXELS_PER_MM
        diameter = min(width_mm, height_mm)
        length = max(width_mm, height_mm)

        if not should_process_pellet(diameter, length):
            continue

        # Feature-based matching
        ref_match = match_to_reference_feature(cnt)

        # Only accept if it matches a trained pellet
        if ref_match is None:
            continue

        within_tol = is_within_tolerance(diameter, length)

        pellets.append({
            'x': x, 'y': y, 'w': w, 'h': h,
            'diameter': diameter,
            'length': length,
            'within_tolerance': within_tol,
            'contour': cnt,
            'reference_match': ref_match,
            'match_confidence': min_dist if 'min_dist' in locals() else 0
        })
    return pellets


# ----------------------------------------------------------------------
# Draw Calibration Mode
# ----------------------------------------------------------------------
def draw_calibration_mode(frame):
    overlay = frame.copy()

    # Panel background
    cv2.rectangle(overlay, (PANEL_X, PANEL_Y), (PANEL_X + PANEL_W, PANEL_Y + PANEL_H),
                  (30, 30, 50), -1)
    cv2.rectangle(overlay, (PANEL_X, PANEL_Y), (PANEL_X + PANEL_W, PANEL_Y + PANEL_H),
                  (100, 150, 255), 3)

    # Title
    cv2.putText(overlay, "CALIBRATION MODE", (PANEL_X + 15, PANEL_Y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Current value
    cv2.putText(overlay, f"{PIXELS_PER_MM:.2f} px/mm", (PANEL_X + 15, PANEL_Y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # UP Button
    cv2.rectangle(overlay, (UP_BTN[0], UP_BTN[1]), (UP_BTN[0] + UP_BTN[2], UP_BTN[1] + UP_BTN[3]),
                  (0, 200, 0), -1)
    cv2.putText(overlay, "UP", (UP_BTN[0] + 12, UP_BTN[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # DOWN Button
    cv2.rectangle(overlay, (DOWN_BTN[0], DOWN_BTN[1]), (DOWN_BTN[0] + DOWN_BTN[2], DOWN_BTN[1] + DOWN_BTN[3]),
                  (200, 0, 0), -1)
    cv2.putText(overlay, "DOWN", (DOWN_BTN[0] + 5, DOWN_BTN[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # BACK Button
    cv2.rectangle(overlay, (BACK_BTN[0], BACK_BTN[1]), (BACK_BTN[0] + BACK_BTN[2], BACK_BTN[1] + BACK_BTN[3]),
                  (100, 100, 100), -1)
    cv2.putText(overlay, "BACK", (BACK_BTN[0] + 10, BACK_BTN[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)


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
        x, y, w, h = p['x'], p['y'], p['w'], p['h']
        color = (0, 255, 0) if p['within_tolerance'] else (0, 0, 255)

        # Draw bounding rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Smaller background (140 px wide to fit source info)
        bg_y = max(y - 55, 0)
        cv2.rectangle(frame, (x, bg_y), (x + 140, y - 5), (0, 0, 0), -1)
        cv2.putText(frame, f"D: {p['diameter']:.2f}", (x + 5, bg_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"L: {p['length']:.2f}", (x + 5, bg_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Show source image name
        src = p['reference_match']['file_name']
        short_src = src.split('_')[-1]  # e.g., Pro.jpg
        cv2.putText(frame, f"Src: {short_src}", (x + 5, bg_y + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 255, 200), 1)

        if not p['within_tolerance']:
            cv2.circle(frame, (x + w - 10, y + 10), 8, (0, 0, 255), -1)
            cv2.putText(frame, "!", (x + w - 14, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Hint
    if not in_calib_mode:
        hint_text = "Press 'c' for calibration"
        if reference_features:
            hint_text += f" | {len(reference_features)} trained pellets"
        cv2.putText(frame, hint_text,
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)

    # Calibration panel
    if in_calib_mode:
        draw_calibration_mode(frame)

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
    global in_calib_mode, PIXELS_PER_MM

    print("\nPellet Inspector + Trained Feature Recognition")
    print("=" * 60)

    # Load COCO + extract features
    success = load_coco_annotations_and_features("pellets_label.json")
    if not success or not reference_features:
        print("No trained pellet features loaded. Detection disabled.")
        print("Place 'pellets_label.json' and 'training_images/' folder in project root.")
        sys.exit(1)

    print("Press 'c' → Enter calibration mode")
    print("Click UP/DOWN to adjust | Click BACK to exit")
    print("Press 'q' to quit")
    print("=" * 60)

    cap = get_camera()
    if not cap.isOpened():
        print("Cannot open camera.")
        sys.exit(1)

    fps_counter = 0
    fps_start = time.time()
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
            fps_start = time.time()

        cv2.putText(display_frame, f"FPS: {fps_display}",
                    (display_frame.shape[1] - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, display_frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c') and not in_calib_mode:
            in_calib_mode = True

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Shutdown complete.")


if __name__ == "__main__":
    main()