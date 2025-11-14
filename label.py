import cv2
import numpy as np
import time
import sys
import json
import os
from sklearn import svm
from skimage.feature import hog

# ----------------------------------------------------------------------
# Global Calibration
# ----------------------------------------------------------------------
PIXELS_PER_MM = 6.0
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 200.0

# ----------------------------------------------------------------------
# COCO & Training
# ----------------------------------------------------------------------
coco_data = None
detector = None
is_trained = False
trained_samples = 0

# HOG parameters
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# ----------------------------------------------------------------------
# Update Ranges
# ----------------------------------------------------------------------
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
# HOG Feature Extractor
# ----------------------------------------------------------------------
def extract_hog_features(img):
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog(img, orientations=HOG_ORIENTATIONS,
                   pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=HOG_CELLS_PER_BLOCK,
                   block_norm='L2-Hys', visualize=False)
    return features


# ----------------------------------------------------------------------
# Train Detector from COCO
# ----------------------------------------------------------------------
def train_from_coco(coco_json_path="pellets_label.json", images_folder="training_images"):
    global detector, is_trained, trained_samples, coco_data

    if not os.path.exists(coco_json_path):
        print(f"Warning: {coco_json_path} not found. Running without training.")
        return False

    if not os.path.exists(images_folder):
        print(f"Error: {images_folder} folder not found!")
        return False

    try:
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        X = []
        y = []

        print("Loading training samples from COCO annotations...")

        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            bbox = ann.get('bbox', None)  # [x, y, width, height]
            if not bbox or len(bbox) != 4:
                continue

            # Find image info
            img_info = None
            for img in coco_data.get('training_images', []):
                if img['id'] == img_id:
                    img_info = img
                    break
            if not img_info:
                continue

            img_path = os.path.join(images_folder, img_info['file_name'])
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            x, y, w, h = map(int, bbox)
            roi = img[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            # Extract HOG features
            features = extract_hog_features(roi)
            X.append(features)
            y.append(1)  # Positive sample (pellet)

        if len(X) == 0:
            print("No valid training samples found.")
            return False

        # Add negative samples (random non-pellet regions)
        for img_info in coco_data.get('training_images', []):
            img_path = os.path.join(images_folder, img_info['file_name'])
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            for _ in range(3):  # 3 negative patches per image
                nx = np.random.randint(0, w - 64)
                ny = np.random.randint(0, h - 64)
                patch = img[ny:ny+64, nx:nx+64]
                if patch.shape[0] < 64 or patch.shape[1] < 64:
                    continue
                features = extract_hog_features(patch)
                X.append(features)
                y.append(0)  # Negative

        X = np.array(X)
        y = np.array(y)

        print(f"Training SVM with {len(X)} samples ({sum(y)} positive, {len(y)-sum(y)} negative)...")
        detector = svm.SVC(kernel='linear', probability=True, C=1.0, random_state=42)
        detector.fit(X, y)

        is_trained = True
        trained_samples = sum(y)
        print(f"Training complete. {trained_samples} pellet samples trained.")
        return True

    except Exception as e:
        print(f"Error during training: {e}")
        return False


# ----------------------------------------------------------------------
# Predict if ROI is a pellet
# ----------------------------------------------------------------------
def is_pellet_roi(roi):
    if not is_trained or detector is None:
        return False

    try:
        features = extract_hog_features(roi)
        features = features.reshape(1, -1)
        prob = detector.predict_proba(features)[0]
        confidence = prob[1]  # Probability of being pellet
        return confidence > 0.7  # Threshold
    except:
        return False


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
# Detection with Trained Classifier
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

        # Extract ROI
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        # Resize for consistency
        roi_resized = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)

        # Classify using trained model
        if not is_pellet_roi(roi_resized):
            continue  # Not a pellet

        # Optional: match to reference polygon (from original)
        ref_poly = match_to_reference_polygon(cnt, frame.shape)

        pellets.append({
            'x': x, 'y': y, 'w': w, 'h': h,
            'diameter': diameter,
            'length': length,
            'within_tolerance': is_within_tolerance(diameter, length),
            'contour': cnt,
            'reference_polygon': ref_poly
        })
    return pellets


# ----------------------------------------------------------------------
# Polygon Matching (Kept from original)
# ----------------------------------------------------------------------
reference_polygons = []

def match_to_reference_polygon(contour, frame_shape):
    if not reference_polygons:
        return None

    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    min_dist = float('inf')
    closest_poly = None

    for ref_poly in reference_polygons:
        poly = ref_poly['polygon']
        ref_M = cv2.moments(poly)
        if ref_M["m00"] == 0:
            continue
        ref_cx = int(ref_M["m10"] / ref_M["m00"])
        ref_cy = int(ref_M["m01"] / ref_M["m00"])
        dist = np.sqrt((cx - ref_cx) ** 2 + (cy - ref_cy) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest_poly = ref_poly

    if min_dist < 50:
        return closest_poly
    return None


# ----------------------------------------------------------------------
# Load Reference Polygons (from COCO)
# ----------------------------------------------------------------------
def load_reference_polygons(json_path="pellets_label.json"):
    global reference_polygons
    if not os.path.exists(json_path):
        return False

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        for ann in data.get('annotations', []):
            if 'segmentation' in ann and ann['segmentation']:
                for seg in ann['segmentation']:
                    polygon = np.array(seg).reshape(-1, 2).astype(np.int32)
                    reference_polygons.append({
                        'polygon': polygon,
                        'bbox': ann.get('bbox', []),
                        'category_id': ann.get('category_id', 1)
                    })
        print(f"Loaded {len(reference_polygons)} reference polygons.")
        return True
    except Exception as e:
        print(f"Error loading polygons: {e}")
        return False


# ----------------------------------------------------------------------
# Draw Calibration Mode
# ----------------------------------------------------------------------
def draw_calibration_mode(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (PANEL_X, PANEL_Y), (PANEL_X + PANEL_W, PANEL_Y + PANEL_H),
                  (30, 30, 50), -1)
    cv2.rectangle(overlay, (PANEL_X, PANEL_Y), (PANEL_X + PANEL_W, PANEL_Y + PANEL_H),
                  (100, 150, 255), 3)

    cv2.putText(overlay, "CALIBRATION MODE", (PANEL_X + 15, PANEL_Y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"{PIXELS_PER_MM:.2f} px/mm", (PANEL_X + 15, PANEL_Y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.rectangle(overlay, (UP_BTN[0], UP_BTN[1]), (UP_BTN[0] + UP_BTN[2], UP_BTN[1] + UP_BTN[3]),
                  (0, 200, 0), -1)
    cv2.putText(overlay, "UP", (UP_BTN[0] + 12, UP_BTN[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.rectangle(overlay, (DOWN_BTN[0], DOWN_BTN[1]), (DOWN_BTN[0] + DOWN_BTN[2], DOWN_BTN[1] + DOWN_BTN[3]),
                  (200, 0, 0), -1)
    cv2.putText(overlay, "DOWN", (DOWN_BTN[0] + 5, DOWN_BTN[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.rectangle(overlay, (BACK_BTN[0], BACK_BTN[1]), (BACK_BTN[0] + BACK_BTN[2], BACK_BTN[1] + BACK_BTN[3]),
                  (100, 100, 100), -1)
    cv2.putText(overlay, "BACK", (BACK_BTN[0] + 10, BACK_BTN[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)


# ----------------------------------------------------------------------
# Main Overlay
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

        if p['reference_polygon']:
            polygon = p['reference_polygon']['polygon']
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], (255, 255, 0))
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.polylines(frame, [polygon], True, (0, 255, 255), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        bg_y = max(y - 45, 0)
        cv2.rectangle(frame, (x, bg_y), (x + 120, y - 5), (0, 0, 0), -1)
        cv2.putText(frame, f"D: {p['diameter']:.2f}", (x + 5, bg_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"L: {p['length']:.2f}", (x + 5, bg_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if not p['within_tolerance']:
            cv2.circle(frame, (x + w - 10, y + 10), 8, (0, 0, 255), -1)
            cv2.putText(frame, "!", (x + w - 14, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    status = "TRAINED" if is_trained else "NO MODEL"
    hint_text = f"Press 'c' for calibration | {status}: {trained_samples} samples"
    if reference_polygons:
        hint_text += f" | {len(reference_polygons)} ref polygons"
    cv2.putText(frame, hint_text,
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)

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

    print("\nPellet Inspector + Trained Recognition (HOG+SVM)")
    print("=" * 60)

    # Train model
    success = train_from_coco("pellets_label.json", "training_images")
    if not success:
        print("Continuing without trained model (will use contours only).")

    # Load reference polygons for overlay
    load_reference_polygons("pellets_label.json")

    print("Press 'c' → Calibration | 'q' → Quit")
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