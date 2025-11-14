import cv2
import numpy as np
import time
import sys
import json
from pathlib import Path

# ----------------------------------------------------------------------
# Global Calibration
# ----------------------------------------------------------------------
PIXELS_PER_MM = 6.0
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 200.0

# ML Model parameters
ANNOTATION_FILE = "pellets_label.json"
annotations_data = None
template_features = []


def load_annotations():
    """Load COCO format annotations from MakeSense.ai"""
    global annotations_data, template_features

    try:
        with open(ANNOTATION_FILE, 'r') as f:
            annotations_data = json.load(f)

        print(f"✓ Loaded annotations: {len(annotations_data.get('annotations', []))} objects")
        print(f"✓ Images in dataset: {len(annotations_data.get('images', []))}")
        print(f"✓ Categories: {[cat['name'] for cat in annotations_data.get('categories', [])]}")

        # Extract features from annotated pellets for template matching
        extract_template_features()
        return True
    except FileNotFoundError:
        print(f"⚠ Warning: {ANNOTATION_FILE} not found. Using traditional detection.")
        return False
    except json.JSONDecodeError as e:
        print(f"⚠ Error parsing JSON: {e}")
        return False


def extract_template_features():
    """Extract features from annotated pellets to use as templates"""
    global template_features

    if not annotations_data:
        return

    # Create a mapping of image_id to annotations
    image_map = {img['id']: img for img in annotations_data.get('images', [])}

    for ann in annotations_data.get('annotations', []):
        if 'segmentation' in ann and ann['segmentation']:
            # Get polygon points
            segmentation = ann['segmentation'][0]
            points = np.array(segmentation).reshape(-1, 2)

            # Calculate bounding box from polygon
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            y_min, y_max = int(y_coords.min()), int(y_coords.max())

            width = x_max - x_min
            height = y_max - y_min

            # Store template feature
            template_features.append({
                'width': width,
                'height': height,
                'aspect_ratio': width / height if height > 0 else 1.0,
                'area': ann.get('area', width * height),
                'points': points,
                'bbox': (x_min, y_min, width, height)
            })

    print(f"✓ Extracted {len(template_features)} template features")


def match_to_templates(contour, bbox):
    """Check if detected contour matches learned templates"""
    if not template_features:
        return True, 1.0  # No templates, accept all

    x, y, w, h = bbox
    aspect_ratio = w / h if h > 0 else 1.0
    area = cv2.contourArea(contour)

    best_match_score = 0.0

    for template in template_features:
        # Compare aspect ratio
        aspect_diff = abs(aspect_ratio - template['aspect_ratio'])
        aspect_score = max(0, 1.0 - aspect_diff)

        # Compare size
        size_ratio = min(w, template['width']) / max(w, template['width'])
        size_score = size_ratio

        # Compare area
        area_ratio = min(area, template['area']) / max(area, template['area'])
        area_score = area_ratio

        # Combined score
        match_score = (aspect_score * 0.3 + size_score * 0.4 + area_score * 0.3)
        best_match_score = max(best_match_score, match_score)

    # Threshold for accepting a match
    return best_match_score > 0.5, best_match_score


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
show_templates = False

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
# ML-Enhanced Detection
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

        # ML-based template matching
        is_match, confidence = match_to_templates(cnt, (x, y, w, h))

        if not is_match:
            continue  # Skip if doesn't match learned templates

        width_mm = w / PIXELS_PER_MM
        height_mm = h / PIXELS_PER_MM
        diameter = min(width_mm, height_mm)
        length = max(width_mm, height_mm)

        if should_process_pellet(diameter, length):
            # Approximate polygon for smoother contour
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            pellets.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'diameter': diameter,
                'length': length,
                'within_tolerance': is_within_tolerance(diameter, length),
                'contour': approx,
                'confidence': confidence
            })
    return pellets


# ----------------------------------------------------------------------
# Draw Template Visualization
# ----------------------------------------------------------------------
def draw_templates(frame):
    """Draw learned template shapes for reference"""
    if not template_features or not show_templates:
        return

    start_x = frame.shape[1] - 150
    start_y = 100

    cv2.rectangle(frame, (start_x - 10, start_y - 30),
                  (start_x + 140, start_y + len(template_features) * 60 + 10),
                  (40, 40, 40), -1)
    cv2.putText(frame, "Templates", (start_x, start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    for i, template in enumerate(template_features[:5]):  # Show first 5
        y_offset = start_y + i * 60

        # Draw mini rectangle representing template
        scale = 0.3
        tw = int(template['width'] * scale)
        th = int(template['height'] * scale)

        cv2.rectangle(frame, (start_x, y_offset),
                      (start_x + tw, y_offset + th), (0, 255, 255), 1)

        cv2.putText(frame, f"{template['width']}x{template['height']}px",
                    (start_x + tw + 5, y_offset + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)


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
# Main Overlay with Polygon Drawing
# ----------------------------------------------------------------------
def draw_overlay(frame, pellets):
    # Status bar
    total = len(pellets)
    within = sum(1 for p in pellets if p['within_tolerance'])
    out_of = total - within

    ml_status = "ML-Enhanced" if template_features else "Traditional"
    status_text = f"{ml_status} | In: {within} Out: {out_of} Total: {total}"
    status_color = (0, 255, 0) if out_of == 0 else (0, 0, 255)

    cv2.rectangle(frame, (10, 10), (560, 50), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (560, 50), status_color, 2)
    cv2.putText(frame, status_text, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

    # Draw pellets with polygons
    for p in pellets:
        x, y, w, h = p['x'], p['y'], p['w'], p['h']
        color = (0, 255, 0) if p['within_tolerance'] else (0, 0, 255)

        # Draw polygon contour instead of rectangle
        if 'contour' in p:
            cv2.drawContours(frame, [p['contour']], -1, color, 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Info background
        bg_y = max(y - 55, 0)
        cv2.rectangle(frame, (x, bg_y), (x + 130, y - 5), (0, 0, 0), -1)

        # Display measurements
        cv2.putText(frame, f"D: {p['diameter']:.2f}mm", (x + 5, bg_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(frame, f"L: {p['length']:.2f}mm", (x + 5, bg_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Confidence score
        if 'confidence' in p:
            conf_color = (0, 255, 0) if p['confidence'] > 0.7 else (255, 255, 0)
            cv2.putText(frame, f"Conf: {p['confidence']:.2f}", (x + 5, bg_y + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, conf_color, 1)

        # Warning indicator
        if not p['within_tolerance']:
            cv2.circle(frame, (x + w - 10, y + 10), 8, (0, 0, 255), -1)
            cv2.putText(frame, "!", (x + w - 14, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw template visualization
    draw_templates(frame)

    # Calibration hint
    if not in_calib_mode:
        hint_text = "c:Calib | t:Templates" if template_features else "c:Calib"
        cv2.putText(frame, hint_text, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)

    # Draw calibration panel
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
    global in_calib_mode, show_templates

    print("\n" + "=" * 60)
    print("ML-ENHANCED PELLET INSPECTOR")
    print("=" * 60)

    # Load ML annotations
    has_ml = load_annotations()

    print("\nControls:")
    print("  'c' → Calibration mode")
    print("  't' → Toggle template visualization")
    print("  'q' → Quit")
    print("=" * 60 + "\n")

    cap = get_camera()
    if not cap.isOpened():
        print("❌ Cannot open camera.")
        sys.exit(1)

    fps_counter = 0
    fps_start = time.time()
    fps_display = 0

    window_name = "ML Pellet Size Measurement"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Camera lost – reconnecting...")
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
        if key == ord('t'):
            show_templates = not show_templates

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Shutdown complete.")


if __name__ == "__main__":
    main()