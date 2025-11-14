import cv2
import numpy as np
import time
import sys
import json
import os

# ----------------------------------------------------------------------
# Global Calibration
# ----------------------------------------------------------------------
PIXELS_PER_MM = 6.0
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 200.0

# ----------------------------------------------------------------------
# SIFT-based Pellet Detector (exact copy of the PyQt class)
# ----------------------------------------------------------------------
class PelletDetector:
    """Robust pellet detector using SIFT + multi-scale + contour fallback"""

    def __init__(self):
        self.trained_samples = []
        self.feature_detector = cv2.SIFT_create(nfeatures=500)
        self.bf_matcher = cv2.BFMatcher()

    def train_from_coco(self, coco_data, images_folder=""):
        self.trained_samples = []
        images_dict = {img['id']: img for img in coco_data.get('images', [])}

        for ann in coco_data.get('annotations', []):
            image_id = ann['image_id']
            if image_id not in images_dict:
                continue

            img_info = images_dict[image_id]
            img_path = os.path.join(images_folder, img_info['file_name'])
            if not os.path.exists(img_path):
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue

            if 'segmentation' in ann and ann['segmentation']:
                for seg in ann['segmentation']:
                    polygon = np.array(seg).reshape(-1, 2).astype(np.int32)
                    x, y, w, h = cv2.boundingRect(polygon)

                    padding = 15
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(image.shape[1], x + w + padding)
                    y2 = min(image.shape[0], y + h + padding)

                    pellet_sample = image[y1:y2, x1:x2].copy()
                    if pellet_sample.size == 0:
                        continue

                    mask = np.zeros(pellet_sample.shape[:2], dtype=np.uint8)
                    polygon_shifted = polygon - [x1, y1]
                    cv2.fillPoly(mask, [polygon_shifted], 255)

                    gray_sample = cv2.cvtColor(pellet_sample, cv2.COLOR_BGR2GRAY)
                    kp, desc = self.feature_detector.detectAndCompute(gray_sample, mask)

                    if desc is not None and len(kp) > 10:
                        self.trained_samples.append({
                            'image': pellet_sample,
                            'gray': gray_sample,
                            'mask': mask,
                            'keypoints': kp,
                            'descriptors': desc,
                            'size': (w, h),
                            'polygon': polygon_shifted,
                            'bbox': (x1, y1, x2 - x1, y2 - y1)
                        })

        print(f"SIFT training finished – {len(self.trained_samples)} samples")
        return len(self.trained_samples) > 0

    # --------------------------------------------------------------
    # Detection (multi-scale SIFT + contour fallback)
    # --------------------------------------------------------------
    def detect_pellets(self, image):
        if not self.trained_samples:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = []

        # ---- multi-scale SIFT matching ---------------------------------
        scales = [0.8, 1.0, 1.2, 1.4]
        for scale in scales:
            if scale != 1.0:
                h, w = gray.shape
                resized = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            else:
                resized = gray

            kp2, desc2 = self.feature_detector.detectAndCompute(resized, None)
            if desc2 is None:
                continue

            for sample in self.trained_samples:
                matches = self.bf_matcher.knnMatch(sample['descriptors'], desc2, k=2)

                good = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.8 * n.distance]
                if len(good) < 12:
                    continue

                src_pts = np.float32([sample['keypoints'][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                if scale != 1.0:
                    dst_pts /= scale

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                if M is None:
                    continue

                h, w = sample['gray'].shape
                pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                poly = dst.reshape(-1, 2).astype(np.int32)

                x, y, bw, bh = cv2.boundingRect(poly)
                if bw < 20 or bh < 20:
                    continue

                # deduplicate across scales
                if any(self._iou((x, y, bw, bh), d['bbox']) > 0.5 for d in detections):
                    continue

                detections.append({
                    'bbox': (x, y, bw, bh),
                    'polygon': poly,
                    'confidence': len(good),
                    'method': 'sift'
                })

        # ---- contour fallback (shape similarity) -----------------------
        contour_dets = self._contour_fallback(image, gray)
        for d in contour_dets:
            if not any(self._iou(d['bbox'], ex['bbox']) > 0.5 for ex in detections):
                detections.append(d)

        # ---- final NMS -------------------------------------------------
        detections = self._nms(detections, overlap_thr=0.4)
        return detections

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1 = max(x1, x2); yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2); yi2 = min(y1 + h1, y2 + h2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0

    def _contour_fallback(self, image, gray):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.trained_samples:
            sizes = [s['size'] for s in self.trained_samples]
            avg_area = np.mean([w * h for w, h in sizes])
            min_area, max_area = avg_area * 0.4, avg_area * 3.0
        else:
            min_area, max_area = 100, 10000

        dets = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 20 or h < 20:
                continue
            aspect = max(w, h) / min(w, h)
            if not (1.0 <= aspect <= 3.5):
                continue

            score = self._shape_score(cnt)
            if score > 0.6:
                dets.append({
                    'bbox': (x, y, w, h),
                    'polygon': cnt.reshape(-1, 2),
                    'confidence': int(score * 100),
                    'method': 'contour'
                })
        return dets

    def _shape_score(self, contour):
        if not self.trained_samples:
            return 0.0
        scores = []
        for s in self.trained_samples:
            ref = s['polygon'].reshape(-1, 1, 2)
            try:
                sim = cv2.matchShapes(contour, ref, cv2.CONTOURS_MATCH_I1, 0)
                scores.append(1.0 / (1.0 + sim))
            except:
                pass
        return max(scores) if scores else 0.0

    def _nms(self, detections, overlap_thr=0.4):
        if not detections:
            return []
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []
        for d in detections:
            if not any(self._iou(d['bbox'], k['bbox']) > overlap_thr for k in keep):
                keep.append(d)
        return keep


# ----------------------------------------------------------------------
# Global detector instance
# ----------------------------------------------------------------------
detector = PelletDetector()
is_trained = False
trained_samples = 0

# ----------------------------------------------------------------------
# Calibration helpers
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

# ----------------------------------------------------------------------
# Calibration panel UI
# ----------------------------------------------------------------------
in_calib_mode = False
PANEL_X, PANEL_Y = 10, 300
PANEL_W, PANEL_H = 300, 180
UP_BTN = (PANEL_X + 230, PANEL_Y + 40, 50, 40)
DOWN_BTN = (PANEL_X + 230, PANEL_Y + 90, 50, 40)
BACK_BTN = (PANEL_X + 20, PANEL_Y + 130, 80, 35)

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
# Measurement from exact polygon (rotated min-area rect)
# ----------------------------------------------------------------------
def rotated_rect_dimensions(polygon):
    rect = cv2.minAreaRect(polygon.astype(np.float32))
    return rect[1]                     # (width, height) in pixels

def measure_pellet(polygon, pixels_per_mm):
    w_px, h_px = rotated_rect_dimensions(polygon)
    width_mm  = w_px / pixels_per_mm
    height_mm = h_px / pixels_per_mm
    diameter = min(width_mm, height_mm)
    length   = max(width_mm, height_mm)

    within = (
        (TARGET_DIAMETER - TOLERANCE <= diameter <= TARGET_DIAMETER + TOLERANCE) and
        (TARGET_LENGTH   - TOLERANCE <= length   <= TARGET_LENGTH   + TOLERANCE)
    )
    return {"diameter": diameter, "length": length, "within": within}


# ----------------------------------------------------------------------
# Reference-polygon overlay (optional)
# ----------------------------------------------------------------------
reference_polygons = []

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
                    poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                    reference_polygons.append({'polygon': poly})
        print(f"Loaded {len(reference_polygons)} reference polygons.")
        return True
    except Exception as e:
        print(f"Error loading reference polygons: {e}")
        return False


# ----------------------------------------------------------------------
# Main detection (calls the SIFT detector)
# ----------------------------------------------------------------------
def detect_pellets(frame):
    if not is_trained:
        return []                     # fall-back to nothing if training failed

    detections = detector.detect_pellets(frame)

    pellets = []
    for idx, det in enumerate(detections, 1):
        poly = det['polygon']
        meas = measure_pellet(poly, PIXELS_PER_MM)

        # find nearest reference polygon (if any) – just for visual overlay
        ref_poly = None
        if reference_polygons:
            M = cv2.moments(poly)
            cx = int(M["m10"] / M["m00"]) if M["m00"] else int(poly[:, 0].mean())
            cy = int(M["m01"] / M["m00"]) if M["m00"] else int(poly[:, 1].mean())
            best_dist = float('inf')
            for rp in reference_polygons:
                rm = cv2.moments(rp['polygon'])
                rcx = int(rm["m10"] / rm["m00"]) if rm["m00"] else int(rp['polygon'][:, 0].mean())
                rcy = int(rm["m01"] / rm["m00"]) if rm["m00"] else int(rp['polygon'][:, 1].mean())
                d = (cx - rcx) ** 2 + (cy - rcy) ** 2
                if d < best_dist:
                    best_dist, ref_poly = d, rp

        pellets.append({
            'id': idx,
            'polygon': poly,
            'bbox': det['bbox'],
            'diameter': meas['diameter'],
            'length': meas['length'],
            'within_tolerance': meas['within'],
            'confidence': det['confidence'],
            'reference_polygon': ref_poly
        })
    return pellets


# ----------------------------------------------------------------------
# Overlay drawing (same look as original script)
# ----------------------------------------------------------------------
def draw_overlay(frame, pellets):
    total = len(pellets)
    within = sum(p['within_tolerance'] for p in pellets)
    out = total - within
    status_text = f"In: {within}   Out: {out}   Total: {total}"
    status_color = (0, 255, 0) if out == 0 else (0, 0, 255)

    cv2.rectangle(frame, (10, 10), (460, 50), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (460, 50), status_color, 2)
    cv2.putText(frame, status_text, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

    for p in pellets:
        poly = p['polygon']
        color = (0, 255, 0) if p['within_tolerance'] else (0, 0, 255)

        # optional reference polygon overlay
        if p['reference_polygon']:
            rp = p['reference_polygon']['polygon']
            ov = frame.copy()
            cv2.fillPoly(ov, [rp], (255, 255, 0))
            cv2.addWeighted(ov, 0.3, frame, 0.7, 0, frame)
            cv2.polylines(frame, [rp], True, (0, 255, 255), 2)

        # exact mask fill + outline
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly.reshape(-1, 1, 2)], color)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(frame, [poly.reshape(-1, 1, 2)], True, color, 2)

        # centered ID
        M = cv2.moments(poly)
        cx = int(M["m10"] / M["m00"]) if M["m00"] else int(poly[:, 0].mean())
        cy = int(M["m01"] / M["m00"]) if M["m00"] else int(poly[:, 1].mean())
        cv2.putText(frame, str(p['id']), (cx - 12, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # size info
        bg_y = max(cy - 55, 0)
        cv2.rectangle(frame, (cx - 70, bg_y), (cx + 70, bg_y + 40), (0, 0, 0), -1)
        cv2.putText(frame, f"D:{p['diameter']:.2f}", (cx - 65, bg_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"L:{p['length']:.2f}", (cx - 65, bg_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if not p['within_tolerance']:
            cv2.circle(frame, (cx + 55, cy - 30), 8, (0, 0, 255), -1)
            cv2.putText(frame, "!", (cx + 51, cy - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    hint = f"Press 'c' for calibration | SIFT: {trained_samples} samples"
    if reference_polygons:
        hint += f" | {len(reference_polygons)} ref polys"
    cv2.putText(frame, hint,
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)

    if in_calib_mode:
        draw_calibration_mode(frame)

    return frame


# ----------------------------------------------------------------------
# Camera handling
# ----------------------------------------------------------------------
def get_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main():
    global in_calib_mode, is_trained, trained_samples

    print("\n=== Pellet Live Inspector (SIFT + exact mask) ===")
    print("=" * 55)

    # ---- train the SIFT model ------------------------------------------------
    json_path = "pellets_label.json"
    img_folder = "training_images"
    if os.path.exists(json_path) and os.path.isdir(img_folder):
        with open(json_path, 'r') as f:
            coco = json.load(f)
        is_trained = detector.train_from_coco(coco, img_folder)
        trained_samples = len(detector.trained_samples)
    else:
        print("pellets_label.json or training_images folder missing – running without model.")
        is_trained = False

    # ---- load reference polygons for visual overlay -------------------------
    load_reference_polygons(json_path)

    print("Press 'c' → calibration | 'q' → quit")
    print("=" * 55)

    cap = get_camera()
    if not cap.isOpened():
        print("Cannot open camera.")
        sys.exit(1)

    fps_cnt = 0
    fps_start = time.time()
    fps = 0

    win = "Pellet Live Inspector"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera lost – reconnecting...")
            cap.release()
            time.sleep(1)
            cap = get_camera()
            continue

        disp = frame.copy()
        pellets = detect_pellets(disp)
        disp = draw_overlay(disp, pellets)

        # FPS
        fps_cnt += 1
        if time.time() - fps_start >= 1.0:
            fps = fps_cnt
            fps_cnt = 0
            fps_start = time.time()
        cv2.putText(disp, f"FPS: {fps}", (disp.shape[1] - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(win, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c') and not in_calib_mode:
            in_calib_mode = True

        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Shutdown complete.")


if __name__ == "__main__":
    main()