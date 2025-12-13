import cv2
import numpy as np
import cv2.aruco as aruco

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Real-world dimensions of the printed ArUco Sheet (Inner rectangle)
REAL_WIDTH_MM = 80.0
REAL_HEIGHT_MM = 60.0

# Target Pellet Dimensions (mm)
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5

# Camera Settings
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720


# ----------------------------------------------------------------------
# ArUco & Measurement Engine
# ----------------------------------------------------------------------
class PrecisionMeasure:
    def __init__(self):
        # ArUco Configuration
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.matrix = None
        self.width_px = 0
        self.height_px = 0
        self.px_per_mm = 0

        # Calculate pixel dimensions for the warped view (10 px per mm)
        self.scale_factor = 10
        self.width_px = int(REAL_WIDTH_MM * self.scale_factor)
        self.height_px = int(REAL_HEIGHT_MM * self.scale_factor)
        self.px_per_mm = self.scale_factor

    def detect_markers(self, frame):
        """Detects markers and calculates the perspective matrix."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) < 4:
            return False, corners, ids

        # Map IDs to centers
        id_map = {}
        for i, id_val in enumerate(ids):
            c = np.mean(corners[i][0], axis=0)
            id_map[id_val[0]] = c

        if not all(k in id_map for k in [0, 1, 2, 3]):
            return False, corners, ids

        # Source points (from camera)
        src_pts = np.float32([id_map[0], id_map[1], id_map[2], id_map[3]])

        # Destination points (flat top-down view)
        dst_pts = np.float32([
            [0, 0],
            [self.width_px, 0],
            [self.width_px, self.height_px],
            [0, self.height_px]
        ])

        self.matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return True, corners, ids

    def analyze_pellets(self, frame):
        """Warps image and measures pellets."""
        if self.matrix is None:
            return None, []

        # 1. Warp (Flatten)
        warped = cv2.warpPerspective(frame, self.matrix, (self.width_px, self.height_px))

        # 2. Pre-process
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. Watershed Separation
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(warped, markers)

        results = []
        unique_markers = np.unique(markers)

        for mark in unique_markers:
            if mark <= 1: continue  # Skip background/boundaries

            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[markers == mark] = 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours: continue
            cnt = contours[0]

            # Area filter (ignore tiny noise)
            if cv2.contourArea(cnt) < (1.5 * self.px_per_mm) ** 2: continue

            # Fit Ellipse for sub-pixel accuracy
            if len(cnt) >= 5:
                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)

                # Convert pixels to mm
                dim1 = ma / self.px_per_mm
                dim2 = MA / self.px_per_mm

                length_mm = max(dim1, dim2)
                diameter_mm = min(dim1, dim2)

                is_good = (abs(diameter_mm - TARGET_DIAMETER) <= TOLERANCE and
                           abs(length_mm - TARGET_LENGTH) <= TOLERANCE)

                results.append({
                    'pos': (x, y),
                    'shape': ((x, y), (MA, ma), angle),
                    'd': diameter_mm,
                    'l': length_mm,
                    'ok': is_good
                })

        return warped, results


# ----------------------------------------------------------------------
# UI Drawing
# ----------------------------------------------------------------------
def draw_results(image, results):
    """Draws measurements on the warped image."""
    output = image.copy()

    good_count = 0
    bad_count = 0

    for r in results:
        (x, y), (MA, ma), angle = r['shape']
        color = (0, 255, 0) if r['ok'] else (0, 0, 255)

        if r['ok']:
            good_count += 1
        else:
            bad_count += 1

        # Draw ellipse
        cv2.ellipse(output, ((x, y), (MA, ma), angle), color, 2)

        # Draw text
        label = f"{r['d']:.2f}x{r['l']:.2f}"
        cv2.putText(output, label, (int(x) - 30, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)  # Outline
        cv2.putText(output, label, (int(x) - 30, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Status Bar
    h, w = output.shape[:2]
    cv2.rectangle(output, (0, 0), (w, 50), (30, 30, 30), -1)
    cv2.putText(output, f"TOTAL: {len(results)} | GOOD: {good_count} | BAD: {bad_count}",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return output


# ----------------------------------------------------------------------
# Main Application
# ----------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # ... (Keep your camera settings here) ...

    engine = PrecisionMeasure()

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Detect Markers
        is_calibrated, corners, ids = engine.detect_markers(frame)

        if is_calibrated:
            # 2. Get the Flattened View IMMEDIATELY (for debugging)
            warped, pellets = engine.analyze_pellets(frame)

            # 3. Draw results on the flattened view
            output_view = draw_results(warped, pellets)

            # Show the "Top-Down" view so you can see if it is twisted
            cv2.imshow("DEBUG VIEW (Top-Down)", output_view)

            # Draw markers on raw frame
            aruco.drawDetectedMarkers(frame, corners, ids)
        else:
            cv2.putText(frame, "Need IDs: 0(TL), 1(TR), 2(BR), 3(BL)", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Raw Camera", frame)

        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()