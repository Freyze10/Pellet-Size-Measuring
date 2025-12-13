import cv2
import numpy as np
import cv2.aruco as aruco

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
REAL_WIDTH_MM = 80.0
REAL_HEIGHT_MM = 80.0
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720


# ----------------------------------------------------------------------
# Debug Engine
# ----------------------------------------------------------------------
class PrecisionMeasure:
    def __init__(self):
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.matrix = None

        # Scale: 10 pixels = 1 millimeter
        self.scale_factor = 10
        self.width_px = int(REAL_WIDTH_MM * self.scale_factor)
        self.height_px = int(REAL_HEIGHT_MM * self.scale_factor)
        self.px_per_mm = self.scale_factor

    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) < 4: return False, corners, ids
        id_map = {}
        for i, id_val in enumerate(ids):
            c = np.mean(corners[i][0], axis=0)
            id_map[id_val[0]] = c
        if not all(k in id_map for k in [0, 1, 2, 3]): return False, corners, ids
        src_pts = np.float32([id_map[0], id_map[1], id_map[2], id_map[3]])
        dst_pts = np.float32([[0, 0], [self.width_px, 0], [self.width_px, self.height_px], [0, self.height_px]])
        self.matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return True, corners, ids

    def analyze_pellets(self, frame):
        if self.matrix is None: return None, [], None

        # 1. Warp
        warped = cv2.warpPerspective(frame, self.matrix, (self.width_px, self.height_px))

        # 2. Threshold
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Assuming LIGHT Background and DARK Pellets.
        # If Pellets are WHITE and background BLACK, remove "cv2.THRESH_BINARY_INV +"
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. Simple Contour Detection (No Watershed)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []

        print(f"\n--- Frame Scan ---")
        print(f"Contours found: {len(contours)}")

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)

            # --- DEBUG INFO ---
            # We calculate what the size is in mm^2 just to show you
            area_mm = area / (self.px_per_mm ** 2)

            # This is the filter. I lowered it to ALMOST ZERO (10 pixels)
            # A 3mm pellet should be around 700 pixels.
            if area < 20:
                print(f"#{i}: IGNORED (Too small: {area:.0f} px)")
                continue

            if len(cnt) >= 5:
                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)

                dim1 = ma / self.px_per_mm
                dim2 = MA / self.px_per_mm

                # Ellipse logic: shortest side is diameter, longest is length
                length_mm = max(dim1, dim2)
                diameter_mm = min(dim1, dim2)

                print(f"#{i}: MEASURED! D:{diameter_mm:.2f}mm L:{length_mm:.2f}mm (Area: {area:.0f} px)")

                results.append({
                    'pos': (x, y),
                    'shape': ((x, y), (MA, ma), angle),
                    'd': diameter_mm,
                    'l': length_mm,
                    'ok': True
                })
            else:
                print(f"#{i}: IGNORED (Shape weird, points: {len(cnt)})")

        return warped, results, thresh


def draw_results(image, results):
    output = image.copy()
    for r in results:
        (x, y), (MA, ma), angle = r['shape']
        cv2.ellipse(output, ((x, y), (MA, ma), angle), (0, 255, 0), 2)
        label = f"{r['d']:.2f}x{r['l']:.2f}"
        cv2.putText(output, label, (int(x) - 30, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return output


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    engine = PrecisionMeasure()

    while True:
        ret, frame = cap.read()
        if not ret: break

        is_calibrated, corners, ids = engine.detect_markers(frame)

        if is_calibrated:
            warped, pellets, debug_thresh = engine.analyze_pellets(frame)
            output_view = draw_results(warped, pellets)

            cv2.imshow("1. MEASUREMENT", output_view)
            cv2.imshow("2. X-RAY (White = Pellet)", debug_thresh)
            aruco.drawDetectedMarkers(frame, corners, ids)
        else:
            cv2.putText(frame, "ALIGN MARKERS", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("3. RAW CAMERA", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()