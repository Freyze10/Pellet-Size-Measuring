import cv2
import numpy as np
import cv2.aruco as aruco

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
REAL_WIDTH_MM = 80.0
REAL_HEIGHT_MM = 80.0
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720


class PrecisionMeasure:
    def __init__(self):
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.matrix = None
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

        # 2. Pre-process (Grayscale)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Threshold (The most critical part)
        # Using OTSU to automatically separate dark objects from light background
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 4. Watershed
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
            if mark <= 1: continue
            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[markers == mark] = 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            cnt = contours[0]

            # SIZE FILTER: Check if pellet is being deleted here
            area = cv2.contourArea(cnt)
            min_area = (1.0 * self.px_per_mm) ** 2  # Reduced filter size

            if area < min_area:
                # Uncomment to debug small noise
                # print(f"Ignored small blob: {area:.2f}")
                continue

            if len(cnt) >= 5:
                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                dim1, dim2 = ma / self.px_per_mm, MA / self.px_per_mm
                length_mm, diameter_mm = max(dim1, dim2), min(dim1, dim2)
                is_good = (abs(diameter_mm - TARGET_DIAMETER) <= TOLERANCE and abs(
                    length_mm - TARGET_LENGTH) <= TOLERANCE)
                results.append({'pos': (x, y), 'shape': ((x, y), (MA, ma), angle), 'd': diameter_mm, 'l': length_mm,
                                'ok': is_good})

        # Return the 'thresh' image so user can see what computer sees
        return warped, results, thresh


def draw_results(image, results):
    output = image.copy()
    for r in results:
        (x, y), (MA, ma), angle = r['shape']
        color = (0, 255, 0) if r['ok'] else (0, 0, 255)
        cv2.ellipse(output, ((x, y), (MA, ma), angle), color, 2)
        label = f"{r['d']:.2f}x{r['l']:.2f}"
        cv2.putText(output, label, (int(x) - 30, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
            # Get the threshold image for debugging
            warped, pellets, debug_thresh = engine.analyze_pellets(frame)
            output_view = draw_results(warped, pellets)

            # SHOW 3 WINDOWS
            cv2.imshow("1. RESULT (Top-Down)", output_view)

            # THIS IS THE IMPORTANT WINDOW:
            # White = Object, Black = Background.
            # If your pellet is Black here, the computer can't see it.
            if debug_thresh is not None:
                cv2.imshow("2. X-RAY (Threshold)", debug_thresh)

            aruco.drawDetectedMarkers(frame, corners, ids)
        else:
            cv2.putText(frame, "ALIGN MARKERS", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("3. RAW CAMERA", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()