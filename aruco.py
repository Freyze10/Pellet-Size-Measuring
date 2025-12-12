import cv2
import numpy as np
import cv2.aruco as aruco

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Real-world dimensions for the 80x60mm calibration sheet
REAL_WIDTH_MM = 80.0
REAL_HEIGHT_MM = 60.0

# Target Pellet Dimensions (mm)
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5


# ----------------------------------------------------------------------
# Advanced Measurement Class (Updated for OpenCV 4.7+)
# ----------------------------------------------------------------------
class PrecisionMeasure:
    def __init__(self):
        # NEW API: Define dictionary and parameters
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()

        # NEW API: Create the Detector object
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.matrix = None  # Perspective matrix
        self.width_px = 0
        self.height_px = 0
        self.px_per_mm = 0

    def calibrate_perspective(self, frame):
        """
        Finds 4 ArUco markers (IDs 0,1,2,3) to define the workspace.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # NEW API: Use the detector object instead of cv2.aruco.detectMarkers
        corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is None or len(ids) < 4:
            return False, frame, corners, ids

        # Map IDs to centers
        id_map = {}
        for i, id_val in enumerate(ids):
            c = np.mean(corners[i][0], axis=0)
            id_map[id_val[0]] = c

        # Check if we have all 4 required markers
        if not all(k in id_map for k in [0, 1, 2, 3]):
            return False, frame, corners, ids

        src_pts = np.float32([id_map[0], id_map[1], id_map[2], id_map[3]])

        # We assume 10 pixels per mm for high resolution
        scale_factor = 10
        self.width_px = int(REAL_WIDTH_MM * scale_factor)
        self.height_px = int(REAL_HEIGHT_MM * scale_factor)

        dst_pts = np.float32([
            [0, 0],
            [self.width_px, 0],
            [self.width_px, self.height_px],
            [0, self.height_px]
        ])

        self.matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.px_per_mm = scale_factor
        return True, frame, corners, ids

    def process_view(self, frame):
        if self.matrix is None:
            return frame, []

        warped = cv2.warpPerspective(frame, self.matrix, (self.width_px, self.height_px))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Watershed Segmentation
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
            if cv2.contourArea(cnt) < (2 * self.px_per_mm) ** 2: continue

            if len(cnt) >= 5:
                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                dim1, dim2 = ma / self.px_per_mm, MA / self.px_per_mm
                length_mm, diameter_mm = max(dim1, dim2), min(dim1, dim2)

                is_good = abs(diameter_mm - TARGET_DIAMETER) <= TOLERANCE and \
                          abs(length_mm - TARGET_LENGTH) <= TOLERANCE

                results.append({'pos': (int(x), int(y)), 'd': diameter_mm, 'l': length_mm, 'ok': is_good})
                color = (0, 255, 0) if is_good else (0, 0, 255)
                cv2.ellipse(warped, ((x, y), (MA, ma), angle), color, 2)
                cv2.putText(warped, f"{diameter_mm:.1f}x{length_mm:.1f}", (int(x) - 20, int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return warped, results


# ----------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    system = PrecisionMeasure()

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Step 1: Detect Calibration
        success, _, corners, ids = system.calibrate_perspective(frame)

        if success:
            view, results = system.process_view(frame)
            cv2.putText(view, f"Pellets: {len(results)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Top-Down View", view)
        else:
            cv2.putText(frame, "Align 4 ArUco Markers (0,1,2,3)", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw detected markers on the raw feed for debugging
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.imshow("Raw Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()