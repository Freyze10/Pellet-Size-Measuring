import cv2
import numpy as np
import cv2.aruco as aruco

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
REAL_WIDTH_MM = 80.0
REAL_HEIGHT_MM = 80.0

# TARGETS
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
PELLET_THICKNESS_MM = 3.3  # We still need to know how thick the pellet is

# CAMERA RESOLUTION
DESIRED_WIDTH = 1920
DESIRED_HEIGHT = 1080


# ----------------------------------------------------------------------
# Measurement Engine
# ----------------------------------------------------------------------
class PrecisionMeasure:
    def __init__(self):
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.matrix = None

        # Scale Factor
        self.scale_factor = 15
        self.width_px = int(REAL_WIDTH_MM * self.scale_factor)
        self.height_px = int(REAL_HEIGHT_MM * self.scale_factor)
        self.px_per_mm = self.scale_factor

        # Variable to store calculated camera height
        self.calculated_camera_height = 0

    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) < 4: return False, corners, ids

        # Map IDs to centers
        id_map = {}
        for i, id_val in enumerate(ids):
            c = np.mean(corners[i][0], axis=0)
            id_map[id_val[0]] = c

        if not all(k in id_map for k in [0, 1, 2, 3]): return False, corners, ids

        # Get the pixel coordinates of the 4 marker centers
        img_points = np.float32([id_map[0], id_map[1], id_map[2], id_map[3]])

        # Define the 3D coordinates of those centers (Z=0 for the paper)
        # TL(0,0), TR(80,0), BR(80,80), BL(0,80)
        obj_points = np.float32([
            [0, 0, 0],
            [REAL_WIDTH_MM, 0, 0],
            [REAL_WIDTH_MM, REAL_HEIGHT_MM, 0],
            [0, REAL_HEIGHT_MM, 0]
        ])

        # --- AUTO-DETECT CAMERA HEIGHT (solvePnP) ---
        # 1. Approximate Camera Matrix (if uncalibrated, this is a standard guess)
        # Focal length is usually close to the image width for webcams
        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))  # Assume no lens distortion for simple approximation

        # 2. Calculate Position
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)

        if success:
            # The Z component of the translation vector is the distance
            self.calculated_camera_height = abs(tvec[2][0])

        # --- END AUTO-DETECT ---

        # Create Warp Matrix
        dst_pts = np.float32([[0, 0], [self.width_px, 0], [self.width_px, self.height_px], [0, self.height_px]])
        self.matrix = cv2.getPerspectiveTransform(img_points, dst_pts)

        return True, corners, ids

    def measure_pellets(self, frame):
        if self.matrix is None: return None, []

        warped = cv2.warpPerspective(frame, self.matrix, (self.width_px, self.height_px))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate dynamic compensation factor based on the auto-detected height
        if self.calculated_camera_height > PELLET_THICKNESS_MM:
            comp_factor = 1.0 - (PELLET_THICKNESS_MM / self.calculated_camera_height)
        else:
            comp_factor = 1.0

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (1.0 * self.px_per_mm) ** 2: continue

            # Smooth contour
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            rect = cv2.minAreaRect(approx)
            (center_x, center_y), (w, h), angle = rect

            raw_dim1 = w / self.px_per_mm
            raw_dim2 = h / self.px_per_mm

            # Apply Auto-Calculated Compensation
            comp_dim1 = raw_dim1 * comp_factor
            comp_dim2 = raw_dim2 * comp_factor

            length_mm = max(comp_dim1, comp_dim2)
            diameter_mm = min(comp_dim1, comp_dim2)

            is_good = (abs(diameter_mm - TARGET_DIAMETER) <= TOLERANCE and
                       abs(length_mm - TARGET_LENGTH) <= TOLERANCE)

            box = np.intp(cv2.boxPoints(rect))

            results.append({
                'box': box,
                'center': (int(center_x), int(center_y)),
                'd': diameter_mm,
                'l': length_mm,
                'ok': is_good
            })

        return warped, results


# ----------------------------------------------------------------------
# UI / Drawing
# ----------------------------------------------------------------------
def draw_ui(image, results, cam_height, is_frozen=False):
    output = image.copy()
    good_count = 0
    bad_count = 0

    if results:
        for r in results:
            color = (0, 255, 0) if r['ok'] else (0, 0, 255)
            if r['ok']:
                good_count += 1
            else:
                bad_count += 1

            cv2.drawContours(output, [r['box']], 0, color, 2)
            cx, cy = r['center']
            label = f"{r['d']:.2f} x {r['l']:.2f}"
            cv2.putText(output, label, (cx - 40, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(output, label, (cx - 40, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Status Bar
    h, w = output.shape[:2]
    cv2.rectangle(output, (0, 0), (w, 80), (30, 30, 30), -1)

    if is_frozen:
        status = f"CAPTURED | Total: {len(results)} | Good: {good_count} | Bad: {bad_count}"
        instr = "Press 'ESC' for Live View"
        col = (0, 255, 0)
    else:
        status = "LIVE VIEW"
        instr = "Align Markers & Press 'SPACE'"
        col = (200, 200, 200)

    cv2.putText(output, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    cv2.putText(output, instr, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    # Show Auto-Detected Height
    height_msg = f"Auto-Height: {cam_height:.0f}mm"
    cv2.putText(output, height_msg, (w - 300, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    return output


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    engine = PrecisionMeasure()
    in_capture_mode = False
    captured_view = None

    print("=" * 60)
    print("AUTO-DETECT HEIGHT PELLET INSPECTOR")
    print("=" * 60)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:
            in_capture_mode = False
        elif key == ord(' '):
            if not in_capture_mode:
                ret, frame = cap.read()
                if ret:
                    is_valid, _, _ = engine.detect_markers(frame)
                    if is_valid:
                        warped, results = engine.measure_pellets(frame)
                        captured_view = draw_ui(warped, results, engine.calculated_camera_height, is_frozen=True)
                        in_capture_mode = True

        if in_capture_mode and captured_view is not None:
            cv2.imshow("Inspector", captured_view)
        else:
            ret, frame = cap.read()
            if not ret: break
            is_valid, corners, ids = engine.detect_markers(frame)
            if is_valid: aruco.drawDetectedMarkers(frame, corners, ids)

            cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, 80), (30, 30, 30), -1)
            cv2.putText(frame, "LIVE VIEW", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            # Show live height estimation
            if is_valid:
                h_txt = f"Cam Height: {engine.calculated_camera_height:.0f}mm"
                cv2.putText(frame, h_txt, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "MARKERS NOT FOUND", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Inspector", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()