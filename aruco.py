yimport cv2
import numpy as np
import cv2.aruco as aruco

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# WORKSPACE: The distance between ArUco markers (Inner rectangle)
REAL_WIDTH_MM = 80.0
REAL_HEIGHT_MM = 80.0

# TARGETS
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5

# CAMERA
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720


# ----------------------------------------------------------------------
# Measurement Engine (ArUco + Bounding Box)
# ----------------------------------------------------------------------
class PrecisionMeasure:
    def __init__(self):
        # ArUco Setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.matrix = None

        # Scale: High resolution (10 px = 1 mm)
        self.scale_factor = 10
        self.width_px = int(REAL_WIDTH_MM * self.scale_factor)
        self.height_px = int(REAL_HEIGHT_MM * self.scale_factor)
        self.px_per_mm = self.scale_factor

    def detect_markers(self, frame):
        """Detects markers to establish the workspace."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) < 4:
            return False, corners, ids

        id_map = {}
        for i, id_val in enumerate(ids):
            c = np.mean(corners[i][0], axis=0)
            id_map[id_val[0]] = c

        if not all(k in id_map for k in [0, 1, 2, 3]):
            return False, corners, ids

        # Order: TL, TR, BR, BL
        src_pts = np.float32([id_map[0], id_map[1], id_map[2], id_map[3]])
        dst_pts = np.float32([[0, 0], [self.width_px, 0],
                              [self.width_px, self.height_px], [0, self.height_px]])

        self.matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return True, corners, ids

    def measure_pellets(self, frame):
        """Warps the view and detects pellets using Bounding Boxes."""
        if self.matrix is None:
            return None, []

        # 1. Warp Perspective (Flatten image)
        warped = cv2.warpPerspective(frame, self.matrix, (self.width_px, self.height_px))

        # 2. Pre-processing
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Threshold (Otsu is usually best for calibration sheets)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Clean up noise
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # 4. Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter tiny noise (approx 1mm^2)
            if area < (1.0 * self.px_per_mm) ** 2: continue

            # --- BOUNDING BOX LOGIC ---
            rect = cv2.minAreaRect(cnt)
            (center_x, center_y), (w, h), angle = rect

            # Convert px to mm
            dim1 = w / self.px_per_mm
            dim2 = h / self.px_per_mm

            # Assumption: Longest side is Length
            length_mm = max(dim1, dim2)
            diameter_mm = min(dim1, dim2)

            # Tolerance Check
            is_good = (abs(diameter_mm - TARGET_DIAMETER) <= TOLERANCE and
                       abs(length_mm - TARGET_LENGTH) <= TOLERANCE)

            # Get box points for drawing
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
def draw_ui(image, results, is_frozen=False):
    output = image.copy()

    good_count = 0
    bad_count = 0

    # Draw Pellets
    if results:
        for r in results:
            color = (0, 255, 0) if r['ok'] else (0, 0, 255)  # Green vs Red
            if r['ok']:
                good_count += 1
            else:
                bad_count += 1

            # Draw Rectangle
            cv2.drawContours(output, [r['box']], 0, color, 2)

            # Draw Text
            cx, cy = r['center']
            label = f"{r['d']:.2f}x{r['l']:.2f}"
            cv2.putText(output, label, (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 4)  # Outline
            cv2.putText(output, label, (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Draw Status Bar
    cv2.rectangle(output, (0, 0), (output.shape[1], 60), (30, 30, 30), -1)

    if is_frozen:
        status = f"CAPTURED | Total: {len(results)} | Good: {good_count} | Bad: {bad_count}"
        instr = "Press 'ESC' for Live View"
        col = (0, 255, 0)
    else:
        status = "LIVE VIEW - Align Markers"
        instr = "Press 'SPACE' to Capture"
        col = (200, 200, 200)

    cv2.putText(output, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 1)
    cv2.putText(output, instr, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return output


# ----------------------------------------------------------------------
# Main Application
# ----------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Important

    engine = PrecisionMeasure()

    # State
    in_capture_mode = False
    captured_view = None

    print("=" * 60)
    print("ARUCO PELLET INSPECTOR")
    print("Align 4 markers (IDs 0,1,2,3)")
    print("SPACE: Capture | ESC: Live View")
    print("=" * 60)

    while True:
        # User Interaction
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:  # ESC
            in_capture_mode = False
            print("Returned to Live View")
        elif key == ord(' '):  # SPACE
            if not in_capture_mode:
                ret, frame = cap.read()
                if ret:
                    is_valid, _, _ = engine.detect_markers(frame)
                    if is_valid:
                        warped, results = engine.measure_pellets(frame)
                        captured_view = draw_ui(warped, results, is_frozen=True)
                        in_capture_mode = True
                        print(f"Captured {len(results)} pellets.")
                    else:
                        print("Cannot capture: Markers not aligned.")

        # Display Logic
        if in_capture_mode and captured_view is not None:
            cv2.imshow("Inspector", captured_view)
        else:
            ret, frame = cap.read()
            if not ret: break

            # Detect markers just for visual feedback
            is_valid, corners, ids = engine.detect_markers(frame)

            # Show Raw Feed with Markers
            if is_valid:
                aruco.drawDetectedMarkers(frame, corners, ids)

            # Draw simple overlay
            cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, 60), (30, 30, 30), -1)
            cv2.putText(frame, "LIVE VIEW - Press SPACE to Capture", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            if not is_valid:
                cv2.putText(frame, "MARKERS NOT DETECTED", (20, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Inspector", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()