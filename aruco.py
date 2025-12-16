import cv2
import numpy as np
import cv2.aruco as aruco

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# WORKSPACE: The distance between ArUco markers (Inner rectangle)
REAL_WIDTH_MM = 80.0
REAL_HEIGHT_MM = 80.0

# TARGETS - Pellet dimensions
TARGET_DIAMETER = 3.0  # 3mm diameter (width when viewed from top)
TARGET_LENGTH = 3.0  # 3mm length (assuming cylindrical pellets)
TOLERANCE = 0.3  # Tolerance for measurements

# CAMERA
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720


# ----------------------------------------------------------------------
# Measurement Engine (ArUco + Dimension Detection)
# ----------------------------------------------------------------------
class PrecisionMeasure:
    def __init__(self):
        # ArUco Setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.matrix = None

        # Scale: Higher resolution for better accuracy (15 px = 1 mm)
        self.scale_factor = 15
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
            # Use center of marker
            c = np.mean(corners[i][0], axis=0)
            id_map[id_val[0]] = c

        if not all(k in id_map for k in [0, 1, 2, 3]):
            return False, corners, ids

        # Order: TL, TR, BR, BL (IDs 0, 1, 2, 3)
        src_pts = np.float32([id_map[0], id_map[1], id_map[2], id_map[3]])
        dst_pts = np.float32([[0, 0], [self.width_px, 0],
                              [self.width_px, self.height_px], [0, self.height_px]])

        self.matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return True, corners, ids

    def measure_pellets(self, frame):
        """Warps the view and measures pellet width and length."""
        if self.matrix is None:
            return None, []

        # 1. Warp Perspective (Flatten image)
        warped = cv2.warpPerspective(frame, self.matrix, (self.width_px, self.height_px))

        # 2. Enhanced Pre-processing
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Bilateral filter to reduce noise while preserving edges
        blur = cv2.bilateralFilter(gray, 9, 75, 75)

        # 3. Adaptive thresholding for better edge detection
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # 4. Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Filter by area - reasonable pellet size
            if area < (0.5 * self.px_per_mm) ** 2:  # Too small (< 0.5mm²)
                continue
            if area > (20.0 * self.px_per_mm) ** 2:  # Too large (> 20mm²)
                continue

            # Use minAreaRect to get the minimum bounding rectangle
            # This gives us the width and length of the pellet
            rect = cv2.minAreaRect(cnt)
            (center_x, center_y), (w_px, h_px), angle = rect

            # Convert dimensions from pixels to mm
            width_mm = w_px / self.px_per_mm
            height_mm = h_px / self.px_per_mm

            # Determine which is length and which is width
            # For cylindrical pellets viewed from top: smallest dimension is diameter
            if width_mm < height_mm:
                diameter_mm = width_mm
                length_mm = height_mm
            else:
                diameter_mm = height_mm
                length_mm = width_mm

            # Calculate circularity for quality check
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

            # Tolerance Check - both dimensions should be within spec
            diameter_ok = abs(diameter_mm - TARGET_DIAMETER) <= TOLERANCE
            length_ok = abs(length_mm - TARGET_LENGTH) <= TOLERANCE
            is_good = diameter_ok and length_ok

            # Get box points for drawing the rotated rectangle
            box = np.intp(cv2.boxPoints(rect))
            center = (int(center_x), int(center_y))

            results.append({
                'box': box,
                'center': center,
                'width': diameter_mm,  # Width (diameter)
                'length': length_mm,  # Length
                'angle': angle,
                'circularity': circularity,
                'area_mm2': area / (self.px_per_mm ** 2),
                'ok': is_good,
                'width_ok': diameter_ok,
                'length_ok': length_ok
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

            # Draw rotated rectangle (bounding box)
            cv2.drawContours(output, [r['box']], 0, color, 2)

            # Draw center point
            cv2.circle(output, r['center'], 3, color, -1)

            # Draw crosshair to show orientation
            cx, cy = r['center']
            angle_rad = np.deg2rad(r['angle'])
            line_len = 15
            dx = int(line_len * np.cos(angle_rad))
            dy = int(line_len * np.sin(angle_rad))
            cv2.line(output, (cx - dx, cy - dy), (cx + dx, cy + dy), color, 1)

            # Draw Text with measurements (Width x Length)
            label = f"W:{r['width']:.2f} L:{r['length']:.2f}"

            # Add quality indicators if bad
            if not r['ok']:
                issues = []
                if not r['width_ok']:
                    issues.append('W')
                if not r['length_ok']:
                    issues.append('L')
                label += f" [{'/'.join(issues)}]"

            # White outline for readability
            text_y = cy - 20
            cv2.putText(output, label, (cx - 50, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 3)
            cv2.putText(output, label, (cx - 50, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw Status Bar
    cv2.rectangle(output, (0, 0), (output.shape[1], 80), (30, 30, 30), -1)

    if is_frozen:
        status = f"CAPTURED | Total: {len(results)} | Good: {good_count} | Bad: {bad_count}"
        detail = f"Target: W={TARGET_DIAMETER}mm, L={TARGET_LENGTH}mm (±{TOLERANCE}mm)"
        instr = "Press 'ESC' for Live View | 'S' to Save"
        col = (0, 255, 0)
    else:
        status = "LIVE VIEW - Align Markers (IDs 0,1,2,3)"
        detail = f"Measuring pellet dimensions: W={TARGET_DIAMETER}mm, L={TARGET_LENGTH}mm"
        instr = "Press 'SPACE' to Capture"
        col = (200, 200, 200)

    cv2.putText(output, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.putText(output, detail, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(output, instr, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return output


# ----------------------------------------------------------------------
# Main Application
# ----------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Try without DSHOW

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
    cap.set(cv2.CAP_PROP_FOCUS, 0)  # Manual focus for flat surface

    engine = PrecisionMeasure()

    # State
    in_capture_mode = False
    captured_view = None
    captured_results = None

    print("=" * 60)
    print("ARUCO PELLET INSPECTOR - Width & Length Measurement")
    print("=" * 60)
    print(f"Target Width (Diameter): {TARGET_DIAMETER}mm")
    print(f"Target Length: {TARGET_LENGTH}mm")
    print(f"Tolerance: ±{TOLERANCE}mm")
    print(f"Workspace: {REAL_WIDTH_MM}mm x {REAL_HEIGHT_MM}mm")
    print("=" * 60)
    print("Place ArUco markers at corners (IDs 0,1,2,3):")
    print("  0: Top-Left    1: Top-Right")
    print("  3: Bottom-Left 2: Bottom-Right")
    print("=" * 60)
    print("Controls:")
    print("  SPACE: Capture and measure")
    print("  ESC:   Return to live view")
    print("  S:     Save captured image")
    print("  Q:     Quit")
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
                        captured_results = results
                        in_capture_mode = True

                        good = sum(1 for r in results if r['ok'])
                        bad = len(results) - good
                        print(f"\n✓ Captured {len(results)} pellets: {good} good, {bad} bad")

                        if results:
                            widths = [r['width'] for r in results]
                            lengths = [r['length'] for r in results]
                            print(f"  Width range:  {min(widths):.2f} - {max(widths):.2f}mm")
                            print(f"  Width avg:    {np.mean(widths):.2f}mm (±{np.std(widths):.2f}mm)")
                            print(f"  Length range: {min(lengths):.2f} - {max(lengths):.2f}mm")
                            print(f"  Length avg:   {np.mean(lengths):.2f}mm (±{np.std(lengths):.2f}mm)")
                    else:
                        print("⚠ Cannot capture: Markers not aligned (need IDs 0,1,2,3)")
        elif key == ord('s') and in_capture_mode and captured_view is not None:
            # Save captured image
            filename = f"pellet_measurement_{len(captured_results)}.png"
            cv2.imwrite(filename, captured_view)
            print(f"✓ Saved: {filename}")

        # Display Logic
        if in_capture_mode and captured_view is not None:
            cv2.imshow("Pellet Inspector", captured_view)
        else:
            ret, frame = cap.read()
            if not ret:
                print("⚠ Cannot read from camera")
                break

            # Detect markers for visual feedback
            is_valid, corners, ids = engine.detect_markers(frame)

            # Show Raw Feed with Markers
            if is_valid:
                aruco.drawDetectedMarkers(frame, corners, ids)

            # Draw overlay
            cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, 80), (30, 30, 30), -1)
            cv2.putText(frame, "LIVE VIEW - Press SPACE to Capture", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(frame, "Align all 4 ArUco markers (IDs 0,1,2,3)", (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            if not is_valid:
                cv2.putText(frame, "⚠ MARKERS NOT DETECTED", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "✓ Ready to capture", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Pellet Inspector", frame)

    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "=" * 60)
    print("Application closed")
    print("=" * 60)


if __name__ == "__main__":
    main()