import cv2
import numpy as np
import cv2.aruco as aruco

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# WORKSPACE: The distance between ArUco markers (Inner rectangle)
REAL_WIDTH_MM = 80.0
REAL_HEIGHT_MM = 80.0

# TARGETS - Pellet is 3mm THICK (height), measuring top-down view
# We're measuring the LENGTH and WIDTH of the pellet from above
TARGET_WIDTH = 3.0  # Width of pellet (mm) - viewed from top
TARGET_LENGTH = 3.0  # Length of pellet (mm) - viewed from top
TOLERANCE = 0.3  # Tighter tolerance

# CAMERA
DESIRED_WIDTH = 1920  # Higher resolution
DESIRED_HEIGHT = 1080

# CALIBRATION - Measure your actual ArUco marker size
ARUCO_MARKER_SIZE_MM = 20.0  # Adjust this to your actual marker size


# ----------------------------------------------------------------------
# Measurement Engine (ArUco + Enhanced Detection)
# ----------------------------------------------------------------------
class PrecisionMeasure:
    def __init__(self):
        # ArUco Setup with refined parameters
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()

        # Fine-tune detection parameters
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.cornerRefinementMaxIterations = 30

        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.matrix = None
        self.actual_px_per_mm = None  # Dynamically calculated

        # Scale: Fixed at 12 px/mm for optimal balance
        self.scale_factor = 12
        self.width_px = int(REAL_WIDTH_MM * self.scale_factor)
        self.height_px = int(REAL_HEIGHT_MM * self.scale_factor)
        self.px_per_mm = self.scale_factor  # Fixed at 12

    def detect_markers(self, frame):
        """Detects markers to establish the workspace with sub-pixel accuracy."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for better marker detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) < 4:
            return False, corners, ids

        id_map = {}
        for i, id_val in enumerate(ids):
            # Use precise corner averaging
            c = np.mean(corners[i][0], axis=0)
            id_map[id_val[0]] = c

        if not all(k in id_map for k in [0, 1, 2, 3]):
            return False, corners, ids

        # Calculate actual pixel-per-mm ratio from marker positions
        # This accounts for camera distortion and angle
        marker_dist_px = np.linalg.norm(id_map[0] - id_map[1])
        self.actual_px_per_mm = marker_dist_px / REAL_WIDTH_MM

        # Order: TL, TR, BR, BL
        src_pts = np.float32([id_map[0], id_map[1], id_map[2], id_map[3]])
        dst_pts = np.float32([[0, 0], [self.width_px, 0],
                              [self.width_px, self.height_px], [0, self.height_px]])

        self.matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return True, corners, ids

    def measure_pellets(self, frame):
        """Warps the view and detects pellets with multi-method measurement."""
        if self.matrix is None:
            return None, []

        # 1. Warp Perspective with high-quality interpolation
        warped = cv2.warpPerspective(frame, self.matrix, (self.width_px, self.height_px),
                                     flags=cv2.INTER_CUBIC)

        # 2. Multi-stage pre-processing
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Denoise while preserving edges
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Multiple blur for stability
        blur1 = cv2.GaussianBlur(enhanced, (3, 3), 0)
        blur2 = cv2.bilateralFilter(blur1, 5, 50, 50)

        # 3. Multi-threshold approach (combine methods)
        # Method 1: Otsu
        _, thresh_otsu = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Method 2: Adaptive
        thresh_adaptive = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 15, 2)

        # Combine both thresholds
        thresh = cv2.bitwise_and(thresh_otsu, thresh_adaptive)

        # Advanced morphological operations
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_ellipse, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_ellipse, iterations=1)

        # Remove small noise with connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

        # 4. Find Contours with hierarchy
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Stricter area filtering
            min_area = (0.8 * self.px_per_mm) ** 2
            max_area = (8.0 * self.px_per_mm) ** 2
            if area < min_area or area > max_area:
                continue

            # Calculate perimeter for shape analysis
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            # Circularity check (helps filter noise)
            circularity = 4 * np.pi * area / (perimeter ** 2)

            # Solidity check (ratio of contour area to convex hull area)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # --- MULTI-METHOD MEASUREMENT ---

            # Method 1: minAreaRect (rotated bounding box)
            rect = cv2.minAreaRect(cnt)
            (center_x, center_y), (w1, h1), angle = rect
            dim1_mm = w1 / self.px_per_mm
            dim2_mm = h1 / self.px_per_mm

            # Method 2: Ellipse fitting (more accurate for round objects)
            if len(cnt) >= 5:  # Need at least 5 points for ellipse
                ellipse = cv2.fitEllipse(cnt)
                (ex, ey), (ew, eh), eangle = ellipse
                edim1_mm = ew / self.px_per_mm
                edim2_mm = eh / self.px_per_mm

                # Average the two methods
                dim1_mm = (dim1_mm + edim1_mm) / 2
                dim2_mm = (dim2_mm + edim2_mm) / 2

            # Method 3: Equivalent diameter from area (cross-validation)
            equiv_diameter_mm = 2 * np.sqrt(area / np.pi) / self.px_per_mm

            # Determine width and length (both are face dimensions, not thickness)
            # For rectangular pellets: smaller dimension = width, larger = length
            width_mm = min(dim1_mm, dim2_mm)
            length_mm = max(dim1_mm, dim2_mm)

            # Note: 3mm thickness is NOT measured (it's the height perpendicular to table)
            # We only measure the top rectangular face: width x length

            # Quality checks
            is_circular = circularity > 0.65
            is_solid = solidity > 0.85

            # Tolerance Check with quality weighting
            width_ok = abs(width_mm - TARGET_WIDTH) <= TOLERANCE
            length_ok = abs(length_mm - TARGET_LENGTH) <= TOLERANCE
            is_good = width_ok and length_ok and is_circular and is_solid

            # Get box points for drawing
            box = np.intp(cv2.boxPoints(rect))

            results.append({
                'box': box,
                'center': (int(center_x), int(center_y)),
                'w': width_mm,  # Width (smaller dimension)
                'l': length_mm,  # Length (larger dimension)
                'ok': is_good,
                'circularity': circularity,
                'solidity': solidity,
                'area_mm2': area / (self.px_per_mm ** 2)
            })

        return warped, results


# ----------------------------------------------------------------------
# UI / Drawing
# ----------------------------------------------------------------------
def draw_ui(image, results, is_frozen=False, px_per_mm=None):
    output = image.copy()

    # Resize output for smaller display window (50% scale)
    display_scale = 0.6
    display_width = int(output.shape[1] * display_scale)
    display_height = int(output.shape[0] * display_scale)

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

            # Draw center crosshair
            cx, cy = r['center']
            cv2.line(output, (cx - 5, cy), (cx + 5, cy), color, 1)
            cv2.line(output, (cx, cy - 5), (cx, cy + 5), color, 1)

            # Draw Text with quality indicators
            label = f"W:{r['w']:.2f} L:{r['l']:.2f}"
            if not r['ok']:
                label += f" Q:{r['circularity']:.2f}"

            # White outline for readability
            cv2.putText(output, label, (cx - 35, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 3)
            cv2.putText(output, label, (cx - 35, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Draw Status Bar
    cv2.rectangle(output, (0, 0), (output.shape[1], 90), (30, 30, 30), -1)

    if is_frozen:
        status = f"CAPTURED | Total: {len(results)} | Good: {good_count} | Bad: {bad_count}"
        detail = f"Resolution: 12 px/mm | Target: {TARGET_WIDTH}x{TARGET_LENGTH}mm ±{TOLERANCE}mm"
        instr = "Press 'ESC' for Live View | 'S' to Save | 'R' for Stats"
        col = (0, 255, 0)
    else:
        status = "LIVE VIEW - Align Markers (IDs 0,1,2,3)"
        detail = "Measuring pellet top face (3mm thickness perpendicular to table)"
        instr = "Press 'SPACE' to Capture | 'Q' to Quit"
        col = (200, 200, 200)

    cv2.putText(output, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.putText(output, detail, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    cv2.putText(output, instr, (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # Resize for display
    output_display = cv2.resize(output, (display_width, display_height), interpolation=cv2.INTER_AREA)
    return output_display


def print_statistics(results):
    """Print detailed measurement statistics."""
    if not results:
        print("No pellets detected.")
        return

    print("\n" + "=" * 60)
    print("DETAILED STATISTICS")
    print("=" * 60)

    diameters = [r['w'] for r in results]  # Width measurements
    lengths = [r['l'] for r in results]
    circularities = [r['circularity'] for r in results]
    solidities = [r['solidity'] for r in results]

    print(f"Total Pellets: {len(results)}")
    print(f"  Good: {sum(1 for r in results if r['ok'])}")
    print(f"  Bad:  {sum(1 for r in results if not r['ok'])}")
    print()
    print(f"Width (mm) - top face measurement:")
    print(f"  Range:  {min(diameters):.3f} - {max(diameters):.3f}")
    print(f"  Mean:   {np.mean(diameters):.3f} ± {np.std(diameters):.3f}")
    print(f"  Median: {np.median(diameters):.3f}")
    print()
    print(f"Length (mm) - top face measurement:")
    print(f"  Range:  {min(lengths):.3f} - {max(lengths):.3f}")
    print(f"  Mean:   {np.mean(lengths):.3f} ± {np.std(lengths):.3f}")
    print(f"  Median: {np.median(lengths):.3f}")
    print()
    print(f"Note: 3mm thickness (height) is perpendicular to table, not measured")
    print()
    print(f"Quality Metrics:")
    print(f"  Circularity: {np.mean(circularities):.3f} (avg)")
    print(f"  Solidity:    {np.mean(solidities):.3f} (avg)")
    print("=" * 60)


# ----------------------------------------------------------------------
# Main Application
# ----------------------------------------------------------------------
def main():
    # Try different camera backends for best quality
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    # Set highest resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)

    # Manual focus for consistent measurements
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 0)

    # Disable auto exposure for consistent lighting
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Adjust as needed

    # Set high FPS for stability
    cap.set(cv2.CAP_PROP_FPS, 30)

    engine = PrecisionMeasure()

    # State
    in_capture_mode = False
    captured_view = None
    captured_results = None

    print("=" * 60)
    print("ENHANCED ARUCO PELLET INSPECTOR")
    print("=" * 60)
    print(f"Target: Width={TARGET_WIDTH}mm, Length={TARGET_LENGTH}mm (±{TOLERANCE}mm)")
    print(f"Note: Pellet is 3mm THICK (perpendicular to table)")
    print(f"      We measure the TOP FACE width x length")
    print(f"Workspace: {REAL_WIDTH_MM}mm x {REAL_HEIGHT_MM}mm")
    print(f"Resolution: {DESIRED_WIDTH}x{DESIRED_HEIGHT}")
    print("=" * 60)
    print("SETUP REQUIREMENTS:")
    print("  • ArUco markers flat on table (same level as pellets)")
    print("  • Camera positioned directly above (perpendicular)")
    print("  • Even, diffuse lighting (no shadows)")
    print("  • Manual focus enabled (disable autofocus)")
    print("  • Stable camera (tripod recommended)")
    print("=" * 60)
    print("Controls:")
    print("  SPACE: Capture and measure")
    print("  ESC:   Return to live view")
    print("  S:     Save captured image")
    print("  R:     Show detailed statistics")
    print("  Q:     Quit")
    print("=" * 60)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == 27:  # ESC
            in_capture_mode = False
            print("→ Returned to Live View")
        elif key == ord(' '):  # SPACE
            if not in_capture_mode:
                ret, frame = cap.read()
                if ret:
                    is_valid, _, _ = engine.detect_markers(frame)
                    if is_valid:
                        warped, results = engine.measure_pellets(frame)
                        captured_view = draw_ui(warped, results, is_frozen=True,
                                                px_per_mm=12)  # Fixed at 12 px/mm
                        captured_results = results
                        in_capture_mode = True

                        good = sum(1 for r in results if r['ok'])
                        bad = len(results) - good
                        print(f"\n✓ Captured {len(results)} pellets: {good} good, {bad} bad")
                        print(f"  Resolution: 12 px/mm (fixed)")
                    else:
                        print("⚠ Cannot capture: Ensure all 4 markers (IDs 0,1,2,3) are visible")
        elif key == ord('s') and in_capture_mode and captured_view is not None:
            filename = f"pellet_measure_{len(captured_results)}.png"
            cv2.imwrite(filename, captured_view)
            print(f"✓ Saved: {filename}")
        elif key == ord('r') and in_capture_mode and captured_results is not None:
            print_statistics(captured_results)

        # Display Logic
        if in_capture_mode and captured_view is not None:
            cv2.imshow("Pellet Inspector", captured_view)
        else:
            ret, frame = cap.read()
            if not ret:
                print("⚠ Cannot read from camera")
                break

            is_valid, corners, ids = engine.detect_markers(frame)

            if is_valid:
                aruco.drawDetectedMarkers(frame, corners, ids)

            # Draw overlay with smaller display
            overlay_height = 70
            cv2.rectangle(frame, (0, 0), (DESIRED_WIDTH, overlay_height), (30, 30, 30), -1)
            cv2.putText(frame, "LIVE VIEW - Press SPACE to Capture", (20, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            if not is_valid:
                cv2.putText(frame, "⚠ MARKERS NOT DETECTED", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "Align all 4 ArUco markers (IDs 0,1,2,3)", (20, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            else:
                cv2.putText(frame, "✓ Ready - Good alignment", (20, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Resize live view for smaller display
            display_width = 640
            display_height = 360
            frame_display = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
            cv2.imshow("Pellet Inspector", frame_display)

    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "=" * 60)
    print("Application closed")
    print("=" * 60)


if __name__ == "__main__":
    main()