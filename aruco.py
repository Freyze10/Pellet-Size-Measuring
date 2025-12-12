import cv2
import numpy as np
import cv2.aruco as aruco

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Real-world dimensions of the rectangle formed by the 4 ArUco markers
# Width and Height in Millimeters (Measure exactly with a physical tape)
REAL_WIDTH_MM = 200.0
REAL_HEIGHT_MM = 150.0

# Target Pellet Dimensions (mm)
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5


# ----------------------------------------------------------------------
# Advanced Measurement Class
# ----------------------------------------------------------------------
class PrecisionMeasure:
    def __init__(self):
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.matrix = None  # Perspective matrix
        self.width_px = 0
        self.height_px = 0
        self.px_per_mm = 0

    def calibrate_perspective(self, frame):
        """
        Finds 4 ArUco markers (IDs 0,1,2,3) to define the workspace.
        Calculates the transformation matrix to 'flatten' the view.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None or len(ids) < 4:
            return False, frame

        # Map IDs to corners (Top-Left: 0, Top-Right: 1, Bottom-Right: 2, Bottom-Left: 3)
        # You must arrange markers 0,1,2,3 in clockwise order on your table
        id_map = {}
        for i, id_val in enumerate(ids):
            # center of the marker
            c = np.mean(corners[i][0], axis=0)
            id_map[id_val[0]] = c

        if not all(k in id_map for k in [0, 1, 2, 3]):
            return False, frame

        src_pts = np.float32([id_map[0], id_map[1], id_map[2], id_map[3]])

        # Determine output image size based on aspect ratio of real world size
        # We assume 10 pixels per mm for high resolution processing
        scale_factor = 10
        self.width_px = int(REAL_WIDTH_MM * scale_factor)
        self.height_px = int(REAL_HEIGHT_MM * scale_factor)

        dst_pts = np.float32([
            [0, 0],
            [self.width_px, 0],
            [self.width_px, self.height_px],
            [0, self.height_px]
        ])

        # Get the transformation matrix
        self.matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.px_per_mm = scale_factor
        return True, frame

    def process_view(self, frame):
        """
        Warps the live frame to top-down view and measures objects.
        """
        if self.matrix is None:
            return frame, []

        # 1. Warp Perspective (Flatten the image)
        warped = cv2.warpPerspective(frame, self.matrix, (self.width_px, self.height_px))

        # 2. Pre-processing (Optimize for measurement)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Using OTSU thresholding for automatic distinct separation
        # Note: If you use a backlight, this becomes trivial and highly accurate
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. Watershed Algorithm (Separate touching pellets)
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        # Finding sure foreground area (center of pellets)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Apply Watershed
        markers = cv2.watershed(warped, markers)

        results = []

        # 4. Measure blobs
        unique_markers = np.unique(markers)
        for mark in unique_markers:
            if mark <= 1: continue  # Skip background and unknown

            # Create mask for this specific object
            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[markers == mark] = 255

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue

            cnt = contours[0]
            area = cv2.contourArea(cnt)

            # Filter noise (approx 2mm^2 minimum)
            if area < (2 * self.px_per_mm) ** 2: continue

            # Precision Measurement: Fit Ellipse
            # Ellipse fits are more accurate than Rects for round/cylindrical objects
            if len(cnt) >= 5:
                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)

                # In an ellipse, the shorter axis is usually the diameter (if viewed from top)
                # The longer axis is the length
                dim1 = ma / self.px_per_mm
                dim2 = MA / self.px_per_mm

                length_mm = max(dim1, dim2)
                diameter_mm = min(dim1, dim2)

                # Check Tolerance
                d_ok = abs(diameter_mm - TARGET_DIAMETER) <= TOLERANCE
                l_ok = abs(length_mm - TARGET_LENGTH) <= TOLERANCE
                is_good = d_ok and l_ok

                results.append({
                    'pos': (int(x), int(y)),
                    'd': diameter_mm,
                    'l': length_mm,
                    'ok': is_good
                })

                # Draw on Warped Image
                color = (0, 255, 0) if is_good else (0, 0, 255)
                cv2.ellipse(warped, ((x, y), (MA, ma), angle), color, 2)
                cv2.putText(warped, f"{diameter_mm:.2f}x{length_mm:.2f}",
                            (int(x) - 20, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return warped, results


# ----------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Disable Autofocus (Very Important for Calibration Stability)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 150)  # Adjust this value manually

    system = PrecisionMeasure()

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Step 1: Detect Calibration Board
        calibrated, _ = system.calibrate_perspective(frame)

        # Step 2: Process View
        if calibrated:
            # We display the warped (top-down) view as the main UI
            # This view is mathematically corrected for perspective
            view, results = system.process_view(frame)

            # UI Overlay
            cv2.putText(view, f"Scale: {system.px_per_mm:.1f} px/mm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(view, f"Pellets: {len(results)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show the corrected view
            cv2.imshow("Precision Inspector (Top-Down)", view)

            # Show raw frame with markers for debugging
            aruco.drawDetectedMarkers(frame,
                                      aruco.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                                          system.aruco_dict)[0])
            cv2.imshow("Raw Feed", frame)

        else:
            cv2.putText(frame, "Looking for 4 ArUco Markers...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Precision Inspector (Top-Down)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()