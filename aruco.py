import cv2
import numpy as np
import cv2.aruco as aruco

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
# 1. PHYSICAL SETUP
# The distance between the specific corners of the markers (Outer dimensions usually)
# Measure the distance from the Top-Left corner of Marker 0 to Top-Right of Marker 1
WORK_WIDTH_MM = 100.0
WORK_HEIGHT_MM = 100.0

# 2. PARALLAX COMPENSATION (Crucial for Accuracy)
# Because the pellet surface is closer to the camera than the table, it looks bigger.
# Value < 1.0 shrinks the result.
# Formula approx: 1.0 - (Pellet_Height / Camera_Height)
# Example: If Camera is 200mm up and pellet is 3mm tall: 1 - (3/200) = 0.985
HEIGHT_COMPENSATION = 0.985

# 3. TARGETS
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5


# ----------------------------------------------------------------------
# MEASUREMENT ENGINE
# ----------------------------------------------------------------------
class ArucoInspector:
    def __init__(self):
        # Dictionary: Using 4x4 markers is faster/robust for this
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # High Resolution Scale: Keep as much detail as possible
        # We calculate pixels based on 1080p height approx
        self.target_px_height = 1000
        self.px_per_mm = self.target_px_height / WORK_HEIGHT_MM
        self.target_px_width = int(WORK_WIDTH_MM * self.px_per_mm)

    def get_perspective_transform(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is None or len(ids) < 4:
            return None, "Need 4 Markers (0,1,2,3)"

        # Map IDs to coordinates
        id_map = {}
        for i, id_val in enumerate(ids):
            # Use the center of the marker for positioning
            c = np.mean(corners[i][0], axis=0)
            id_map[id_val[0]] = c

        # Check if we have IDs 0, 1, 2, 3
        # Layout Assumption: 0=TL, 1=TR, 2=BR, 3=BL
        if not all(x in id_map for x in [0, 1, 2, 3]):
            return None, "Missing ID 0,1,2, or 3"

        # Source Points (From Camera)
        src_pts = np.float32([id_map[0], id_map[1], id_map[2], id_map[3]])

        # Destination Points (Flat Top-Down View)
        dst_pts = np.float32([
            [0, 0],
            [self.target_px_width, 0],
            [self.target_px_width, self.target_px_height],
            [0, self.target_px_height]
        ])

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return matrix, "OK"

    def analyze_pellets(self, frame, matrix):
        # 1. Warp the image to be perfectly flat and to scale
        warped = cv2.warpPerspective(frame, matrix, (self.target_px_width, self.target_px_height))

        # 2. Image Enhancement
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # Bilateral filter keeps edges sharp while removing noise
        blur = cv2.bilateralFilter(gray, 9, 75, 75)

        # 3. Adaptive Thresholding (Better than Otsu for uneven light)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 3)

        # 4. Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter by area (approx 2mm x 2mm min size)
            min_area = (2.0 * self.px_per_mm) ** 2
            if area < min_area: continue

            # Rotated Rectangle for tightest fit
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect

            # Convert px to mm
            raw_w_mm = w / self.px_per_mm
            raw_h_mm = h / self.px_per_mm

            # Apply Height Compensation
            real_w_mm = raw_w_mm * HEIGHT_COMPENSATION
            real_h_mm = raw_h_mm * HEIGHT_COMPENSATION

            dim1 = min(real_w_mm, real_h_mm)  # Diameter
            dim2 = max(real_w_mm, real_h_mm)  # Length

            # Aspect Ratio check to ignore noise lines
            if dim1 > 0 and (dim2 / dim1) < 4.0:
                is_good = (
                        abs(dim1 - TARGET_DIAMETER) <= TOLERANCE and
                        abs(dim2 - TARGET_LENGTH) <= TOLERANCE
                )

                box = np.intp(cv2.boxPoints(rect))
                results.append({
                    'box': box, 'cx': int(cx), 'cy': int(cy),
                    'd': dim1, 'l': dim2, 'ok': is_good
                })

        return warped, results


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Request High Res
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Lock focus

    inspector = ArucoInspector()

    # State
    mode = 'LIVE'  # LIVE or CAPTURED
    matrix = None
    last_results = []
    display_frame = None

    print("--- ARUCO INSPECTOR ---")
    print("1. Place markers (IDs 0,1,2,3) in a rectangle.")
    print("2. Put pellets INSIDE the rectangle.")
    print("3. Press SPACE to measure.")

    while True:
        if mode == 'LIVE':
            ret, frame = cap.read()
            if not ret: break

            # Just visualize markers in live mode
            matrix, status = inspector.get_perspective_transform(frame)

            # Draw HUD
            cv2.putText(frame, f"Status: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "Press SPACE to Measure", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if matrix is not None:
                # Draw a green border showing active area
                cv2.putText(frame, "READY", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            cv2.imshow("Inspector", frame)

        elif mode == 'CAPTURED':
            # Static display of results
            final_img = display_frame.copy()

            # Stats
            good = sum(1 for r in last_results if r['ok'])
            bad = len(last_results) - good

            # Draw
            for r in last_results:
                color = (0, 255, 0) if r['ok'] else (0, 0, 255)
                cv2.drawContours(final_img, [r['box']], 0, color, 2)
                lbl = f"{r['d']:.2f}x{r['l']:.2f}"
                cv2.putText(final_img, lbl, (r['cx'] - 40, r['cy'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 4)
                cv2.putText(final_img, lbl, (r['cx'] - 40, r['cy'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # Info Bar
            cv2.rectangle(final_img, (0, 0), (1280, 80), (30, 30, 30), -1)
            cv2.putText(final_img, f"TOTAL: {len(last_results)} | OK: {good} | BAD: {bad}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(final_img, "Press SPACE for Live View", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255),
                        1)

            cv2.imshow("Inspector", final_img)

        # Inputs
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # SPACE
            if mode == 'LIVE':
                # Capture and Process
                if matrix is not None:
                    warped, results = inspector.analyze_pellets(frame, matrix)
                    display_frame = warped
                    last_results = results
                    mode = 'CAPTURED'
                    print(f"Measured {len(results)} pellets.")
                else:
                    print("Cannot capture - Markers not ready.")
            else:
                mode = 'LIVE'

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()