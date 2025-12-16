import cv2
import numpy as np
import cv2.aruco as aruco

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

# 1. PHYSICAL DIMENSIONS (For Aspect Ratio Only)
# This keeps the image "square". It doesn't affect measurement accuracy,
# but ensures circles look like circles, not ovals.
# Distance between Marker 0 (Top-Left) and Marker 1 (Top-Right)
WORKSPACE_WIDTH_MM = 80.0
# Distance between Marker 0 (Top-Left) and Marker 3 (Bottom-Left)
WORKSPACE_HEIGHT_MM = 80.0

# 2. REFERENCE OBJECT (The "Key" to Accuracy)
# Exact diameter of the coin/washer you place INSIDE the workspace.
# US Quarter = 24.26mm, 1 Euro = 23.25mm
REFERENCE_DIAMETER_MM = 24.26

# 3. PELLET TARGETS
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5


# ----------------------------------------------------------------------
# SYSTEM STATE
# ----------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.mode = 'LIVE'  # LIVE or ANALYSIS
        self.frame = None
        self.warped_view = None

        # Calibration Data
        self.pixels_per_mm = None
        self.reference_contour = None

        # Results
        self.contours = []
        self.results = []

        # ArUco
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)


state = AppState()


# ----------------------------------------------------------------------
# IMAGE PROCESSING
# ----------------------------------------------------------------------
def get_warped_workspace(frame):
    """
    Detects ArUco markers and flattens the image (Top-Down View).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = state.detector.detectMarkers(gray)

    if ids is None or len(ids) < 4:
        return None, "Markers not found"

    # Map IDs to corners
    id_map = {}
    for i, id_val in enumerate(ids):
        c = np.mean(corners[i][0], axis=0)
        id_map[id_val[0]] = c

    if not all(k in id_map for k in [0, 1, 2, 3]):
        return None, "Missing ID 0,1,2, or 3"

    # Define Resolution (High Res for Accuracy)
    # We map the physical workspace to a 1000px wide image
    target_width = 1000
    ratio = WORKSPACE_HEIGHT_MM / WORKSPACE_WIDTH_MM
    target_height = int(target_width * ratio)

    # Source Points (From Camera)
    src = np.float32([id_map[0], id_map[1], id_map[2], id_map[3]])

    # Dest Points (Flat Image)
    dst = np.float32([
        [0, 0],
        [target_width, 0],
        [target_width, target_height],
        [0, target_height]
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, matrix, (target_width, target_height))

    return warped, "OK"


def find_objects(img):
    """
    Finds all blobs in the warped image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 100:  # Filter noise
            valid_contours.append(c)

    return valid_contours


def measure_results():
    """
    Calculates sizes based on the reference coin.
    """
    if state.pixels_per_mm is None: return

    state.results = []

    for c in state.contours:
        # Skip the reference coin itself
        if c is state.reference_contour: continue

        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect

        # Convert to MM
        dim1 = min(w, h) / state.pixels_per_mm
        dim2 = max(w, h) / state.pixels_per_mm

        # Filter Logic (Is it a pellet?)
        if dim1 < 0.5: continue  # Too small
        aspect = dim2 / dim1
        if aspect > 4.0: continue  # Too long (scratch/hair)

        is_good = (
                abs(dim1 - TARGET_DIAMETER) <= TOLERANCE and
                abs(dim2 - TARGET_LENGTH) <= TOLERANCE
        )

        box = np.intp(cv2.boxPoints(rect))
        state.results.append({
            'box': box, 'cx': int(cx), 'cy': int(cy),
            'd': dim1, 'l': dim2, 'ok': is_good
        })


# ----------------------------------------------------------------------
# MOUSE INPUT
# ----------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    if state.mode != 'ANALYSIS': return

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check which contour was clicked
        clicked = None
        for c in state.contours:
            if cv2.pointPolygonTest(c, (x, y), False) >= 0:
                clicked = c
                break

        if clicked is not None:
            state.reference_contour = clicked

            # Calculate Scale
            rect = cv2.minAreaRect(clicked)
            # Use average of width/height for circular coin
            avg_px = (rect[1][0] + rect[1][1]) / 2.0

            state.pixels_per_mm = avg_px / REFERENCE_DIAMETER_MM
            print(f"✓ Calibrated off Coin: {state.pixels_per_mm:.2f} px/mm")

            # Update Measurements
            measure_results()


# ----------------------------------------------------------------------
# MAIN UI
# ----------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    cv2.namedWindow("Hybrid Inspector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Hybrid Inspector", mouse_callback)

    print("--- HYBRID ACCURACY SYSTEM ---")
    print("1. Place ArUco markers to define workspace.")
    print(f"2. Place Coin ({REFERENCE_DIAMETER_MM}mm) and Pellets inside.")
    print("3. Press SPACE to Flatten & Capture.")
    print("4. Click the Coin to calibrate.")

    while True:
        if state.mode == 'LIVE':
            ret, frame = cap.read()
            if not ret: break
            state.frame = frame.copy()

            # Visualize Markers
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = state.detector.detectMarkers(gray)
            aruco.drawDetectedMarkers(frame, corners, ids)

            cv2.putText(frame, "LIVE: Press SPACE to Capture", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Hybrid Inspector", frame)

        elif state.mode == 'ANALYSIS':
            output = state.warped_view.copy()

            # Draw Instructions
            if state.pixels_per_mm is None:
                cv2.putText(output, "CLICK ON THE COIN", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Draw Reference Coin
                rect = cv2.minAreaRect(state.reference_contour)
                box = np.intp(cv2.boxPoints(rect))
                cv2.drawContours(output, [box], 0, (0, 255, 255), 2)
                cv2.putText(output, "REF", (int(rect[0][0]), int(rect[0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Draw Pellets
                good_cnt = 0
                bad_cnt = 0
                for r in state.results:
                    col = (0, 255, 0) if r['ok'] else (0, 0, 255)
                    cv2.drawContours(output, [r['box']], 0, col, 2)
                    lbl = f"{r['d']:.2f}x{r['l']:.2f}"
                    cv2.putText(output, lbl, (r['cx'] - 30, r['cy'] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
                    cv2.putText(output, lbl, (r['cx'] - 30, r['cy'] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    if r['ok']:
                        good_cnt += 1
                    else:
                        bad_cnt += 1

                # Stats
                h = output.shape[0]
                cv2.rectangle(output, (0, h - 60), (1000, h), (30, 30, 30), -1)
                cv2.putText(output, f"GOOD: {good_cnt} | BAD: {bad_cnt}", (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Hybrid Inspector", output)

        # Inputs
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # Reset
            state.mode = 'LIVE'
            state.pixels_per_mm = None
            state.reference_contour = None
            state.results = []

        elif key == 32:  # SPACE
            if state.mode == 'LIVE':
                # Warp Image (Flatten)
                warped, msg = get_warped_workspace(state.frame)
                if warped is not None:
                    state.warped_view = warped
                    state.mode = 'ANALYSIS'
                    # Pre-calculate contours
                    state.contours = find_objects(state.warped_view)
                    print("View Warped. Finding objects...")
                else:
                    print(f"Cannot capture: {msg}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()