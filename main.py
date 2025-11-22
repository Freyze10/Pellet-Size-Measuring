import cv2
import numpy as np
import time
import math

# ----------------------------------------------------------------------
# Global Config
# ----------------------------------------------------------------------
PIXELS_PER_MM = 6.0
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0

# Ranges (Updated dynamically)
DIAMETER_MIN = 0
DIAMETER_MAX = 0
LENGTH_MIN = 0
LENGTH_MAX = 0

# Calibration State
in_ruler_calib_mode = False
calibration_frozen_frame = None
ref_start = None
ref_end = None
REFERENCE_MM = 76.2  # 3 inches


def update_ranges():
    global DIAMETER_MIN, DIAMETER_MAX, LENGTH_MIN, LENGTH_MAX
    DIAMETER_MIN = TARGET_DIAMETER - TOLERANCE
    DIAMETER_MAX = TARGET_DIAMETER + TOLERANCE
    LENGTH_MIN = TARGET_LENGTH - TOLERANCE
    LENGTH_MAX = TARGET_LENGTH + TOLERANCE


update_ranges()


# ----------------------------------------------------------------------
# IMPROVED DETECTION LOGIC
# ----------------------------------------------------------------------
def detect_pellets_advanced(frame, pixels_per_mm):
    # 1. Pre-processing
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Thresholding (Otsu is usually better than Adaptive for objects on contrast bg)
    # If you have backlighting (silhouettes), use THRESH_BINARY_INV
    # If you have top lighting (bright pellets on dark bg), use THRESH_BINARY
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Noise Removal (Morphological Opening)
    # This removes small white dots (noise) from the background
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4. Sure Background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 5. Finding Sure Foreground area (Distance Transform)
    # This helps separate touching pellets
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # Threshold the distance transform to get the "peaks" (centers of pellets)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 6. Unknown region (borders between touching pellets)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 7. Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # 8. Apply Watershed
    # formatting image for watershed (needs 3 channel)
    markers = cv2.watershed(frame, markers)

    pellets = []

    # Loop through unique markers (skipping 0=unknown, 1=background)
    for label in np.unique(markers):
        if label <= 1:
            continue

        # Create a mask for this specific object
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == label] = 255

        # Find contours on this specific mask
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) > 0:
            c = cnts[0]
            area = cv2.contourArea(c)

            # Basic noise filter by area
            if area < 50 or area > 10000:
                continue

            # Get Rotated Rectangle
            rect = cv2.minAreaRect(c)
            (center, (w, h), angle) = rect

            # Sort dimensions (w vs h)
            dim1 = w / pixels_per_mm
            dim2 = h / pixels_per_mm

            diameter = min(dim1, dim2)
            length = max(dim1, dim2)

            # Convert box points to int for drawing
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # Logic Check: Don't process extremely tiny shards
            if diameter < 0.5: continue

            is_valid = (DIAMETER_MIN <= diameter <= DIAMETER_MAX and
                        LENGTH_MIN <= length <= LENGTH_MAX)

            pellets.append({
                'box': box,
                'center': center,
                'diameter': diameter,
                'length': length,
                'valid': is_valid
            })

    return pellets, markers


# ----------------------------------------------------------------------
# Main Loop Components
# ----------------------------------------------------------------------
def mouse_calib(event, x, y, flags, param):
    global ref_start, ref_end, in_ruler_calib_mode, PIXELS_PER_MM
    if not in_ruler_calib_mode: return

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_start = (x, y)
        ref_end = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        ref_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        ref_end = (x, y)
        # Calculate
        dist_px = math.sqrt((ref_end[0] - ref_start[0]) ** 2 + (ref_end[1] - ref_start[1]) ** 2)
        if dist_px > 0:
            PIXELS_PER_MM = dist_px / REFERENCE_MM
            print(f"Calibration Update: {PIXELS_PER_MM:.2f} px/mm")


def main():
    global in_ruler_calib_mode, calibration_frozen_frame

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Set highest res possible for better measurement accuracy
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Inspector")
    cv2.setMouseCallback("Inspector", mouse_calib)

    while True:
        ret, frame = cap.read()
        if not ret: break

        display = frame.copy()

        if in_ruler_calib_mode:
            if calibration_frozen_frame is None:
                calibration_frozen_frame = frame.copy()
            display = calibration_frozen_frame.copy()

            # Draw instructions
            cv2.putText(display, f"Draw line for {REFERENCE_MM}mm (3 inches)", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if ref_start and ref_end:
                cv2.line(display, ref_start, ref_end, (0, 255, 0), 2)
                mid = ((ref_start[0] + ref_end[0]) // 2, (ref_start[1] + ref_end[1]) // 2)
                cv2.putText(display, f"{PIXELS_PER_MM:.2f} px/mm", mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            calibration_frozen_frame = None
            # Run detection
            pellets, markers = detect_pellets_advanced(frame, PIXELS_PER_MM)

            # Visualization
            # Optional: Visualize Watershed markers to debug separation
            # marker_vis = np.uint8(markers * 10) # Scale for visibility
            # cv2.imshow("Debug Markers", marker_vis)

            for p in pellets:
                color = (0, 255, 0) if p['valid'] else (0, 0, 255)
                cv2.drawContours(display, [p['box']], 0, color, 2)

                # Label
                lbl = f"D:{p['diameter']:.1f} L:{p['length']:.1f}"
                cv2.putText(display, lbl, (p['box'][1][0], p['box'][1][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.putText(display, "Press 'c' to Calibrate", (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Inspector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('c'): in_ruler_calib_mode = not in_ruler_calib_mode

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()