import cv2
import numpy as np
import cv2.aruco as aruco


def create_calibration_sheet():
    # --- Configuration ---
    # A4 Size at 300 DPI (Landscape)
    PAGE_WIDTH_MM = 297
    PAGE_HEIGHT_MM = 210
    DPI = 300

    # Distance between marker CENTERS (Must match REAL_WIDTH_MM in your detection code)
    WORK_AREA_WIDTH_MM = 200
    WORK_AREA_HEIGHT_MM = 150

    # Size of the ArUco markers
    MARKER_SIZE_MM = 40

    # ---------------------

    # Conversion factor
    MM_TO_PX = DPI / 25.4

    # Create white blank A4 image
    width_px = int(PAGE_WIDTH_MM * MM_TO_PX)
    height_px = int(PAGE_HEIGHT_MM * MM_TO_PX)
    img = np.ones((height_px, width_px), dtype=np.uint8) * 255

    # Define Dictionary (4x4_50)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # Calculate Center of Page
    cx = width_px // 2
    cy = height_px // 2

    # Calculate Offsets (half of the work area size)
    dx = int((WORK_AREA_WIDTH_MM * MM_TO_PX) / 2)
    dy = int((WORK_AREA_HEIGHT_MM * MM_TO_PX) / 2)

    # Marker size in pixels
    m_px = int(MARKER_SIZE_MM * MM_TO_PX)
    half_m = m_px // 2

    # Positions (Center of markers):
    # ID 0: Top-Left
    # ID 1: Top-Right
    # ID 2: Bottom-Right
    # ID 3: Bottom-Left
    # (Matches the order expected by cv2.getPerspectiveTransform)

    positions = {
        0: (cx - dx, cy - dy),  # TL
        1: (cx + dx, cy - dy),  # TR
        2: (cx + dx, cy + dy),  # BR
        3: (cx - dx, cy + dy)  # BL
    }

    print(f"Generating sheet...")
    print(f"Distance between centers: {WORK_AREA_WIDTH_MM}mm x {WORK_AREA_HEIGHT_MM}mm")

    for marker_id, (px, py) in positions.items():
        # Generate marker
        marker_img = aruco.generateImageMarker(aruco_dict, marker_id, m_px)

        # Calculate top-left corner for pasting
        y1 = py - half_m
        y2 = py + half_m
        x1 = px - half_m
        x2 = px + half_m

        # Paste into image
        img[y1:y2, x1:x2] = marker_img

        # Add Text Label
        cv2.putText(img, f"ID: {marker_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # Draw Helper Lines (To verify print accuracy)
    # Horizontal Center Line
    pt1 = positions[0]
    pt2 = positions[1]
    cv2.line(img, pt1, pt2, (200, 200, 200), 2)
    cv2.putText(img, f"{WORK_AREA_WIDTH_MM} mm", (cx - 100, pt1[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (150, 150, 150), 3)

    # Vertical Center Line
    pt3 = positions[0]
    pt4 = positions[3]
    cv2.line(img, pt3, pt4, (200, 200, 200), 2)
    cv2.putText(img, f"{WORK_AREA_HEIGHT_MM} mm", (pt3[0] - 120, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (150, 150, 150), 3)

    # Save
    filename = "calibration_sheet_A4.png"
    cv2.imwrite(filename, img)
    print(f"✓ Saved to {filename}")
    print("IMPORTANT: Print at 100% Scale (Do not Scale to Fit)")


if __name__ == "__main__":
    create_calibration_sheet()