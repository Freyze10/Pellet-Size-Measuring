import cv2
import numpy as np
import time
import sys
import math
from collections import Counter

# ----------------------------------------------------------------------
# Global Calibration
# ----------------------------------------------------------------------
PIXELS_PER_MM = 6.0
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 1.0


def update_ranges():
    global DIAMETER_MIN, DIAMETER_MAX, LENGTH_MIN, LENGTH_MAX
    global DIAMETER_EXCLUDE_MIN, DIAMETER_EXCLUDE_MAX
    global LENGTH_EXCLUDE_MIN, LENGTH_EXCLUDE_MAX

    DIAMETER_MIN = TARGET_DIAMETER - TOLERANCE
    DIAMETER_MAX = TARGET_DIAMETER + TOLERANCE
    LENGTH_MIN = TARGET_LENGTH - TOLERANCE
    LENGTH_MAX = TARGET_LENGTH + TOLERANCE

    DIAMETER_EXCLUDE_MIN = TARGET_DIAMETER - EXCLUSION_THRESHOLD
    DIAMETER_EXCLUDE_MAX = TARGET_DIAMETER + EXCLUSION_THRESHOLD
    LENGTH_EXCLUDE_MIN = TARGET_LENGTH - EXCLUSION_THRESHOLD
    LENGTH_EXCLUDE_MAX = TARGET_LENGTH + EXCLUSION_THRESHOLD


update_ranges()

MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 20000

# ----------------------------------------------------------------------
# Auto Calibration State
# ----------------------------------------------------------------------
in_auto_calib_mode = False
auto_calib_frozen_frame = None
detected_ruler_data = None


# ----------------------------------------------------------------------
# Helper Checks
# ----------------------------------------------------------------------
def is_within_tolerance(diameter: float, length: float) -> bool:
    return (DIAMETER_MIN <= diameter <= DIAMETER_MAX and
            LENGTH_MIN <= length <= LENGTH_MAX)


def should_process_pellet(diameter: float, length: float) -> bool:
    return (DIAMETER_EXCLUDE_MIN <= diameter <= DIAMETER_EXCLUDE_MAX and
            LENGTH_EXCLUDE_MIN <= length <= LENGTH_EXCLUDE_MAX)


def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ----------------------------------------------------------------------
# AUTOMATIC RULER DETECTION
# ----------------------------------------------------------------------
def detect_ruler_automatically(frame):
    """
    Automatically detect ruler tick marks and calculate calibration.
    Works with partial rulers - only needs 2-3 visible tick marks.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Enhance edges
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=30, maxLineGap=10)

    if lines is None:
        return None

    # Filter for mostly horizontal or vertical lines (rulers are usually straight)
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = get_distance((x1, y1), (x2, y2))
        angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / np.pi)

        # Keep lines that are mostly horizontal (0-20° or 160-180°) or vertical (80-100°)
        if length > 50 and (angle < 20 or angle > 160 or (80 < angle < 100)):
            filtered_lines.append(((x1, y1), (x2, y2), length, angle))

    if len(filtered_lines) < 2:
        return None

    # Group parallel lines (likely tick marks)
    tick_marks = find_tick_mark_groups(filtered_lines)

    if not tick_marks:
        return None

    # Analyze tick spacing to determine calibration
    calibration = analyze_tick_spacing(tick_marks)

    return calibration


def find_tick_mark_groups(lines):
    """
    Find groups of parallel, evenly-spaced lines (tick marks).
    """
    if len(lines) < 3:
        return []

    # Sort lines by angle to group parallel ones
    lines_sorted = sorted(lines, key=lambda x: x[3])

    groups = []
    current_group = [lines_sorted[0]]

    for i in range(1, len(lines_sorted)):
        prev_angle = lines_sorted[i - 1][3]
        curr_angle = lines_sorted[i][3]

        # If angles are within 10 degrees, consider them parallel
        if abs(curr_angle - prev_angle) < 10:
            current_group.append(lines_sorted[i])
        else:
            if len(current_group) >= 3:
                groups.append(current_group)
            current_group = [lines_sorted[i]]

    if len(current_group) >= 3:
        groups.append(current_group)

    return groups


def analyze_tick_spacing(tick_groups):
    """
    Analyze spacing between tick marks to determine mm/inch calibration.
    Returns calibration data with detected spacing.
    """
    best_calibration = None

    for group in tick_groups:
        if len(group) < 3:
            continue

        # Get midpoints of each tick mark
        midpoints = []
        for (x1, y1), (x2, y2), length, angle in group:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            midpoints.append((mid_x, mid_y))

        # Sort midpoints along the ruler direction
        if len(midpoints) < 3:
            continue

        # Determine if ruler is more horizontal or vertical
        angle = group[0][3]
        if angle < 45 or angle > 135:
            # Horizontal ruler - sort by x
            midpoints = sorted(midpoints, key=lambda p: p[0])
        else:
            # Vertical ruler - sort by y
            midpoints = sorted(midpoints, key=lambda p: p[1])

        # Calculate spacings between consecutive ticks
        spacings = []
        for i in range(len(midpoints) - 1):
            dist = get_distance(midpoints[i], midpoints[i + 1])
            spacings.append(dist)

        if not spacings:
            continue

        # Find the most common spacing (handles minor variations)
        spacing_rounded = [round(s, 1) for s in spacings]
        spacing_counter = Counter(spacing_rounded)
        most_common_spacing = spacing_counter.most_common(1)[0][0]

        # Filter spacings close to the most common one
        consistent_spacings = [s for s in spacings if abs(s - most_common_spacing) < most_common_spacing * 0.2]

        if len(consistent_spacings) < 2:
            continue

        avg_spacing = sum(consistent_spacings) / len(consistent_spacings)

        # Try to identify if it's mm, cm, or inch marks
        # Common patterns:
        # - 1mm marks: very close together (typically 2-10 pixels at normal distances)
        # - 1cm marks: ~10x further (typically 20-100 pixels)
        # - 1 inch marks: ~25.4x mm spacing (typically 50-250 pixels)

        calibration_options = []

        # Test if it's 1mm spacing
        px_per_mm_if_1mm = avg_spacing / 1.0
        if 1.5 < px_per_mm_if_1mm < 15:  # Reasonable range for 1mm marks
            calibration_options.append({
                'px_per_mm': px_per_mm_if_1mm,
                'spacing_type': '1mm',
                'confidence': len(consistent_spacings),
                'avg_spacing_px': avg_spacing,
                'tick_positions': midpoints,
                'group': group
            })

        # Test if it's 1cm (10mm) spacing
        px_per_mm_if_1cm = avg_spacing / 10.0
        if 1.5 < px_per_mm_if_1cm < 15:
            calibration_options.append({
                'px_per_mm': px_per_mm_if_1cm,
                'spacing_type': '1cm',
                'confidence': len(consistent_spacings) * 1.5,  # Prefer cm marks
                'avg_spacing_px': avg_spacing,
                'tick_positions': midpoints,
                'group': group
            })

        # Test if it's 0.5 inch (12.7mm) spacing
        px_per_mm_if_half_inch = avg_spacing / 12.7
        if 1.5 < px_per_mm_if_half_inch < 15:
            calibration_options.append({
                'px_per_mm': px_per_mm_if_half_inch,
                'spacing_type': '0.5 inch',
                'confidence': len(consistent_spacings) * 1.3,
                'avg_spacing_px': avg_spacing,
                'tick_positions': midpoints,
                'group': group
            })

        # Test if it's 1 inch (25.4mm) spacing
        px_per_mm_if_1inch = avg_spacing / 25.4
        if 1.5 < px_per_mm_if_1inch < 15:
            calibration_options.append({
                'px_per_mm': px_per_mm_if_1inch,
                'spacing_type': '1 inch',
                'confidence': len(consistent_spacings) * 1.2,
                'avg_spacing_px': avg_spacing,
                'tick_positions': midpoints,
                'group': group
            })

        # Pick best calibration from this group
        if calibration_options:
            best_from_group = max(calibration_options, key=lambda x: x['confidence'])
            if best_calibration is None or best_from_group['confidence'] > best_calibration['confidence']:
                best_calibration = best_from_group

    return best_calibration


def draw_auto_calibration_ui(frame, ruler_data):
    """
    Draw the auto-calibration interface with detected ruler visualization.
    """
    overlay = frame.copy()

    # Semi-transparent panel
    panel_x, panel_y, panel_w, panel_h = 10, 80, 450, 300
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (30, 30, 50), -1)
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (100, 150, 255), 3)

    cv2.putText(overlay, "AUTO RULER CALIBRATION", (panel_x + 70, panel_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    y_offset = panel_y + 60

    if ruler_data is None:
        # No ruler detected
        instructions = [
            "Searching for ruler...",
            "",
            "Tips:",
            "- Place ruler flat in frame",
            "- Ensure good lighting",
            "- Show at least 2-3 tick marks",
            "- Works with partial rulers!",
            "",
            "Press 'a' again to retry",
            "Press 'ESC' to cancel"
        ]

        for instr in instructions:
            cv2.putText(overlay, instr, (panel_x + 20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
    else:
        # Ruler detected - show results
        info_lines = [
            f"✓ Ruler detected!",
            f"",
            f"Detected spacing: {ruler_data['spacing_type']}",
            f"Average spacing: {ruler_data['avg_spacing_px']:.1f} px",
            f"Tick marks found: {len(ruler_data['tick_positions'])}",
            f"",
            f"Calculated calibration:",
            f"  {ruler_data['px_per_mm']:.3f} pixels per mm",
            f"",
            f"Current: {PIXELS_PER_MM:.3f} px/mm",
        ]

        for line in info_lines:
            color = (100, 255, 100) if "✓" in line else (200, 200, 200)
            cv2.putText(overlay, line, (panel_x + 20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 23

    # Buttons
    btn_y = panel_y + panel_h - 50

    if ruler_data is not None:
        # Apply button (green when ruler detected)
        cv2.rectangle(overlay, (panel_x + 30, btn_y),
                      (panel_x + 150, btn_y + 35), (0, 200, 0), -1)
        cv2.putText(overlay, "APPLY (Enter)", (panel_x + 40, btn_y + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Retry button
    cv2.rectangle(overlay, (panel_x + 170, btn_y),
                  (panel_x + 290, btn_y + 35), (100, 100, 200), -1)
    cv2.putText(overlay, "RETRY (a)", (panel_x + 185, btn_y + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Cancel button
    cv2.rectangle(overlay, (panel_x + 310, btn_y),
                  (panel_x + 430, btn_y + 35), (100, 100, 100), -1)
    cv2.putText(overlay, "CANCEL (ESC)", (panel_x + 320, btn_y + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Draw detected ruler marks on the frame
    if ruler_data is not None:
        # Draw tick marks
        for (x1, y1), (x2, y2), length, angle in ruler_data['group']:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        # Draw connecting line through tick midpoints
        positions = ruler_data['tick_positions']
        for i in range(len(positions) - 1):
            p1 = (int(positions[i][0]), int(positions[i][1]))
            p2 = (int(positions[i + 1][0]), int(positions[i + 1][1]))
            cv2.line(frame, p1, p2, (255, 0, 255), 1)
            cv2.circle(frame, p1, 4, (0, 255, 0), -1)

        # Mark the last point
        if positions:
            last = (int(positions[-1][0]), int(positions[-1][1]))
            cv2.circle(frame, last, 4, (0, 255, 0), -1)

        # Show spacing between first two marks
        if len(positions) >= 2:
            mid_x = int((positions[0][0] + positions[1][0]) / 2)
            mid_y = int((positions[0][1] + positions[1][1]) / 2)
            cv2.putText(frame, f"{ruler_data['avg_spacing_px']:.1f}px",
                        (mid_x + 10, mid_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 2)


# ----------------------------------------------------------------------
# Detection with Rotated Bounding Boxes
# ----------------------------------------------------------------------
def detect_pellets(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pellets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA):
            continue

        rect = cv2.minAreaRect(cnt)
        (center_x, center_y), (w, h), angle = rect
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        edge1 = get_distance(box[0], box[1])
        edge2 = get_distance(box[1], box[2])

        if edge1 < edge2:
            width_px = edge1
            height_px = edge2
        else:
            width_px = edge2
            height_px = edge1

        width_mm = width_px / PIXELS_PER_MM
        height_mm = height_px / PIXELS_PER_MM

        diameter = width_mm
        length = height_mm

        if should_process_pellet(diameter, length):
            pellets.append({
                'contour': cnt,
                'box': box,
                'center': (center_x, center_y),
                'angle': angle,
                'width_px': width_px,
                'height_px': height_px,
                'diameter': diameter,
                'length': length,
                'within_tolerance': is_within_tolerance(diameter, length)
            })
    return pellets


# ----------------------------------------------------------------------
# Main Overlay
# ----------------------------------------------------------------------
def draw_overlay(frame, pellets):
    total = len(pellets)
    within = sum(1 for p in pellets if p['within_tolerance'])
    out_of = total - within
    status_text = f"In: {within}   Out: {out_of}   Total: {total}"
    status_color = (0, 255, 0) if out_of == 0 else (0, 0, 255)

    cv2.rectangle(frame, (10, 10), (460, 50), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (460, 50), status_color, 2)
    cv2.putText(frame, status_text, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

    for p in pellets:
        box = p['box']
        center = p['center']
        color = (0, 255, 0) if p['within_tolerance'] else (0, 0, 255)

        cv2.drawContours(frame, [box], 0, color, 1)
        cv2.circle(frame, (int(center[0]), int(center[1])), 2, color, -1)

        top_y = int(min(box[:, 1]))
        left_x = int(min(box[:, 0]))
        bg_y = max(top_y - 30, 0)

        cv2.rectangle(frame, (left_x, bg_y), (left_x + 70, top_y - 5), (0, 0, 0), -1)
        cv2.putText(frame, f"D: {p['diameter']:.2f}mm", (left_x + 3, bg_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(frame, f"L: {p['length']:.2f}mm", (left_x + 3, bg_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        if not p['within_tolerance']:
            top_right = box[np.argmax(box[:, 0])]
            cv2.circle(frame, tuple(top_right), 8, (0, 0, 255), -1)
            cv2.putText(frame, "!", (top_right[0] - 4, top_right[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if not in_auto_calib_mode:
        cv2.putText(frame, "Press 'a' for AUTO calibration | 'q' to quit",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)

    return frame


# ----------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------
def get_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    return cap


# ----------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------
def main():
    global in_auto_calib_mode, auto_calib_frozen_frame, detected_ruler_data, PIXELS_PER_MM

    print("\n╔═══════════════════════════════════════════════════════════╗")
    print("║     Pellet Inspector - AUTO RULER CALIBRATION v2.0       ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print("\n📏 NEW: Automatic ruler detection!")
    print("   • Works with PARTIAL rulers (just 2-3 tick marks needed)")
    print("   • Detects mm, cm, and inch markings")
    print("   • Any angle, any position\n")
    print("Controls:")
    print("  'a' → Auto-detect ruler and calibrate")
    print("  'q' → Quit\n")
    print("═" * 61)

    cap = get_camera()
    if not cap.isOpened():
        print("❌ Cannot open camera.")
        sys.exit(1)

    fps_counter = 0
    fps_start = time.time()
    fps_display = 0

    window_name = "Pellet Inspector - Auto Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera lost – reconnecting...")
            cap.release()
            time.sleep(1)
            cap = get_camera()
            if not cap.isOpened():
                break
            continue

        # Handle auto-calibration mode
        if in_auto_calib_mode:
            if auto_calib_frozen_frame is None:
                auto_calib_frozen_frame = frame.copy()
                print("🔍 Analyzing frame for ruler...")
                detected_ruler_data = detect_ruler_automatically(auto_calib_frozen_frame)
                if detected_ruler_data:
                    print(f"✓ Ruler detected! Type: {detected_ruler_data['spacing_type']}")
                    print(f"  Calibration: {detected_ruler_data['px_per_mm']:.3f} px/mm")
                else:
                    print("⚠ No ruler detected. Try repositioning.")

            display_frame = auto_calib_frozen_frame.copy()
            draw_auto_calibration_ui(display_frame, detected_ruler_data)
        else:
            auto_calib_frozen_frame = None
            detected_ruler_data = None
            display_frame = frame.copy()
            pellets = detect_pellets(display_frame)
            display_frame = draw_overlay(display_frame, pellets)

        # FPS counter
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_counter // int(elapsed)
            fps_counter = 0
            fps_start = time.time()

        cv2.putText(display_frame, f"FPS: {fps_display}",
                    (display_frame.shape[1] - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('a'):
            if not in_auto_calib_mode:
                print("\n📸 Entering auto-calibration mode...")
                in_auto_calib_mode = True
            else:
                # Retry detection
                print("🔄 Retrying detection...")
                auto_calib_frozen_frame = None
        elif key == 27:  # ESC
            if in_auto_calib_mode:
                print("❌ Calibration cancelled.")
                in_auto_calib_mode = False
                auto_calib_frozen_frame = None
                detected_ruler_data = None
        elif key == 13:  # Enter
            if in_auto_calib_mode and detected_ruler_data is not None:
                # Apply calibration
                old_value = PIXELS_PER_MM
                PIXELS_PER_MM = detected_ruler_data['px_per_mm']
                update_ranges()
                print(f"\n✓ CALIBRATION APPLIED!")
                print(f"  Old: {old_value:.3f} px/mm")
                print(f"  New: {PIXELS_PER_MM:.3f} px/mm")
                print(f"  Based on: {detected_ruler_data['spacing_type']} marks\n")
                in_auto_calib_mode = False
                auto_calib_frozen_frame = None
                detected_ruler_data = None

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Shutdown complete.")


if __name__ == "__main__":
    main()