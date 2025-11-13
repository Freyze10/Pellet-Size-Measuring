import cv2
import numpy as np
import time

# Configuration
PIXEL_TO_MM = 1 / 20  # Calibration: 20 pixels = 1 mm
TARGET_DIAMETER = 3.0  # mm
TARGET_LENGTH = 3.0  # mm
TOLERANCE = 0.5  # mm
EXCLUSION_THRESHOLD = 1.0  # mm - objects beyond this are ignored

# Calculate ranges
DIAMETER_MIN = TARGET_DIAMETER - TOLERANCE
DIAMETER_MAX = TARGET_DIAMETER + TOLERANCE
LENGTH_MIN = TARGET_LENGTH - TOLERANCE
LENGTH_MAX = TARGET_LENGTH + TOLERANCE

# Exclusion ranges
DIAMETER_EXCLUDE_MIN = TARGET_DIAMETER - EXCLUSION_THRESHOLD
DIAMETER_EXCLUDE_MAX = TARGET_DIAMETER + EXCLUSION_THRESHOLD
LENGTH_EXCLUDE_MIN = TARGET_LENGTH - EXCLUSION_THRESHOLD
LENGTH_EXCLUDE_MAX = TARGET_LENGTH + EXCLUSION_THRESHOLD

# Minimum contour area to avoid noise (in pixels)
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 10000


def is_within_tolerance(diameter, length):
    """Check if pellet is within acceptable tolerance."""
    return (DIAMETER_MIN <= diameter <= DIAMETER_MAX and
            LENGTH_MIN <= length <= LENGTH_MAX)


def should_process_pellet(diameter, length):
    """Check if pellet should be processed (not excluded)."""
    return (DIAMETER_EXCLUDE_MIN <= diameter <= DIAMETER_EXCLUDE_MAX and
            LENGTH_EXCLUDE_MIN <= length <= LENGTH_EXCLUDE_MAX)


def detect_pellets(frame):
    """Detect pellets in the frame and return their measurements."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive threshold for better edge detection
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Apply morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    pellets = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter by area
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate dimensions in mm
        # For cylindrical pellets viewed from side:
        # - Diameter is typically the smaller dimension
        # - Length is typically the larger dimension
        width_mm = w * PIXEL_TO_MM
        height_mm = h * PIXEL_TO_MM

        # Assign diameter and length based on orientation
        if w < h:  # Vertical orientation
            diameter = width_mm
            length = height_mm
        else:  # Horizontal orientation
            diameter = height_mm
            length = width_mm

        # Check if pellet should be processed
        if should_process_pellet(diameter, length):
            within_tolerance = is_within_tolerance(diameter, length)

            pellets.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'diameter': diameter,
                'length': length,
                'within_tolerance': within_tolerance
            })

    return pellets


def draw_overlay(frame, pellets):
    """Draw measurements and status on the frame."""
    # Determine overall status
    if not pellets:
        status_text = "No pellets detected"
        status_color = (128, 128, 128)  # Gray
    elif all(p['within_tolerance'] for p in pellets):
        status_text = "✅ Within tolerance"
        status_color = (0, 255, 0)  # Green
    else:
        status_text = "❌ Out of tolerance"
        status_color = (0, 0, 255)  # Red

    # Draw status indicator at top-left
    cv2.rectangle(frame, (10, 10), (300, 50), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (300, 50), status_color, 2)
    cv2.putText(frame, status_text, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Draw each pellet
    for pellet in pellets:
        x, y, w, h = pellet['x'], pellet['y'], pellet['w'], pellet['h']
        diameter = pellet['diameter']
        length = pellet['length']
        within_tolerance = pellet['within_tolerance']

        # Draw rectangle around pellet (green for valid)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Prepare measurement text
        text_diameter = f"Dia: {diameter:.2f} mm"
        text_length = f"Len: {length:.2f} mm"

        # Draw background for text
        text_bg_y = max(y - 45, 0)
        cv2.rectangle(frame, (x, text_bg_y), (x + 150, y - 5), (0, 0, 0), -1)

        # Draw text
        cv2.putText(frame, text_diameter, (x + 5, text_bg_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, text_length, (x + 5, text_bg_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Add warning indicator if out of tolerance
        if not within_tolerance:
            cv2.circle(frame, (x + w - 10, y + 10), 8, (0, 0, 255), -1)
            cv2.putText(frame, "!", (x + w - 14, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame


def main():
    """Main function to run the pellet measurement system."""
    # Initialize camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # FPS calculation variables
    fps_start_time = time.time()
    fps_counter = 0
    fps_display = 0

    print("Pellet Size Measurement System")
    print("=" * 50)
    print(f"Target Diameter: {TARGET_DIAMETER} mm (±{TOLERANCE} mm)")
    print(f"Target Length: {TARGET_LENGTH} mm (±{TOLERANCE} mm)")
    print(f"Acceptable Range: {DIAMETER_MIN}-{DIAMETER_MAX} mm")
    print(f"Calibration: {1 / PIXEL_TO_MM} pixels = 1 mm")
    print("=" * 50)
    print("Press 'q' to quit")
    print("Press 'c' to adjust calibration (coming soon)")
    print()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame")
            break

        # Detect pellets
        pellets = detect_pellets(frame)

        # Draw overlay
        frame = draw_overlay(frame, pellets)

        # Calculate and display FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        cv2.putText(frame, f"FPS: {fps_display}", (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display pellet count
        cv2.putText(frame, f"Pellets: {len(pellets)}", (frame.shape[1] - 120, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show frame
        cv2.imshow("Pellet Size Measurement", frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nSystem shutdown complete")


if __name__ == "__main__":
    main()