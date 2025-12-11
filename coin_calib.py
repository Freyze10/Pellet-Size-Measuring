import cv2
import numpy as np

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
REAL_COIN_DIAMETER_MM = 23.0  # NGC 1-Piso
MIN_PELLET_AREA = 100  # Minimum size for a pellet


def process_frame_hybrid(frame):
    """
    Uses Hough Circles for the Coin, and Contours for Pellets.
    """
    display_img = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. PRE-PROCESSING
    # Blur helps ignore the "Jose Rizal" face details and focus on the round shape
    blur = cv2.medianBlur(gray, 5)

    # 2. FIND THE COIN (HOUGH CIRCLES)
    # This method is much better for coins than finding contours
    rows = blur.shape[0]
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=rows / 8,  # Minimum distance between circles
        param1=100,  # High threshold for Canny
        param2=30,  # Accumulator threshold (Lower = detects more circles)
        minRadius=20,  # Min size (pixels)
        maxRadius=150  # Max size (pixels)
    )

    pixels_per_mm = None

    # If we found a circle (The Coin)
    if circles is not None:
        circles = np.uint16(np.around(circles))

        # We assume the largest circle found is the Peso coin
        largest_circle = max(circles[0, :], key=lambda x: x[2])
        center_x, center_y, radius = largest_circle

        # Draw the detected coin
        cv2.circle(display_img, (center_x, center_y), radius, (0, 255, 255), 2)
        cv2.circle(display_img, (center_x, center_y), 2, (0, 0, 255), 3)

        # Calculate Scale
        pixel_diameter = radius * 2
        pixels_per_mm = pixel_diameter / REAL_COIN_DIAMETER_MM

        cv2.putText(display_img, f"REF: {pixels_per_mm:.2f} px/mm",
                    (center_x - 60, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(display_img, "COIN NOT FOUND", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 3. FIND PELLETS (CONTOURS)
    # We still use contours for pellets because pellets aren't perfect circles

    # Stronger edge detection for pellets
    edges = cv2.Canny(blur, 50, 150)
    # Dilate slightly to close gaps in dark pellets
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pellet_count = 0
    if pixels_per_mm:
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Filter: Must be big enough to be a pellet, but not HUGE (the coin)
            # We assume the pellet is significantly smaller than the coin detected above
            if area < MIN_PELLET_AREA or area > (np.pi * (radius ** 2) * 0.8):
                continue

            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect

            dim1 = w / pixels_per_mm
            dim2 = h / pixels_per_mm

            diameter = min(dim1, dim2)
            length = max(dim1, dim2)

            # Visualize
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            cv2.drawContours(display_img, [box], 0, (0, 255, 0), 1)

            label = f"{diameter:.2f} x {length:.2f}"
            cv2.putText(display_img, label, (int(cx) - 20, int(cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            pellet_count += 1

    return display_img, edges  # Return image AND the debug view


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    print("System Started.")
    print("Look at the 'Debug View' window.")
    print("If the coin is not a white circle in Debug View, fix your lighting.")

    is_frozen = False
    frozen_frame_display = None
    debug_view = None

    while True:
        if is_frozen:
            cv2.imshow("Inspector", frozen_frame_display)
            if debug_view is not None:
                cv2.imshow("Debug View (Edges)", debug_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 32:  # Spacebar
                is_frozen = False
                print("Live View...")
            continue

        ret, frame = cap.read()
        if not ret: break

        # Run detection LIVE so you can adjust lighting before capturing
        display_frame, edges_frame = process_frame_hybrid(frame.copy())

        cv2.imshow("Inspector", display_frame)
        cv2.imshow("Debug View (Edges)", edges_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # Spacebar
            print("Capturing...")
            frozen_frame_display = display_frame
            debug_view = edges_frame
            is_frozen = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()