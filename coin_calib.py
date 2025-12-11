import cv2
import numpy as np
import time

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
REAL_COIN_DIAMETER_MM = 23.0  # NGC 1-Piso
MIN_AREA_THRESHOLD = 100  # Ignore dust
CANNY_THRESH_1 = 50
CANNY_THRESH_2 = 150

# Global Scale Variable (Starts empty)
LOCKED_PIXELS_PER_MM = None


def get_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, CANNY_THRESH_1, CANNY_THRESH_2)
    # Close small gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def perform_calibration(cap):
    """
    Captures 30 frames, finds the coin in each, averages the diameter.
    Returns the calculated Pixels_Per_MM.
    """
    print("--- CALIBRATING... DO NOT MOVE COIN ---")

    diameters = []
    frames_to_read = 30

    for _ in range(frames_to_read):
        ret, frame = cap.read()
        if not ret: continue

        contours = get_contours(frame)

        # Find largest object (Assumed to be the coin)
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_cnt)

            if area > 2000:  # Must be big enough to be a coin
                ellipse = cv2.fitEllipse(largest_cnt)
                (cx, cy), (w, h), angle = ellipse
                avg_diameter = (w + h) / 2.0
                diameters.append(avg_diameter)

        time.sleep(0.01)  # Small delay

    if len(diameters) < 10:
        print("Calibration Failed: Coin not seen clearly.")
        return None

    # Mathematical Average (removes the jitter/noise)
    final_avg_pixel_diameter = np.mean(diameters)
    scale = final_avg_pixel_diameter / REAL_COIN_DIAMETER_MM

    print(f"--- CALIBRATION LOCKED: {scale:.2f} px/mm ---")
    return scale


def main():
    global LOCKED_PIXELS_PER_MM

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # DISABLE AUTOFOCUS (Crucial)

    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()

        # ---------------------------------------------------------
        # MODE 1: NOT CALIBRATED YET
        # ---------------------------------------------------------
        if LOCKED_PIXELS_PER_MM is None:
            # Show instructions
            cv2.putText(display_frame, "STEP 1: Place 1-Piso Coin", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display_frame, "Press 'c' to Calibrate", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Show what the computer sees (to help you align)
            contours = get_contours(frame)
            for cnt in contours:
                if cv2.contourArea(cnt) > 2000:
                    cv2.drawContours(display_frame, [cnt], -1, (0, 255, 255), 2)

        # ---------------------------------------------------------
        # MODE 2: CALIBRATED (MEASURING PELLETS)
        # ---------------------------------------------------------
        else:
            contours = get_contours(frame)

            pellet_count = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)

                # Filter: Ignore tiny noise AND ignore the huge coin if it's still there
                # We assume a pellet is roughly between 100 and 3000 pixels (adjust as needed)
                if area < MIN_AREA_THRESHOLD: continue

                # Check if this object is the Coin (so we don't measure the coin as a pellet)
                # If area is huge (like > 80% of what the coin was), ignore it.
                # (Simple heuristic: Pellets are small)
                if area > 10000:  # Adjust this limit if your coin is huge on screen
                    cv2.drawContours(display_frame, [cnt], -1, (100, 100, 100), 1)  # Draw coin gray
                    continue

                # MEASURE THE PELLET
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect

                # Apply the LOCKED scale
                dim1 = w / LOCKED_PIXELS_PER_MM
                dim2 = h / LOCKED_PIXELS_PER_MM

                diameter = min(dim1, dim2)
                length = max(dim1, dim2)

                # Draw
                box = cv2.boxPoints(rect)
                box = np.int64(box)
                cv2.drawContours(display_frame, [box], 0, (0, 255, 0), 1)

                # Text
                label = f"{diameter:.2f}mm"
                cv2.putText(display_frame, label, (int(cx) - 20, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                pellet_count += 1

            # Status Bar
            cv2.rectangle(display_frame, (0, 0), (1280, 40), (0, 0, 0), -1)
            cv2.putText(display_frame,
                        f"Scale: {LOCKED_PIXELS_PER_MM:.2f} px/mm  |  Pellets: {pellet_count}  |  'r' to Reset",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow("Stable Inspector", display_frame)

        # KEYBOARD CONTROLS
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and LOCKED_PIXELS_PER_MM is None:
            # Trigger Calibration Routine
            scale = perform_calibration(cap)
            if scale:
                LOCKED_PIXELS_PER_MM = scale
        elif key == ord('r'):
            # Reset
            print("Resetting Calibration...")
            LOCKED_PIXELS_PER_MM = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()