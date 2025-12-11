import cv2
import numpy as np

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
REAL_COIN_DIAMETER_MM = 23.0

# Area settings (You may need to tweak these if lighting is very bright/dim)
MIN_COIN_AREA = 2000
MIN_PELLET_AREA = 100
CIRCULARITY_THRESHOLD = 0.85  # Increased Strictness for coin


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Reduced Blur: Less blur = sharper edges = more accuracy
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Auto-Tuned Canny Edge Detection
    # We use the median brightness to guess the best edge thresholds
    v = np.median(blur)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    canny = cv2.Canny(blur, lower, upper)

    # REMOVED DILATION: We want the exact 1-pixel edge, not a thickened one.

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def calculate_circularity(cnt):
    perimeter = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
    if perimeter == 0: return 0
    return 4 * np.pi * (area / (perimeter * perimeter))


# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Maximize Resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    print(f"Precision Mode Ready. Place NGC 1-Piso ({REAL_COIN_DIAMETER_MM}mm).")

    pixels_per_mm = None

    while True:
        success, frame = cap.read()
        if not success: break

        contours = get_contours(frame)

        possible_coins = []
        pellets = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_PELLET_AREA: continue

            circ = calculate_circularity(cnt)

            if area > MIN_COIN_AREA and circ > CIRCULARITY_THRESHOLD:
                possible_coins.append(cnt)
            else:
                pellets.append(cnt)

        # --- CALIBRATION (Using FitEllipse) ---
        if len(possible_coins) > 0:
            coin_cnt = max(possible_coins, key=cv2.contourArea)

            # FIT ELLIPSE: This is the high-precision fix.
            # It fits a shape to the average of points, ignoring outliers/dust.
            # Returns ((center_x, center_y), (width, height), angle)
            ellipse = cv2.fitEllipse(coin_cnt)
            (cx, cy), (ew, eh), angle = ellipse

            # Average the width and height of the ellipse to get diameter
            # (Handles slight camera tilt better than minEnclosingCircle)
            pixel_diameter = (ew + eh) / 2.0

            pixels_per_mm = pixel_diameter / REAL_COIN_DIAMETER_MM

            # Draw the Ellipse (Green)
            cv2.ellipse(frame, ellipse, (0, 255, 255), 1)
            cv2.putText(frame, "REF", (int(cx) - 20, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Scale: {pixels_per_mm:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        else:
            pixels_per_mm = None
            cv2.putText(frame, "NO COIN DETECTED", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        # --- MEASURE PELLETS ---
        if pixels_per_mm:
            for cnt in pellets:
                # Use minAreaRect for pellets (still the best for rectangles)
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect

                dim_a = w / pixels_per_mm
                dim_b = h / pixels_per_mm

                diameter = min(dim_a, dim_b)
                length = max(dim_a, dim_b)

                box = cv2.boxPoints(rect)
                box = np.int64(box)

                cv2.drawContours(frame, [box], 0, (0, 255, 0), 1)

                label = f"{diameter:.2f} x {length:.2f}"
                cv2.putText(frame, label, (int(cx) - 40, int(cy) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow("Precision Inspector", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()