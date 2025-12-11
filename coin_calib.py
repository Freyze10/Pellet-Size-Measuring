import cv2
import numpy as np

# -----------------------------------------------------------------------------
# CONSTANTS - PHILIPPINE PESO CONFIGURATION
# -----------------------------------------------------------------------------
# The New Generation Currency (NGC) 1-Piso coin is exactly 23.0 mm
REAL_COIN_DIAMETER_MM = 23.0

# Thresholds to distinguish Coin from Pellet
MIN_COIN_AREA = 2000  # Pixels (ignore small noise)
MIN_PELLET_AREA = 100  # Pixels (ignore dust)
CIRCULARITY_THRESHOLD = 0.8  # Above 0.8 is a circle (coin), below is a pellet


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_contours(img):
    """
    Converts image to binary and finds contours.
    """
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Blur (removes grain/noise)
    blur = cv2.GaussianBlur(gray, (7, 7), 1)

    # 3. Canny Edge Detection (Great for clean shapes)
    canny = cv2.Canny(blur, 50, 150)

    # 4. Dilate (Make the edges thicker to ensure they close up)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=1)

    # 5. Find Contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def calculate_circularity(cnt):
    """
    Math to determine if a shape is a Circle or a Rectangle.
    Returns: Close to 1.0 for Circle, Lower for Rectangle/Pellet.
    """
    perimeter = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
    if perimeter == 0: return 0
    return 4 * np.pi * (area / (perimeter * perimeter))


# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Set Resolution (Higher is better for measurement)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus if possible

    print(f"System Ready. Place a 1-Piso coin ({REAL_COIN_DIAMETER_MM}mm) in view.")

    pixels_per_mm = None  # Will be calculated automatically

    while True:
        success, frame = cap.read()
        if not success: break

        contours = get_contours(frame)

        # Reset lists
        possible_coins = []
        pellets = []

        # --- PASS 1: CLASSIFY OBJECTS ---
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_PELLET_AREA: continue  # Ignore noise/dust

            # Calculate Circularity
            circ = calculate_circularity(cnt)

            if area > MIN_COIN_AREA and circ > CIRCULARITY_THRESHOLD:
                # It's likely the coin (Big and Round)
                possible_coins.append(cnt)
            else:
                # It's likely a pellet (Smaller or Rectangular)
                pellets.append(cnt)

        # --- PASS 2: CALIBRATE FROM COIN ---
        # We assume the largest "Round" object is the Peso
        if len(possible_coins) > 0:
            # Find the largest circle contour
            coin_cnt = max(possible_coins, key=cv2.contourArea)

            # Fit a perfect circle around it
            ((cx, cy), radius) = cv2.minEnclosingCircle(coin_cnt)

            # Draw the Coin detected
            cv2.circle(frame, (int(cx), int(cy)), int(radius), (0, 255, 255), 2)
            cv2.putText(frame, "1 PISO (REF)", (int(cx) - 40, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Calculate Scale
            pixel_diameter = radius * 2
            pixels_per_mm = pixel_diameter / REAL_COIN_DIAMETER_MM

            # Display Scale info
            cv2.putText(frame, f"Scale: {pixels_per_mm:.2f} px/mm", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            pixels_per_mm = None
            cv2.putText(frame, "WAITING FOR COIN...", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- PASS 3: MEASURE PELLETS ---
        if pixels_per_mm:
            for cnt in pellets:
                # Fit a rotated rectangle (handles angled pellets)
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect

                # Convert pixels to mm
                dim_a = w / pixels_per_mm
                dim_b = h / pixels_per_mm

                # Determine which is Diameter (width) and Length
                # Usually Length > Diameter for pellets
                diameter = min(dim_a, dim_b)
                length = max(dim_a, dim_b)

                # --- VISUALIZATION ---
                box = cv2.boxPoints(rect)

                # --- FIX: Use int64 instead of int0 ---
                box = np.int64(box)

                # Draw Box (Green)
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

                # Draw Measurements
                label = f"{diameter:.2f} x {length:.2f} mm"

                # Smart label placement (slightly above the object)
                cv2.putText(frame, label, (int(cx) - 40, int(cy) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Inspector - Coin Mode", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()