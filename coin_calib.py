import cv2
import numpy as np

# -----------------------------------------------------------------------------
# CONSTANTS - PHILIPPINE PESO CONFIGURATION
# -----------------------------------------------------------------------------
REAL_COIN_DIAMETER_MM = 23.0  # NGC 1-Piso (New Series)

# SETTINGS
MIN_COIN_AREA = 2000
MIN_PELLET_AREA = 100
CIRCULARITY_THRESHOLD = 0.82


# -----------------------------------------------------------------------------
# CORE PROCESSING
# -----------------------------------------------------------------------------
def get_contours(img):
    """
    Standard pre-processing pipeline.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Slight blur to reduce noise, but kept low to keep edges sharp
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Auto-Canny (Adjusts to lighting brightness automatically)
    v = np.median(blur)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    canny = cv2.Canny(blur, lower, upper)

    # Close gaps in edges
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def calculate_circularity(cnt):
    perimeter = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
    if perimeter == 0: return 0
    return 4 * np.pi * (area / (perimeter * perimeter))


def process_frame(frame):
    """
    This function analyzes the frozen frame and draws measurements.
    """
    processed_img = frame.copy()
    contours = get_contours(processed_img)

    possible_coins = []
    pellets = []

    # 1. Sort objects
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_PELLET_AREA: continue

        circ = calculate_circularity(cnt)
        if area > MIN_COIN_AREA and circ > CIRCULARITY_THRESHOLD:
            possible_coins.append(cnt)
        else:
            pellets.append(cnt)

    # 2. Find Coin & Calculate Scale
    pixels_per_mm = None

    if len(possible_coins) > 0:
        # Pick largest circular object
        coin_cnt = max(possible_coins, key=cv2.contourArea)

        # Use fitEllipse for best edge fitting
        ellipse = cv2.fitEllipse(coin_cnt)
        (cx, cy), (ew, eh), angle = ellipse

        # Average width/height to account for slight tilts
        avg_pixel_diameter = (ew + eh) / 2.0
        pixels_per_mm = avg_pixel_diameter / REAL_COIN_DIAMETER_MM

        # Draw Coin
        cv2.ellipse(processed_img, ellipse, (0, 255, 255), 1)
        cv2.putText(processed_img, f"REF: {pixels_per_mm:.2f} px/mm",
                    (int(cx) - 60, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(processed_img, "ERROR: Coin not found!", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 3. Measure Pellets
    if pixels_per_mm:
        count = 0
        for cnt in pellets:
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect

            # Convert to mm
            dim1 = w / pixels_per_mm
            dim2 = h / pixels_per_mm

            diameter = min(dim1, dim2)
            length = max(dim1, dim2)

            # Draw
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            cv2.drawContours(processed_img, [box], 0, (0, 255, 0), 1)

            # Label
            label = f"{diameter:.2f} x {length:.2f}"
            cv2.putText(processed_img, label, (int(cx) - 40, int(cy) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            count += 1

        cv2.putText(processed_img, f"Measured {count} pellets", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return processed_img


# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    is_frozen = False
    frozen_frame_display = None

    print("System Started. Please CLICK on the video window to focus it.")

    while True:
        # A. IF FROZEN, SHOW RESULT
        if is_frozen:
            cv2.imshow("Inspector", frozen_frame_display)

            # Key Handling for Frozen State
            key = cv2.waitKey(1) & 0xFF  # <--- FIX ADDED HERE

            if key == ord('q'):
                break
            elif key == 32 or key == ord(' '):  # Spacebar
                is_frozen = False
                frozen_frame_display = None
                print("Returning to Live Preview...")
            continue

        # B. LIVE PREVIEW
        ret, frame = cap.read()
        if not ret: break

        # Draw crosshair
        h, w = frame.shape[:2]
        cv2.line(frame, (w // 2, 0), (w // 2, h), (50, 50, 50), 1)
        cv2.line(frame, (0, h // 2), (w, h // 2), (50, 50, 50), 1)

        cv2.putText(frame, "PREVIEW - Click Window & Press SPACE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Inspector", frame)

        # C. KEY HANDLING
        # We use & 0xFF to strip extra data ensuring standard ASCII codes
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == 32 or key == ord(' '):  # 32 is the ASCII code for Spacebar
            print("Spacebar detected! Capturing...")
            frozen_frame_display = process_frame(frame.copy())
            is_frozen = True

        # Optional: Debug print to see what key code your computer sends
        # if key != 255: print(f"Key Pressed: {key}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()