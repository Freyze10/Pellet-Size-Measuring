import cv2
import numpy as np
import time

# -----------------------------------------------------------------------------
# CONSTANTS - PHILIPPINES SETTINGS
# -----------------------------------------------------------------------------
REAL_COIN_DIAMETER_MM = 23.0  # NGC 1-Piso (New Series)
MIN_PELLET_AREA = 150  # Ignore small dust
MAX_PELLET_AREA = 8000  # Ignore the coin after calibration

# Global Variable to store the "Locked" scale
LOCKED_SCALE = None


def get_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Auto-adjust thresholds based on lighting
    v = np.median(blur)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))

    edges = cv2.Canny(blur, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def calibrate_system(cap):
    """
    Locks the scale by averaging 20 frames of the coin.
    """
    print("--- CALIBRATING... ---")
    measurements = []

    for i in range(20):
        ret, frame = cap.read()
        if not ret: continue

        contours = get_contours(frame)
        if contours:
            # Assume largest object is the coin
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 2000:
                ellipse = cv2.fitEllipse(largest)
                (cx, cy), (w, h), angle = ellipse
                avg_diam = (w + h) / 2.0
                measurements.append(avg_diam)
        time.sleep(0.05)

    if len(measurements) < 5:
        print("Calibration Failed. Coin not clear.")
        return None

    avg_px_diameter = np.mean(measurements)
    final_scale = avg_px_diameter / REAL_COIN_DIAMETER_MM
    print(f"--- LOCKED SCALE: {final_scale:.2f} px/mm ---")
    return final_scale


def main():
    global LOCKED_SCALE
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        display = frame.copy()

        # ----------------------------------------------------
        # STATE 1: NOT CALIBRATED
        # ----------------------------------------------------
        if LOCKED_SCALE is None:
            cv2.putText(display, "STEP 1: Place 1-Piso Coin", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display, "STEP 2: Press 'c' to Calibrate", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show yellow outline of what computer sees
            contours = get_contours(frame)
            for cnt in contours:
                if cv2.contourArea(cnt) > 2000:
                    cv2.drawContours(display, [cnt], -1, (0, 255, 255), 2)

        # ----------------------------------------------------
        # STATE 2: CALIBRATED & MEASURING
        # ----------------------------------------------------
        else:
            contours = get_contours(frame)
            count = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)

                # Filter: Must be pellet size
                if area < MIN_PELLET_AREA or area > MAX_PELLET_AREA:
                    continue

                # 1. Get Geometry
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect

                # 2. Calculate Dimensions
                dim1 = w / LOCKED_SCALE
                dim2 = h / LOCKED_SCALE

                # Logic: The longer side is Length, shorter is Height/Dia
                length_mm = max(dim1, dim2)
                height_mm = min(dim1, dim2)

                # 3. Draw Box
                box = cv2.boxPoints(rect)
                box = np.int64(box)
                cv2.drawContours(display, [box], 0, (0, 255, 0), 1)

                # 4. Draw Text (Separated Lines)
                # Position text slightly to the right of the pellet
                text_x = int(cx) + 20
                text_y = int(cy)

                # LINE 1: Length (Red)
                cv2.putText(display, f"L: {length_mm:.2f}", (text_x, text_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Bold Red

                # LINE 2: Height (Blue/Cyan)
                cv2.putText(display, f"H: {height_mm:.2f}", (text_x, text_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)  # Cyan

                count += 1

            # Top Info Bar
            cv2.rectangle(display, (0, 0), (1280, 40), (20, 20, 20), -1)
            msg = f"Scale Locked: {LOCKED_SCALE:.2f} px/mm  |  Pellets: {count}  |  'r' Reset"
            cv2.putText(display, msg, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow("Pellet Inspector", display)

        # Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and LOCKED_SCALE is None:
            scale = calibrate_system(cap)
            if scale: LOCKED_SCALE = scale
        elif key == ord('r'):
            LOCKED_SCALE = None
            print("Resetting...")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()