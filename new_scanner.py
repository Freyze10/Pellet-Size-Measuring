import sys
import cv2
import numpy as np
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox, QScrollArea,
    QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImage, QDoubleValidator
from PyQt6.QtCore import Qt, QSize


# --- ROBUST IMAGE CONVERSION HELPER FUNCTION (CRASH FIX) ---

def cv_to_qpixmap(cv_img, target_size=None):
    """
    Converts an OpenCV BGR/Grayscale image (numpy array) to a PyQt6 QPixmap.
    This version explicitly converts to RGB and makes a C-contiguous copy for QImage stability.
    """
    if cv_img is None:
        return QPixmap()

    # 1. Handle Color/Channel Conversion and prepare for QImage
    if len(cv_img.shape) == 3:
        # Convert BGR to RGB (Qt expects RGB for Format_RGB888)
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        format = QImage.Format.Format_RGB888
        data_to_pass = rgb_image
    elif len(cv_img.shape) == 2:
        # Grayscale
        h, w = cv_img.shape
        bytes_per_line = w
        format = QImage.Format.Format_Grayscale8
        data_to_pass = cv_img
    else:
        return QPixmap()  # Unsupported format

    # CRITICAL FIX: Explicitly create a C-contiguous array copy.
    contiguous_data = data_to_pass.copy(order='C')

    convert_to_Qt_format = QImage(
        contiguous_data.data,
        w,
        h,
        bytes_per_line,
        format
    )

    qpixmap = QPixmap.fromImage(convert_to_Qt_format)

    if target_size and not target_size.isNull():
        return qpixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
    return qpixmap


# --- Helper Function for Safe File Load ---
def cv2_safe_imread(path):
    """
    Reads an image using cv2.imdecode to handle file path/encoding issues.
    """
    try:
        with open(path, 'rb') as f:
            data = f.read()
        np_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error during safe image read: {e}")
        return None


# --- Main Application Window ---

class PelletAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Size Analyzer (Minimum Area Bounding Box Mode)")
        self.setGeometry(100, 100, 1200, 800)

        self.raw_image = None
        self.PPM = 0.0  # Pixels Per Millimeter

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel: Controls and Results ---
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(300)
        control_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self.load_btn = QPushButton("Load Scanned Image")
        self.load_btn.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_btn)

        # 1. DPI (Scanner Resolution) Input
        control_layout.addWidget(QLabel("Scanner Resolution (DPI):"))
        self.dpi_input = QLineEdit("600")
        self.dpi_input.setValidator(QDoubleValidator(1.0, 4800.0, 0))
        control_layout.addWidget(self.dpi_input)

        control_layout.addSpacing(10)

        # 2. Analysis Parameters (Min Area)
        control_layout.addWidget(QLabel("Min Pellet Area (pixels^2, e.g., 1000):"))
        self.min_area_input = QLineEdit("1000")
        self.min_area_input.setValidator(QDoubleValidator(1.0, 100000.0, 0))
        control_layout.addWidget(self.min_area_input)

        control_layout.addSpacing(20)

        # 3. Analysis Button
        self.analyze_btn = QPushButton("Analyze Pellets")
        self.analyze_btn.clicked.connect(self.analyze_image)
        control_layout.addWidget(self.analyze_btn)

        control_layout.addSpacing(20)

        # 4. Results Display
        control_layout.addWidget(QLabel("--- Analysis Results ---"))
        self.results_label = QLabel("Load image and enter DPI to begin.")
        self.results_label.setWordWrap(True)
        control_layout.addWidget(self.results_label)

        control_layout.addStretch(1)
        main_layout.addWidget(control_panel)

        # --- Right Panel: Image Display ---
        self.image_label = QLabel("Image Display Area")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.image_label)

        main_layout.addWidget(scroll_area, 3)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")

        if file_name:
            # Use the safe loading function
            self.raw_image = cv2_safe_imread(file_name)

            if self.raw_image is None:
                QMessageBox.critical(self, "Error", f"Could not load image file: {os.path.basename(file_name)}.")
                return

            self.display_image(self.raw_image)
            self.results_label.setText("Image loaded. Enter DPI (Resolution) and click Analyze.")

    def display_image(self, cv_img):
        pixmap = cv_to_qpixmap(cv_img, target_size=None)
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())

    def analyze_image(self):
        """Performs CV processing using DPI to calculate PPM and MinAreaRect for accurate size."""
        if self.raw_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        try:
            dpi = float(self.dpi_input.text())
            min_area_px = float(self.min_area_input.text())

            if dpi <= 0 or min_area_px <= 0:
                raise ValueError("DPI and Min Area must be positive numbers.")

            self.PPM = dpi / 25.4  # Pixels Per Millimeter

        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Invalid input value: {e}")
            return

        # Use a copy of the raw image for drawing
        display_img = self.raw_image.copy()

        # --- CV Processing: Adaptive Thresholding for Robustness ---
        gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding is better for scans with uneven lighting
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Morphological Close to connect broken pieces and fill holes
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find external contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pellet_dimensions_mm = []  # Stores (width_mm, height_mm)

        # Define Drawing Parameters
        FONT_SCALE = 1.0
        LINE_THICKNESS = 3
        BOX_COLOR = (255, 0, 0)  # Blue for bounding box
        TEXT_COLOR = (255, 255, 255)  # White text

        # Loop through contours to measure
        for c in contours:
            area = cv2.contourArea(c)

            # Filter based on user input
            if area < min_area_px:
                continue

            # 1. Get Minimum Area Bounding Box (Rotated Rectangle)
            rect = cv2.minAreaRect(c)
            (center_x, center_y), (w_px, h_px), angle = rect

            # Sort the dimensions: smaller is Width/Diameter, larger is Length/Height
            width_px = min(w_px, h_px)
            height_px = max(w_px, h_px)

            # Convert to mm
            width_mm = width_px / self.PPM
            height_mm = height_px / self.PPM

            pellet_dimensions_mm.append((width_mm, height_mm))

            # 2. Draw the Rotated Bounding Box
            box_points = cv2.boxPoints(rect)
            box_points = np.intp(box_points)  # Convert to integer coordinates
            cv2.drawContours(display_img, [box_points], 0, BOX_COLOR, LINE_THICKNESS)

            # 3. Draw dimensions text near the center
            text = f"{width_mm:.2f} x {height_mm:.2f} mm"

            # Calculate text position (using the box center for reference)
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, LINE_THICKNESS)
            text_x = int(center_x) - text_w // 2
            text_y = int(center_y) + text_h // 2

            # Draw a black rectangle under the text for better visibility
            cv2.rectangle(display_img, (text_x - 5, text_y - text_h),
                          (text_x + text_w + 5, text_y + baseline + 5),
                          (0, 0, 0), -1)

            cv2.putText(display_img, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, 2)

        # --- 4. Display Results ---

        if not pellet_dimensions_mm:
            self.results_label.setText(
                f"No pellets found that meet the minimum area requirement ({min_area_px} pixels^2). "
                f"Check threshold settings or adjust min area."
            )
            self.display_image(self.raw_image)
            return

        total_pellets = len(pellet_dimensions_mm)

        # Calculate overall average dimensions
        avg_w = np.mean([d[0] for d in pellet_dimensions_mm])
        avg_h = np.mean([d[1] for d in pellet_dimensions_mm])

        results_text = (
            f"Analysis Complete (Rotated Bounding Box):\n"
            f"Total Pellets Found: {total_pellets}\n"
            f"Average Width (Diameter): {avg_w:.3f} mm\n"
            f"Average Length (Height): {avg_h:.3f} mm\n"
            f"Scanner DPI: {dpi:.0f}\n"
            f"PPM Calculated: {self.PPM:.2f}\n"
            f"Min Area Filter: {min_area_px:.0f} pixels^2"
        )
        self.results_label.setText(results_text)

        self.display_image(display_img)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PelletAnalyzer()
    window.show()
    sys.exit(app.exec())