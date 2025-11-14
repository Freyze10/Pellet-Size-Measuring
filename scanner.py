import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox, QScrollArea
)
from PyQt6.QtGui import QPixmap, QImage, QDoubleValidator
from PyQt6.QtCore import Qt, QSize


# Helper Function (remains the same)
def cv_to_qpixmap(cv_img, target_size=None):
    # ... (same function as before)
    if cv_img is None: return QPixmap()
    if len(cv_img.shape) == 3:
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    else:
        h, w = cv_img.shape
        convert_to_Qt_format = QImage(cv_img.data, w, h, w, QImage.Format.Format_Grayscale8)

    qpixmap = QPixmap.fromImage(convert_to_Qt_format)
    if target_size and not target_size.isNull():
        return qpixmap.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
    return qpixmap


# --- Main Application Window ---

class PelletAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Size Analyzer (Scanner/DPI Mode)")
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

        self.load_btn = QPushButton("Load Scanned Image")
        self.load_btn.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_btn)

        # 1. DPI (Scanner Resolution) Input (Replaced PPM)
        control_layout.addWidget(QLabel("Scanner Resolution (DPI):"))
        self.dpi_input = QLineEdit("600")  # Common high resolution setting
        # Validator for standard DPI values
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
        # ... (Same as before)
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")

        if file_name:
            self.raw_image = cv2.imread(file_name)
            if self.raw_image is None:
                QMessageBox.critical(self, "Error", "Could not load image.")
                return

            self.display_image(self.raw_image)
            self.results_label.setText("Image loaded. Enter DPI (Resolution) and click Analyze.")

    def display_image(self, cv_img):
        # ... (Same as before)
        pixmap = cv_to_qpixmap(cv_img)
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())

    def analyze_image(self):
        """Performs CV processing using DPI to calculate PPM."""
        if self.raw_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        try:
            dpi = float(self.dpi_input.text())
            min_area = float(self.min_area_input.text())

            if dpi <= 0 or min_area <= 0:
                raise ValueError("DPI and Min Area must be positive numbers.")

            # CRITICAL CHANGE: Calculate PPM from DPI
            self.PPM = dpi / 25.4

        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Invalid input value: {e}")
            return

        display_img = self.raw_image.copy()

        # --- CV PROCESSING PIPELINE (Optimized for Scanner High Contrast) ---
        gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Scanners usually provide perfect background, so simple thresholding often works best
        # Adjust THRESH_BINARY_INV depending on whether pellets are dark or light
        # Assuming pellets are darker than the background (common scan result):
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

        # Close small holes
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find external contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pellet_diameters_mm = []

        # Define Drawing Parameters for CLEAR VISUALIZATION
        FONT_SCALE = 1.0
        LINE_THICKNESS = 3
        CIRCLE_COLOR = (0, 255, 255)  # Yellow/Cyan
        TEXT_COLOR = (255, 255, 255)  # White text

        # Loop through contours to measure
        for c in contours:
            area = cv2.contourArea(c)

            # Filter based on user input
            if area < min_area:
                continue

            # 1. Get minimum enclosing circle
            ((x, y), radius_pixels) = cv2.minEnclosingCircle(c)

            diameter_pixels = radius_pixels * 2
            diameter_mm = diameter_pixels / self.PPM

            pellet_diameters_mm.append(diameter_mm)

            # 2. Prepare coordinates for drawing
            center = (int(x), int(y))
            radius = int(radius_pixels)
            text = f"{diameter_mm:.2f} mm"

            # Calculate text position for centering and positioning below the pellet
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, LINE_THICKNESS)
            text_x = int(x) - text_w // 2
            text_y = int(y) + int(radius_pixels) + text_h + 5

            # 3. Draw results on the image
            cv2.circle(display_img, center, radius, CIRCLE_COLOR, LINE_THICKNESS)
            cv2.putText(display_img, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, LINE_THICKNESS)

        # --- 4. Display Results ---

        if not pellet_diameters_mm:
            self.results_label.setText(
                f"No pellets found that meet the minimum area requirement ({min_area} pixels^2). "
                f"Check threshold settings or adjust min area."
            )
            self.display_image(self.raw_image)
            return

        total_pellets = len(pellet_diameters_mm)
        avg_diameter = np.mean(pellet_diameters_mm)
        std_dev = np.std(pellet_diameters_mm)

        results_text = (
            f"Analysis Complete (Scanner Mode):\n"
            f"Total Pellets Found: {total_pellets}\n"
            f"Average Diameter: {avg_diameter:.3f} mm\n"
            f"Standard Deviation: {std_dev:.3f} mm\n"
            f"Scanner DPI: {dpi:.0f}\n"
            f"PPM Calculated: {self.PPM:.2f}\n"
            f"Min Area Filter: {min_area:.0f} pixels^2"
        )
        self.results_label.setText(results_text)

        self.display_image(display_img)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PelletAnalyzer()
    window.show()
    sys.exit(app.exec())