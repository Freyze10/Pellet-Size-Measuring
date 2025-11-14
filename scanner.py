import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QWidget, QFileDialog, QSlider, QGridLayout,
    QGroupBox, QSpinBox, QDoubleSpinBox, QTextEdit, QFrame
)
from PyQt6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent
from PyQt6.QtCore import Qt, QMimeData
from typing import List, Dict


# ----------------------------------------------------------------------
# Global Configuration
# ----------------------------------------------------------------------
TARGET_DIAMETER = 3.0
TARGET_LENGTH   = 3.0
TOLERANCE       = 0.5
EXCLUSION_THRESHOLD = 200.0

MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 10000


class PelletInspector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Size Inspector")
        self.setGeometry(100, 100, 1400, 800)

        self.image = None
        self.pixmap = None
        self.pixels_per_mm = 6.0
        self.pellets: List[Dict] = []

        self.init_ui()
        self.update_calibration_ranges()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --------------------- Left Panel ---------------------
        left_panel = QGroupBox("Controls & Results")
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)

        # Upload Button
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        left_layout.addWidget(self.upload_btn)

        # Drag & Drop Label
        self.drop_label = QLabel("Or drag and drop an image here")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setStyleSheet("QLabel { border: 2px dashed #aaa; padding: 20px; }")
        left_layout.addWidget(self.drop_label)

        # Calibration Group
        calib_group = QGroupBox("Calibration")
        calib_layout = QGridLayout(calib_group)

        calib_layout.addWidget(QLabel("Pixels per mm:"), 0, 0)
        self.px_spin = QDoubleSpinBox()
        self.px_spin.setRange(0.1, 50.0)
        self.px_spin.setSingleStep(0.1)
        self.px_spin.setValue(self.pixels_per_mm)
        self.px_spin.valueChanged.connect(self.on_calibration_change)
        calib_layout.addWidget(self.px_spin, 0, 1)

        self.px_slider = QSlider(Qt.Orientation.Horizontal)
        self.px_slider.setRange(1, 500)
        self.px_slider.setValue(int(self.pixels_per_mm * 10))
        self.px_slider.valueChanged.connect(self.on_slider_change)
        calib_layout.addWidget(self.px_slider, 1, 0, 1, 2)

        left_layout.addWidget(calib_group)

        # Statistics
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        self.stats_display.setFixedHeight(150)
        self.stats_display.setStyleSheet("font-family: Consolas; font-size: 11px;")
        left_layout.addWidget(QGroupBox("Pellet Statistics", self.stats_display))

        # --------------------- Right Panel ---------------------
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f0f0f0;")
        self.image_label.setMinimumSize(600, 400)

        # Enable drag and drop
        self.setAcceptDrops(True)
        self.drop_label.setAcceptDrops(True)
        self.drop_label.dragEnterEvent = self.dragEnterEvent
        self.drop_label.dropEvent = self.dropEvent

        # Add to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.image_label, 1)

    # ------------------------------------------------------------------
    # Drag & Drop
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.load_image(files[0])

    # ------------------------------------------------------------------
    # Image Loading
    # ------------------------------------------------------------------
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path: str):
        self.image = cv2.imread(file_path)
        if self.image is None:
            self.stats_display.setText("Error: Could not load image.")
            return

        self.process_and_display()
        self.update_stats()

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def update_calibration_ranges(self):
        self.diameter_min = TARGET_DIAMETER - TOLERANCE
        self.diameter_max = TARGET_DIAMETER + TOLERANCE
        self.length_min = TARGET_LENGTH - TOLERANCE
        self.length_max = TARGET_LENGTH + TOLERANCE

        self.exclude_d_min = TARGET_DIAMETER - EXCLUSION_THRESHOLD
        self.exclude_d_max = TARGET_DIAMETER + EXCLUSION_THRESHOLD
        self.exclude_l_min = TARGET_LENGTH - EXCLUSION_THRESHOLD
        self.exclude_l_max = TARGET_LENGTH + EXCLUSION_THRESHOLD

    def on_calibration_change(self, value):
        self.pixels_per_mm = value
        self.px_slider.blockSignals(True)
        self.px_slider.setValue(int(value * 10))
        self.px_slider.blockSignals(False)
        self.process_and_display()
        self.update_stats()

    def on_slider_change(self, value):
        self.pixels_per_mm = value / 10.0
        self.px_spin.blockSignals(True)
        self.px_spin.setValue(self.pixels_per_mm)
        self.px_spin.blockSignals(False)
        self.process_and_display()
        self.update_stats()

    # ------------------------------------------------------------------
    # Pellet Detection
    # ------------------------------------------------------------------
    def detect_pellets(self, img) -> List[Dict]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pellets = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            diameter_px = min(w, h)
            length_px = max(w, h)

            diameter_mm = diameter_px / self.pixels_per_mm
            length_mm = length_px / self.pixels_per_mm

            # Exclusion filter
            if not (self.exclude_d_min <= diameter_mm <= self.exclude_d_max and
                    self.exclude_l_min <= length_mm <= self.exclude_l_max):
                continue

            within_tol = (
                self.diameter_min <= diameter_mm <= self.diameter_max and
                self.length_min <= length_mm <= self.length_max
            )

            pellets.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'diameter': round(diameter_mm, 3),
                'length': round(length_mm, 3),
                'within_tolerance': within_tol
            })

        return pellets

    # ------------------------------------------------------------------
    # Display & Drawing
    # ------------------------------------------------------------------
    def process_and_display(self):
        if self.image is None:
            return

        display_img = self.image.copy()
        self.pellets = self.detect_pellets(display_img)

        # Draw on image
        for p in self.pellets:
            x, y, w, h = p['x'], p['y'], p['w'], p['h']
            color = (0, 255, 0) if p['within_tolerance'] else (0, 0, 255)
            cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 2)

            # Text background
            bg_y = max(y - 50, 0)
            cv2.rectangle(display_img, (x, bg_y), (x + 130, y - 5), (0, 0, 0), -1)
            cv2.putText(display_img, f"D: {p['diameter']}", (x + 5, bg_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_img, f"L: {p['length']}", (x + 5, bg_y + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if not p['within_tolerance']:
                cv2.circle(display_img, (x + w - 12, y + 12), 10, (0, 0, 255), -1)
                cv2.putText(display_img, "!", (x + w - 18, y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Convert to QPixmap
        height, width, channel = display_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(display_img.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        self.pixmap = QPixmap.fromImage(q_img)

        # Scale to fit label while keeping aspect ratio
        scaled_pixmap = self.pixmap.scaled(
            self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap:
            scaled = self.pixmap.scaled(
                self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def update_stats(self):
        if not self.pellets:
            self.stats_display.setHtml("<p style='color: gray;'>No pellets detected.</p>")
            return

        total = len(self.pellets)
        within = sum(1 for p in self.pellets if p['within_tolerance'])
        out = total - within

        html = f"""
        <b style="color: #333;">Pellet Analysis</b><br>
        <b>Total Detected:</b> {total}<br>
        <b style="color: green;">Within Tolerance:</b> {within}<br>
        <b style="color: red;">Out of Tolerance:</b> {out}<br>
        <hr>
        <small><b>Target:</b> D={TARGET_DIAMETER}±{TOLERANCE}mm, L={TARGET_LENGTH}±{TOLERANCE}mm<br>
        <b>Calibration:</b> {self.pixels_per_mm:.2f} px/mm</small>
        <hr>
        """
        for i, p in enumerate(self.pellets):
            status = "OK" if p['within_tolerance'] else "BAD"
            color = "green" if p['within_tolerance'] else "red"
            html += f"<span style='color: {color};'>#{i+1}: D={p['diameter']} L={p['length']} [{status}]</span><br>"

        self.stats_display.setHtml(html)


# ----------------------------------------------------------------------
# Run App
# ----------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Clean modern look

    window = PelletInspector()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()