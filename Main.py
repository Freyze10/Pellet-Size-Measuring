"""
Real-Time Pellet Size Measurement Application
Measures cylindrical pellets using webcam feed and displays tolerance status.
"""

import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QSpinBox, QGroupBox)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap


class PelletMeasurementApp(QMainWindow):
    """Main application window for pellet measurement system."""

    # Tolerance range in millimeters
    MIN_SIZE = 2.5
    MAX_SIZE = 3.5
    TARGET_SIZE = 3.0

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Pellet Size Measurement")
        self.setGeometry(100, 100, 1000, 700)

        # Initialize webcam
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Calibration: pixels per millimeter (default value, adjustable)
        self.px_per_mm = 10.0  # Adjust based on camera distance

        # UI Setup
        self.setup_ui()

        # Timer for frame capture
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS

    def setup_ui(self):
        """Initialize the user interface components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Video display label
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #333; background-color: black;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.video_label)

        # Calibration controls
        calib_group = QGroupBox("Calibration")
        calib_layout = QHBoxLayout()

        calib_layout.addWidget(QLabel("Pixels per mm:"))
        self.calib_spinbox = QSpinBox()
        self.calib_spinbox.setRange(1, 100)
        self.calib_spinbox.setValue(int(self.px_per_mm))
        self.calib_spinbox.valueChanged.connect(self.update_calibration)
        calib_layout.addWidget(self.calib_spinbox)

        calib_info = QLabel("💡 Adjust based on a known reference (e.g., 10mm = X pixels)")
        calib_info.setStyleSheet("color: #666; font-size: 10px;")
        calib_layout.addWidget(calib_info)
        calib_layout.addStretch()

        calib_group.setLayout(calib_layout)
        main_layout.addWidget(calib_group)

        # Status label
        self.status_label = QLabel("⏳ Initializing...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
                background-color: #f0f0f0;
            }
        """)
        main_layout.addWidget(self.status_label)

    def update_calibration(self, value):
        """Update the pixel-to-millimeter calibration value."""
        self.px_per_mm = float(value)

    def update_frame(self):
        """Capture and process a frame from the webcam."""
        ret, frame = self.capture.read()
        if not ret:
            return

        # Process frame to detect pellets
        processed_frame, all_within_tolerance = self.process_frame(frame)

        # Update status label
        self.update_status(all_within_tolerance)

        # Convert frame to QPixmap and display
        self.display_frame(processed_frame)

    def process_frame(self, frame):
        """
        Detect pellets and measure their dimensions.

        Args:
            frame: Input BGR frame from webcam

        Returns:
            tuple: (processed_frame, all_within_tolerance)
        """
        output_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding for better detection in varying lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        pellets_detected = []
        all_within_tolerance = True

        for contour in contours:
            # Filter by area to avoid noise
            area = cv2.contourArea(contour)
            if area < 100:  # Minimum area threshold
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate dimensions in millimeters
            width_mm = w / self.px_per_mm
            height_mm = h / self.px_per_mm

            # Determine which dimension is diameter and which is length
            # Assume the smaller dimension is diameter
            diameter_mm = min(width_mm, height_mm)
            length_mm = max(width_mm, height_mm)

            # Check tolerance
            diameter_ok = self.MIN_SIZE <= diameter_mm <= self.MAX_SIZE
            length_ok = self.MIN_SIZE <= length_mm <= self.MAX_SIZE
            within_tolerance = diameter_ok and length_ok

            if not within_tolerance:
                all_within_tolerance = False

            # Draw bounding box
            color = (0, 255, 0) if within_tolerance else (0, 0, 255)  # Green or Red
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)

            # Add measurement text
            label_bg_color = (0, 200, 0) if within_tolerance else (0, 0, 200)

            # Diameter label
            diameter_text = f"D: {diameter_mm:.2f}mm"
            cv2.putText(
                output_frame, diameter_text, (x, y - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
            cv2.putText(
                output_frame, diameter_text, (x, y - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_bg_color, 1
            )

            # Length label
            length_text = f"L: {length_mm:.2f}mm"
            cv2.putText(
                output_frame, length_text, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
            cv2.putText(
                output_frame, length_text, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_bg_color, 1
            )

            # Status indicator
            status_text = "✓ OK" if within_tolerance else "✗ OUT"
            cv2.putText(
                output_frame, status_text, (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

            pellets_detected.append({
                'diameter': diameter_mm,
                'length': length_mm,
                'within_tolerance': within_tolerance
            })

        # Add info overlay
        info_text = f"Pellets detected: {len(pellets_detected)} | Calibration: {self.px_per_mm:.1f} px/mm"
        cv2.putText(
            output_frame, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        cv2.putText(
            output_frame, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )

        # If no pellets detected, set status to True (no violations)
        if len(pellets_detected) == 0:
            all_within_tolerance = True

        return output_frame, all_within_tolerance

    def update_status(self, all_within_tolerance):
        """Update the status label based on tolerance check."""
        if all_within_tolerance:
            self.status_label.setText("✅ All pellets within tolerance")
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    font-weight: bold;
                    padding: 15px;
                    border-radius: 5px;
                    background-color: #d4edda;
                    color: #155724;
                }
            """)
        else:
            self.status_label.setText("❌ Out of tolerance detected")
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    font-weight: bold;
                    padding: 15px;
                    border-radius: 5px;
                    background-color: #f8d7da;
                    color: #721c24;
                }
            """)

    def display_frame(self, frame):
        """Convert OpenCV frame to QPixmap and display in label."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        qt_image = QImage(
            rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.video_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        """Clean up resources when closing the application."""
        self.timer.stop()
        self.capture.release()
        event.accept()


def main():
    """Application entry point."""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    window = PelletMeasurementApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()