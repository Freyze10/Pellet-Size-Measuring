import sys
import json
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSpinBox, QDoubleSpinBox, QGroupBox, QScrollArea)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage


class PelletMeasurementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Size Measurement System")
        self.setGeometry(100, 100, 1400, 800)

        # Configuration
        self.pixels_per_mm = 6.0
        self.target_diameter = 3.0
        self.target_length = 3.0
        self.tolerance = 0.5

        # Data storage
        self.current_image = None
        self.current_image_path = None
        self.coco_data = None
        self.detected_pellets = []
        self.annotations_for_image = []

        # Load COCO annotations
        self.load_coco_annotations()

        # Setup UI
        self.init_ui()

    def load_coco_annotations(self):
        """Load COCO format annotations from pellets_label.json"""
        json_path = "pellets_label.json"
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found.")
            return

        try:
            with open(json_path, 'r') as f:
                self.coco_data = json.load(f)
            print(f"✓ Loaded COCO annotations with {len(self.coco_data.get('annotations', []))} annotations")
        except Exception as e:
            print(f"Error loading COCO annotations: {e}")

    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left panel - Controls and Stats
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)

        # Right panel - Image display
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 3)

    def create_left_panel(self):
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Load Image Button
        self.load_btn = QPushButton("Load Pellet Image")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setStyleSheet("QPushButton { padding: 10px; font-size: 14px; }")
        layout.addWidget(self.load_btn)

        # Calibration Group
        calib_group = QGroupBox("Calibration")
        calib_layout = QVBoxLayout()

        # Pixels per mm
        px_layout = QHBoxLayout()
        px_label = QLabel("Pixels per mm:")
        self.px_spinbox = QDoubleSpinBox()
        self.px_spinbox.setRange(0.1, 100.0)
        self.px_spinbox.setValue(self.pixels_per_mm)
        self.px_spinbox.setSingleStep(0.1)
        self.px_spinbox.setDecimals(2)
        self.px_spinbox.valueChanged.connect(self.update_calibration)
        px_layout.addWidget(px_label)
        px_layout.addWidget(self.px_spinbox)
        calib_layout.addLayout(px_layout)

        calib_group.setLayout(calib_layout)
        layout.addWidget(calib_group)

        # Target Specifications Group
        spec_group = QGroupBox("Target Specifications")
        spec_layout = QVBoxLayout()

        spec_layout.addWidget(QLabel(f"Target Diameter: {self.target_diameter} mm"))
        spec_layout.addWidget(QLabel(f"Target Length: {self.target_length} mm"))
        spec_layout.addWidget(QLabel(f"Tolerance: ±{self.tolerance} mm"))
        spec_layout.addWidget(QLabel(
            f"Acceptable Range: {self.target_diameter - self.tolerance:.1f} - {self.target_diameter + self.tolerance:.1f} mm"))

        spec_group.setLayout(spec_layout)
        layout.addWidget(spec_group)

        # Statistics Group
        self.stats_group = QGroupBox("Detection Statistics")
        self.stats_layout = QVBoxLayout()

        self.total_label = QLabel("Total Pellets: 0")
        self.within_label = QLabel("Within Tolerance: 0")
        self.out_label = QLabel("Out of Tolerance: 0")
        self.status_label = QLabel("Status: No image loaded")
        self.status_label.setStyleSheet("font-weight: bold; padding: 5px;")

        self.stats_layout.addWidget(self.total_label)
        self.stats_layout.addWidget(self.within_label)
        self.stats_layout.addWidget(self.out_label)
        self.stats_layout.addWidget(self.status_label)

        self.stats_group.setLayout(self.stats_layout)
        layout.addWidget(self.stats_group)

        # Pellet Details Group (scrollable)
        details_group = QGroupBox("Pellet Details")
        details_layout = QVBoxLayout()

        self.details_scroll = QScrollArea()
        self.details_scroll.setWidgetResizable(True)
        self.details_widget = QWidget()
        self.details_widget_layout = QVBoxLayout()
        self.details_widget.setLayout(self.details_widget_layout)
        self.details_scroll.setWidget(self.details_widget)

        details_layout.addWidget(self.details_scroll)
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        layout.addStretch()

        return panel

    def create_right_panel(self):
        """Create right image display panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Image label
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.image_label.setMinimumSize(800, 600)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)

        layout.addWidget(scroll_area)

        return panel

    def load_image(self):
        """Load an image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Pellet Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            self.current_image_path = file_path
            self.current_image = cv2.imread(file_path)

            if self.current_image is not None:
                # Get annotations for this image
                self.get_annotations_for_image()
                # Process and display
                self.process_image()
            else:
                self.image_label.setText("Error loading image")

    def get_annotations_for_image(self):
        """Get annotations that match the loaded image"""
        self.annotations_for_image = []

        if not self.coco_data or not self.current_image_path:
            return

        # Get filename from path
        filename = os.path.basename(self.current_image_path)

        # Find image ID in COCO data
        image_id = None
        for img in self.coco_data.get('images', []):
            if img['file_name'] == filename:
                image_id = img['id']
                break

        if image_id is not None:
            # Get all annotations for this image
            for ann in self.coco_data.get('annotations', []):
                if ann['image_id'] == image_id:
                    self.annotations_for_image.append(ann)

            print(f"Found {len(self.annotations_for_image)} annotations for {filename}")
        else:
            print(f"No matching image found in COCO data for {filename}")

    def update_calibration(self, value):
        """Update calibration and reprocess image"""
        self.pixels_per_mm = value
        if self.current_image is not None:
            self.process_image()

    def process_image(self):
        """Process the image and detect pellets using COCO annotations"""
        if self.current_image is None:
            return

        # Create a copy for drawing
        display_image = self.current_image.copy()
        self.detected_pellets = []

        # Process each annotation
        for ann in self.annotations_for_image:
            if 'segmentation' not in ann or not ann['segmentation']:
                continue

            # Get polygon points
            for seg in ann['segmentation']:
                polygon = np.array(seg).reshape(-1, 2).astype(np.int32)

                # Calculate bounding box
                x, y, w, h = cv2.boundingRect(polygon)

                # Calculate dimensions in mm
                width_mm = w / self.pixels_per_mm
                height_mm = h / self.pixels_per_mm

                # Determine diameter and length
                diameter = min(width_mm, height_mm)
                length = max(width_mm, height_mm)

                # Check tolerance
                within_tolerance = (
                        (self.target_diameter - self.tolerance <= diameter <= self.target_diameter + self.tolerance) and
                        (self.target_length - self.tolerance <= length <= self.target_length + self.tolerance)
                )

                pellet_info = {
                    'polygon': polygon,
                    'bbox': (x, y, w, h),
                    'diameter': diameter,
                    'length': length,
                    'within_tolerance': within_tolerance
                }

                self.detected_pellets.append(pellet_info)

                # Draw on image
                self.draw_pellet(display_image, pellet_info)

        # Update statistics
        self.update_statistics()

        # Display image
        self.display_image(display_image)

    def draw_pellet(self, image, pellet):
        """Draw pellet detection on image"""
        polygon = pellet['polygon']
        x, y, w, h = pellet['bbox']
        diameter = pellet['diameter']
        length = pellet['length']
        within_tolerance = pellet['within_tolerance']

        # Color based on tolerance
        if within_tolerance:
            color = (0, 255, 0)  # Green
            status_color = (255, 255, 255)
        else:
            color = (0, 0, 255)  # Red
            status_color = (255, 255, 255)

        # Draw polygon with semi-transparent fill
        overlay = image.copy()
        cv2.fillPoly(overlay, [polygon], (255, 255, 0))
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

        # Draw polygon outline
        cv2.polylines(image, [polygon], True, (0, 255, 255), 2)

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # Draw measurements
        bg_y = max(y - 50, 0)
        cv2.rectangle(image, (x, bg_y), (x + 120, y - 5), (0, 0, 0), -1)
        cv2.putText(image, f"D: {diameter:.2f}mm", (x + 5, bg_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        cv2.putText(image, f"L: {length:.2f}mm", (x + 5, bg_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

        # Warning indicator if out of tolerance
        if not within_tolerance:
            cv2.circle(image, (x + w - 10, y + 10), 8, (0, 0, 255), -1)
            cv2.putText(image, "!", (x + w - 14, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def update_statistics(self):
        """Update statistics display"""
        total = len(self.detected_pellets)
        within = sum(1 for p in self.detected_pellets if p['within_tolerance'])
        out_of = total - within

        self.total_label.setText(f"Total Pellets: {total}")
        self.within_label.setText(f"Within Tolerance: {within}")
        self.out_label.setText(f"Out of Tolerance: {out_of}")

        if total == 0:
            self.status_label.setText("Status: No pellets detected")
            self.status_label.setStyleSheet("font-weight: bold; padding: 5px; color: gray;")
        elif out_of == 0:
            self.status_label.setText("Status: ✓ All Within Tolerance")
            self.status_label.setStyleSheet("font-weight: bold; padding: 5px; color: green;")
        else:
            self.status_label.setText(f"Status: ✗ {out_of} Out of Tolerance")
            self.status_label.setStyleSheet("font-weight: bold; padding: 5px; color: red;")

        # Update pellet details
        self.update_pellet_details()

    def update_pellet_details(self):
        """Update detailed pellet information"""
        # Clear existing details
        while self.details_widget_layout.count():
            child = self.details_widget_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add pellet details
        for i, pellet in enumerate(self.detected_pellets, 1):
            detail_text = f"Pellet {i}:\n"
            detail_text += f"  Diameter: {pellet['diameter']:.2f} mm\n"
            detail_text += f"  Length: {pellet['length']:.2f} mm\n"
            detail_text += f"  Status: {'✓ OK' if pellet['within_tolerance'] else '✗ Out'}"

            label = QLabel(detail_text)
            label.setStyleSheet(
                f"padding: 5px; margin: 2px; border: 1px solid "
                f"{'green' if pellet['within_tolerance'] else 'red'};"
            )
            self.details_widget_layout.addWidget(label)

        self.details_widget_layout.addStretch()

    def display_image(self, cv_image):
        """Display OpenCV image in QLabel"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        # Convert to QImage
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Convert to QPixmap and display
        pixmap = QPixmap.fromImage(qt_image)

        # Scale to fit while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setMinimumSize(1, 1)  # Allow shrinking


def main():
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    window = PelletMeasurementApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()