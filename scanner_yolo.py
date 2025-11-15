import sys
import csv
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSpinBox, QDoubleSpinBox, QGroupBox, QScrollArea,
                             QProgressBar, QMessageBox, QTextEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from ultralytics import YOLO
import torch


class YOLOTrainingThread(QThread):
    """Thread for training YOLO model"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, dataset_yaml_path):
        super().__init__()
        self.dataset_yaml_path = dataset_yaml_path

    def run(self):
        try:
            self.progress.emit("Initializing YOLO model...")
            # Start with YOLOv8 nano model for faster training
            model = YOLO('yolov8n.pt')

            self.progress.emit("Starting training... This may take several minutes.")

            # Train the model
            results = model.train(
                data=self.dataset_yaml_path,
                epochs=50,
                imgsz=640,
                batch=8,
                name='pellet_detector',
                patience=10,
                save=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            self.progress.emit("Training completed!")
            self.finished.emit(True, "runs/detect/pellet_detector/weights/best.pt")

        except Exception as e:
            self.progress.emit(f"Training failed: {str(e)}")
            self.finished.emit(False, str(e))


class DatasetPreparer:
    """Prepares YOLO format dataset from CSV"""

    @staticmethod
    def prepare_yolo_dataset(csv_path, images_folder, output_folder="yolo_dataset"):
        """Convert CSV annotations to YOLO format"""

        # Create directory structure
        train_images = os.path.join(output_folder, "images", "train")
        train_labels = os.path.join(output_folder, "labels", "train")
        os.makedirs(train_images, exist_ok=True)
        os.makedirs(train_labels, exist_ok=True)

        # Read CSV
        annotations = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = row['image_name']
                if image_name not in annotations:
                    annotations[image_name] = {
                        'width': int(row['image_width']),
                        'height': int(row['image_height']),
                        'boxes': []
                    }

                # Convert to YOLO format (normalized center x, y, width, height)
                bbox_x = int(row['bbox_x'])
                bbox_y = int(row['bbox_y'])
                bbox_width = int(row['bbox_width'])
                bbox_height = int(row['bbox_height'])
                img_width = int(row['image_width'])
                img_height = int(row['image_height'])

                # Calculate center coordinates
                center_x = (bbox_x + bbox_width / 2) / img_width
                center_y = (bbox_y + bbox_height / 2) / img_height
                norm_width = bbox_width / img_width
                norm_height = bbox_height / img_height

                annotations[image_name]['boxes'].append(
                    f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                )

        # Copy images and create label files
        copied_count = 0
        for image_name, data in annotations.items():
            src_image = os.path.join(images_folder, image_name)
            if os.path.exists(src_image):
                # Copy image
                dst_image = os.path.join(train_images, image_name)
                import shutil
                shutil.copy(src_image, dst_image)

                # Create label file
                label_name = os.path.splitext(image_name)[0] + '.txt'
                label_path = os.path.join(train_labels, label_name)
                with open(label_path, 'w') as f:
                    f.write('\n'.join(data['boxes']))

                copied_count += 1

        # Create dataset.yaml
        yaml_content = f"""path: {os.path.abspath(output_folder)}
train: images/train
val: images/train  # Using same as train for small datasets

nc: 1  # number of classes
names: ['pellet']  # class names
"""
        yaml_path = os.path.join(output_folder, "dataset.yaml")
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        return yaml_path, copied_count


class PelletMeasurementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Size Measurement System - YOLO ML")
        self.setGeometry(100, 100, 1400, 900)

        # Configuration
        self.pixels_per_mm = 10.0
        self.target_diameter = 3.0
        self.target_length = 3.0
        self.tolerance = 0.5

        # Data
        self.current_image = None
        self.current_image_path = None
        self.detected_pellets = []
        self.model = None
        self.is_trained = False

        self.images_folder = "pellet_training/"
        self.csv_path = "labels_pellet.csv"

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        left = self.create_left_panel()
        right = self.create_right_panel()
        main_layout.addWidget(left, 1)
        main_layout.addWidget(right, 3)

    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Training section
        train_group = QGroupBox("Model Training")
        train_layout = QVBoxLayout()

        self.train_btn = QPushButton("Train YOLO Model")
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setStyleSheet(
            "QPushButton { padding: 10px; font-size: 14px; background-color: #4CAF50; color: white; }")
        train_layout.addWidget(self.train_btn)

        self.training_status = QLabel("Status: Ready to train")
        self.training_status.setStyleSheet("padding: 5px; background-color: #e0e0e0;")
        self.training_status.setWordWrap(True)
        train_layout.addWidget(self.training_status)

        self.training_log = QTextEdit()
        self.training_log.setMaximumHeight(150)
        self.training_log.setReadOnly(True)
        train_layout.addWidget(QLabel("Training Log:"))
        train_layout.addWidget(self.training_log)

        train_group.setLayout(train_layout)
        layout.addWidget(train_group)

        # Detection section
        self.load_btn = QPushButton("Load Image for Detection")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setStyleSheet("QPushButton { padding: 10px; font-size: 14px; }")
        self.load_btn.setEnabled(False)
        layout.addWidget(self.load_btn)

        # Calibration
        calib = QGroupBox("Calibration")
        calib_l = QVBoxLayout()
        px_l = QHBoxLayout()
        px_l.addWidget(QLabel("Pixels per mm:"))
        self.px_spinbox = QDoubleSpinBox()
        self.px_spinbox.setRange(0.1, 100.0)
        self.px_spinbox.setValue(self.pixels_per_mm)
        self.px_spinbox.setSingleStep(0.1)
        self.px_spinbox.setDecimals(2)
        self.px_spinbox.valueChanged.connect(self.update_calibration)
        px_l.addWidget(self.px_spinbox)
        calib_l.addLayout(px_l)
        calib.setLayout(calib_l)
        layout.addWidget(calib)

        # Target specs
        spec = QGroupBox("Target Specifications")
        spec_l = QVBoxLayout()
        spec_l.addWidget(QLabel(f"Target Diameter: {self.target_diameter} mm"))
        spec_l.addWidget(QLabel(f"Target Length: {self.target_length} mm"))
        spec_l.addWidget(QLabel(f"Tolerance: ±{self.tolerance} mm"))
        spec_l.addWidget(QLabel(
            f"Acceptable: {self.target_diameter - self.tolerance:.1f}-"
            f"{self.target_diameter + self.tolerance:.1f} mm"))
        spec.setLayout(spec_l)
        layout.addWidget(spec)

        # Statistics
        self.stats_group = QGroupBox("Detection Statistics")
        self.stats_layout = QVBoxLayout()
        self.total_label = QLabel("Total Pellets: 0")
        self.within_label = QLabel("Within Tolerance: 0")
        self.out_label = QLabel("Out of Tolerance: 0")
        self.status_label = QLabel("Status: No image loaded")
        self.status_label.setStyleSheet("font-weight: bold; padding: 5px;")
        for w in (self.total_label, self.within_label, self.out_label, self.status_label):
            self.stats_layout.addWidget(w)
        self.stats_group.setLayout(self.stats_layout)
        layout.addWidget(self.stats_group)

        # Pellet details
        details = QGroupBox("Pellet Details")
        details_l = QVBoxLayout()
        self.details_scroll = QScrollArea()
        self.details_scroll.setWidgetResizable(True)
        self.details_widget = QWidget()
        self.details_widget_layout = QVBoxLayout()
        self.details_widget.setLayout(self.details_widget_layout)
        self.details_scroll.setWidget(self.details_widget)
        details_l.addWidget(self.details_scroll)
        details.setLayout(details_l)
        layout.addWidget(details)

        layout.addStretch()
        return panel

    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        self.image_label = QLabel("Train the YOLO model first, then load an image for detection")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.image_label.setMinimumSize(800, 600)

        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        return panel

    def start_training(self):
        """Start YOLO model training"""
        if not os.path.exists(self.csv_path):
            QMessageBox.critical(self, "Error", f"CSV file not found: {self.csv_path}")
            return

        if not os.path.exists(self.images_folder):
            QMessageBox.critical(self, "Error", f"Images folder not found: {self.images_folder}")
            return

        self.train_btn.setEnabled(False)
        self.training_status.setText("Preparing dataset...")
        self.training_log.append("Starting dataset preparation...")

        try:
            # Prepare YOLO dataset
            yaml_path, count = DatasetPreparer.prepare_yolo_dataset(
                self.csv_path, self.images_folder
            )
            self.training_log.append(f"Dataset prepared: {count} images")
            self.training_status.setText(f"Dataset ready. Training on {count} images...")

            # Start training in separate thread
            self.training_thread = YOLOTrainingThread(yaml_path)
            self.training_thread.progress.connect(self.on_training_progress)
            self.training_thread.finished.connect(self.on_training_finished)
            self.training_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Dataset preparation failed: {str(e)}")
            self.train_btn.setEnabled(True)

    def on_training_progress(self, message):
        """Update training progress"""
        self.training_log.append(message)
        self.training_status.setText(message)

    def on_training_finished(self, success, result):
        """Handle training completion"""
        self.train_btn.setEnabled(True)

        if success:
            self.training_log.append(f"Training completed! Model saved at: {result}")
            self.training_status.setText("Training completed successfully!")

            # Load the trained model
            try:
                self.model = YOLO(result)
                self.is_trained = True
                self.load_btn.setEnabled(True)
                QMessageBox.information(self, "Success",
                                        "Model training completed!\nYou can now load images for detection.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load trained model: {str(e)}")
        else:
            self.training_log.append(f"Training failed: {result}")
            self.training_status.setText("Training failed!")
            QMessageBox.critical(self, "Error", f"Training failed: {result}")

    def load_image(self):
        """Load image for detection"""
        if not self.is_trained:
            QMessageBox.warning(self, "Not Trained", "Train the model first!")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Pellet Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

        if path:
            self.current_image_path = path
            self.current_image = cv2.imread(path)
            if self.current_image is not None:
                self.training_status.setText("Detecting pellets...")
                QApplication.processEvents()
                self.process_image()
                self.training_status.setText("Detection complete")
            else:
                self.image_label.setText("Error loading image")

    def update_calibration(self, value):
        self.pixels_per_mm = value
        if self.current_image is not None:
            self.process_image()

    def process_image(self):
        """Detect pellets using YOLO and measure them"""
        if self.current_image is None or not self.is_trained:
            return

        # Run YOLO detection
        results = self.model(self.current_image, conf=0.25)

        display_img = self.current_image.copy()
        self.detected_pellets = []

        # Process each detection
        for result in results:
            boxes = result.boxes
            for idx, box in enumerate(boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()

                # Create polygon from bounding box
                polygon = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ], dtype=np.int32)

                # Measure the detected pellet
                meas = self.measure_pellet(polygon, self.pixels_per_mm)

                pellet = {
                    'polygon': polygon,
                    'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    'diameter': meas['diameter'],
                    'length': meas['length'],
                    'within_tolerance': meas['within'],
                    'confidence': float(confidence) * 100,
                    'id': idx + 1
                }
                self.detected_pellets.append(pellet)
                self.draw_pellet(display_img, pellet)

        self.update_statistics()
        self.display_image(display_img)

    def measure_pellet(self, polygon, pixels_per_mm):
        """Measure pellet dimensions from detected polygon"""
        # Get rotated rectangle
        rect = cv2.minAreaRect(polygon.astype(np.float32))
        width_px, height_px = rect[1]

        # Convert to mm
        width_mm = width_px / pixels_per_mm
        height_mm = height_px / pixels_per_mm

        diameter = min(width_mm, height_mm)
        length = max(width_mm, height_mm)

        within = (
                (self.target_diameter - self.tolerance <= diameter <= self.target_diameter + self.tolerance) and
                (self.target_length - self.tolerance <= length <= self.target_length + self.tolerance)
        )

        return {"diameter": diameter, "length": length, "within": within}

    def draw_pellet(self, image, pellet):
        """Draw detection on image"""
        poly = pellet['polygon']
        pid = pellet['id']
        ok = pellet['within_tolerance']
        colour = (0, 255, 0) if ok else (0, 0, 255)

        # Semi-transparent fill
        overlay = image.copy()
        cv2.fillPoly(overlay, [poly.reshape(-1, 1, 2)], colour)
        cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)

        # Thick outline
        cv2.polylines(image, [poly.reshape(-1, 1, 2)], True, colour, 2)

        # Centered number
        M = cv2.moments(poly)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else int(poly[:, 0].mean())
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else int(poly[:, 1].mean())
        cv2.putText(image, str(pid), (cx - 12, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def update_statistics(self):
        total = len(self.detected_pellets)
        within = sum(1 for p in self.detected_pellets if p['within_tolerance'])
        out = total - within

        self.total_label.setText(f"Total Pellets: {total}")
        self.within_label.setText(f"Within Tolerance: {within}")
        self.out_label.setText(f"Out of Tolerance: {out}")

        if total == 0:
            self.status_label.setText("Status: No pellets detected")
            self.status_label.setStyleSheet("font-weight: bold; padding: 5px; color: gray;")
        elif out == 0:
            self.status_label.setText("Status: All Within Tolerance")
            self.status_label.setStyleSheet("font-weight: bold; padding: 5px; color: green;")
        else:
            self.status_label.setText(f"Status: {out} Out of Tolerance")
            self.status_label.setStyleSheet("font-weight: bold; padding: 5px; color: red;")

        self.update_pellet_details()

    def update_pellet_details(self):
        while self.details_widget_layout.count():
            child = self.details_widget_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for i, p in enumerate(self.detected_pellets, 1):
            txt = (f"Pellet {i}:\n"
                   f"  Diameter: {p['diameter']:.2f} mm\n"
                   f"  Length:   {p['length']:.2f} mm\n"
                   f"  Confidence: {p['confidence']:.1f}%\n"
                   f"  Status: {'OK' if p['within_tolerance'] else 'Out'}")
            lbl = QLabel(txt)
            lbl.setStyleSheet(
                f"padding:5px;margin:2px;border:1px solid {'green' if p['within_tolerance'] else 'red'};")
            self.details_widget_layout.addWidget(lbl)
        self.details_widget_layout.addStretch()

    def display_image(self, cv_image):
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.image_label.size(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.image_label.setMinimumSize(1, 1)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = PelletMeasurementApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()