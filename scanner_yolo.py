import sys
import os
import cv2
import numpy as np
import glob
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QDoubleSpinBox, QGroupBox, QScrollArea,
                             QMessageBox, QTextEdit)
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
            self.progress.emit("Initializing YOLOv8-nano model...")
            model = YOLO('yolov8n.pt')

            self.progress.emit("Starting training... This may take several minutes.")

            results = model.train(
                data=self.dataset_yaml_path,
                epochs=100,
                imgsz=640,
                batch=16,
                name='pellet_detector',
                patience=20,
                save=True,
                augment=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            model_path = "runs/detect/pellet_detector/weights/best.pt"
            self.progress.emit("Training completed!")
            self.finished.emit(True, model_path)

        except Exception as e:
            self.progress.emit(f"Training failed: {str(e)}")
            self.finished.emit(False, str(e))


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

        # YOLO Dataset Folder
        self.dataset_folder = "pellet_label_yolo"  # YOUR FOLDER NAME

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

        # === Dataset Selection ===
        dataset_group = QGroupBox("YOLO Dataset")
        dataset_layout = QVBoxLayout()

        self.dataset_btn = QPushButton("Select YOLO Dataset Folder")
        self.dataset_btn.clicked.connect(self.select_dataset_folder)
        self.dataset_btn.setStyleSheet("QPushButton { padding: 8px; font-size: 13px; }")
        dataset_layout.addWidget(self.dataset_btn)

        self.dataset_label = QLabel(f"Current: {os.path.basename(self.dataset_folder)}")
        self.dataset_label.setWordWrap(True)
        self.dataset_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        dataset_layout.addWidget(self.dataset_label)

        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)

        # === Training ===
        train_group = QGroupBox("Model Training")
        train_layout = QVBoxLayout()

        self.train_btn = QPushButton("Train YOLO Model")
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setStyleSheet(
            "QPushButton { padding: 10px; font-size: 14px; background-color: #4CAF50; color: white; }")
        train_layout.addWidget(self.train_btn)

        self.training_status = QLabel("Status: Ready")
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

        # === Detection ===
        self.load_btn = QPushButton("Load Image for Detection")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setStyleSheet("QPushButton { padding: 10px; font-size: 14px; }")
        self.load_btn.setEnabled(False)
        layout.addWidget(self.load_btn)

        # === Validation Button ===
        self.validate_btn = QPushButton("Validate on Training Set")
        self.validate_btn.clicked.connect(self.validate_training_set)
        self.validate_btn.setEnabled(False)
        layout.addWidget(self.validate_btn)

        # === Calibration ===
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

        # === Target Specs ===
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

        # === Statistics ===
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

        # === Pellet Details ===
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

        self.image_label = QLabel("Train the model first, then load an image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.image_label.setMinimumSize(800, 600)

        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        return panel

    def select_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select YOLO Dataset Folder")
        if folder and os.path.exists(os.path.join(folder, "dataset.yaml")):
            self.dataset_folder = folder
            self.dataset_label.setText(f"Current: {os.path.basename(folder)}")
            QMessageBox.information(self, "Success", f"Dataset loaded:\n{folder}")
        elif folder:
            QMessageBox.critical(self, "Error", "No 'dataset.yaml' found in selected folder!")

    def start_training(self):
        yaml_path = os.path.join(self.dataset_folder, "dataset.yaml")
        train_img_dir = os.path.join(self.dataset_folder, "images", "train")

        if not os.path.exists(yaml_path):
            QMessageBox.critical(self, "Error", f"dataset.yaml not found in {self.dataset_folder}")
            return
        if not os.path.exists(train_img_dir):
            QMessageBox.critical(self, "Error", f"images/train folder not found!")
            return

        self.train_btn.setEnabled(False)
        self.training_status.setText("Starting training...")
        self.training_log.append("Using YOLO dataset:")
        self.training_log.append(f"  {yaml_path}")

        img_files = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        self.training_log.append(f"Found {len(img_files)} training images")

        self.training_thread = YOLOTrainingThread(yaml_path)
        self.training_thread.progress.connect(self.on_training_progress)
        self.training_thread.finished.connect(self.on_training_finished)
        self.training_thread.start()

    def on_training_progress(self, message):
        self.training_log.append(message)
        self.training_status.setText(message)

    def on_training_finished(self, success, result):
        self.train_btn.setEnabled(True)

        if success:
            self.training_log.append(f"Model saved: {result}")
            self.training_status.setText("Training complete!")
            try:
                self.model = YOLO(result)
                self.is_trained = True
                self.load_btn.setEnabled(True)
                self.validate_btn.setEnabled(True)
                QMessageBox.information(self, "Success", "Model ready! Load an image to detect.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
        else:
            self.training_status.setText("Training failed!")
            QMessageBox.critical(self, "Error", f"Training failed: {result}")

    def validate_training_set(self):
        if not self.is_trained:
            return
        img_dir = os.path.join(self.dataset_folder, "images", "train")
        images = glob.glob(os.path.join(img_dir, "*.*"))
        images = [f for f in images if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        detected = 0
        for img_path in images:
            img = cv2.imread(img_path)
            results = self.model(img, conf=0.1)
            if len(results[0].boxes) > 0:
                detected += 1

        QMessageBox.information(self, "Validation",
                                f"Detected pellets in {detected}/{len(images)} training images")

    def load_image(self):
        if not self.is_trained:
            QMessageBox.warning(self, "Not Trained", "Train the model first!")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Pellet Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

        if path:
            self.current_image_path = path
            self.current_image = cv2.imread(path)
            if self.current_image is not None:
                self.training_status.setText("Detecting...")
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
        if self.current_image is None or not self.is_trained:
            return

        # Resize to 640x640 for consistency
        img_resized = cv2.resize(self.current_image, (640, 640))
        results = self.model(img_resized, conf=0.1, iou=0.45)

        scale_x = self.current_image.shape[1] / 640
        scale_y = self.current_image.shape[0] / 640

        display_img = self.current_image.copy()
        self.detected_pellets = []

        for result in results:
            boxes = result.boxes
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1 *= scale_x; y1 *= scale_y
                x2 *= scale_x; y2 *= scale_y

                polygon = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.int32)
                meas = self.measure_pellet(polygon, self.pixels_per_mm)

                pellet = {
                    'polygon': polygon,
                    'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                    'diameter': meas['diameter'],
                    'length': meas['length'],
                    'within_tolerance': meas['within'],
                    'confidence': box.conf[0].item() * 100,
                    'id': idx + 1
                }
                self.detected_pellets.append(pellet)
                self.draw_pellet(display_img, pellet)

        self.update_statistics()
        self.display_image(display_img)

    def measure_pellet(self, polygon, pixels_per_mm):
        rect = cv2.minAreaRect(polygon.astype(np.float32))
        width_px, height_px = rect[1]
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
        poly = pellet['polygon']
        ok = pellet['within_tolerance']
        colour = (0, 255, 0) if ok else (0, 0, 255)
        overlay = image.copy()
        cv2.fillPoly(overlay, [poly.reshape(-1, 1, 2)], colour)
        cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)
        cv2.polylines(image, [poly.reshape(-1, 1, 2)], True, colour, 2)
        M = cv2.moments(poly)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else int(poly[:, 0].mean())
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else int(poly[:, 1].mean())
        cv2.putText(image, str(pellet['id']), (cx - 12, cy + 8),
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
            self.status_label.setStyleSheet("color: gray;")
        elif out == 0:
            self.status_label.setText("Status: All Within Tolerance")
            self.status_label.setStyleSheet("color: green;")
        else:
            self.status_label.setText(f"Status: {out} Out of Tolerance")
            self.status_label.setStyleSheet("color: red;")
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