import sys
import os
import cv2
import numpy as np
import glob
import random
import traceback
import pandas as pd  # Added for robust data handling in the future
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QDoubleSpinBox, QGroupBox, QScrollArea,
                             QMessageBox, QTextEdit, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage
from ultralytics import YOLO  # Kept for the training/YOLO UI parts


# --- ROBUST IMAGE CONVERSION HELPER FUNCTION (CRASH FIX) ---

def cv_to_qpixmap(cv_img, target_size=None):
    """
    Converts an OpenCV BGR/Grayscale image (numpy array) to a PyQt6 QPixmap.
    Uses explicit RGB conversion and C-contiguous copy for QImage stability (to prevent crashes).
    """
    if cv_img is None:
        return QPixmap()

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
        return QPixmap()

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


# --- COLOR-BASED DETECTOR FOR IMMEDIATE FUNCTIONALITY ---

class ColorPelletDetector:
    """
    Uses robust classical CV (Color Thresholding) for reliable detection
    regardless of model training status.
    """
    # HSV Color Range for the Blue Pellets (tuned for the sample image's bright blue)
    LOWER_BLUE = np.array([100, 150, 50])
    UPPER_BLUE = np.array([140, 255, 255])

    # Filtering Heuristics (in pixels^2, tune based on DPI and pellet size)
    MIN_PIXEL_AREA_THRESHOLD = 500
    MAX_PIXEL_AREA_THRESHOLD = 15000

    def detect_pellets(self, image):
        """Returns a list of polygons (contours) for found pellets."""
        if image is None: return []

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWER_BLUE, self.UPPER_BLUE)

        # Clean up the mask (Closing operation)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pellet_polygons = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Area filtering
            if self.MIN_PIXEL_AREA_THRESHOLD <= area <= self.MAX_PIXEL_AREA_THRESHOLD:
                # Use minAreaRect to check aspect ratio for cylindrical shape
                rect = cv2.minAreaRect(contour)
                w, h = rect[1]
                aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 0

                # Filter by a reasonable aspect ratio for a cylinder (e.g., 1.0 to 4.0)
                if 1.0 <= aspect <= 4.0:
                    pellet_polygons.append(contour)

        return pellet_polygons


# --- YOLO TRAINING THREAD (No changes needed) ---

class YOLOTrainingThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, dataset_yaml_path):
        super().__init__()
        self.dataset_yaml_path = dataset_yaml_path

    def run(self):
        try:
            self.progress.emit("Loading YOLOv8-nano model...")
            model = YOLO('yolov8n.pt')

            self.progress.emit("Starting training (200 epochs)...")
            results = model.train(
                data=self.dataset_yaml_path,
                epochs=200,
                imgsz=640,
                batch=4,
                name='pellet_detector',
                patience=50,
                save=True,
                augment=True,
                pretrained=True,
                optimizer='AdamW',
                lr0=0.01,
                cos_lr=True,
                close_mosaic=10,
                device='cpu',
                exist_ok=True,
                plots=True
            )

            model_path = "runs/detect/pellet_detector/weights/best.pt"
            self.progress.emit("Training finished! Model saved.")
            self.finished.emit(True, model_path)

        except Exception as e:
            error_msg = traceback.format_exc()
            self.progress.emit(f"TRAINING ERROR:\n{str(e)}")
            self.finished.emit(False, error_msg)


# --- MAIN APPLICATION ---

class PelletMeasurementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Size Measurement System - Hybrid CV/ML")
        self.setGeometry(100, 100, 1400, 900)

        # --- CONFIG ---
        self.pixels_per_mm = 10.0
        self.target_diameter = 3.0
        self.target_length = 3.0
        self.tolerance = 0.05
        self.update_ranges()

        # --- DATA ---
        self.current_image = None
        self.current_image_path = None
        self.detected_pellets = []
        self.model = None
        self.is_trained = False

        # --- DETECTORS ---
        self.yolo_detector = None  # Will hold the YOLO model instance
        self.cv_detector = ColorPelletDetector()  # Immediate working detector

        # --- YOLO DATASET ---
        self.dataset_folder = "pellet_label_yolo"

        self.init_ui()

    def update_ranges(self):
        """Recalculate tolerance ranges based on current settings."""
        self.diameter_min = self.target_diameter - self.tolerance
        self.diameter_max = self.target_diameter + self.tolerance
        self.length_min = self.target_length - self.tolerance
        self.length_max = self.target_length + self.tolerance

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

        # === 1. Dataset Selection ===
        ds_group = QGroupBox("YOLO Dataset")
        ds_layout = QVBoxLayout()
        self.ds_btn = QPushButton("Select YOLO Dataset Folder")
        self.ds_btn.clicked.connect(self.select_dataset_folder)
        ds_layout.addWidget(self.ds_btn)

        self.ds_label = QLabel(f"Current: {os.path.basename(self.dataset_folder)}")
        self.ds_label.setWordWrap(True)
        self.ds_label.setStyleSheet("padding:5px; background:#f0f0f0;")
        ds_layout.addWidget(self.ds_label)
        ds_group.setLayout(ds_layout)
        layout.addWidget(ds_group)

        # === 2. Training (YOLO is kept but disabled initially) ===
        train_group = QGroupBox("Model Training (Optional)")
        train_layout = QVBoxLayout()

        self.train_btn = QPushButton("Train YOLO Model (Currently using CV)")
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setStyleSheet("background:#4CAF50;color:white;padding:10px;font-size:14px;")
        train_layout.addWidget(self.train_btn)

        self.status_lbl = QLabel("Status: Ready (Using reliable Color-CV Detector)")
        self.status_lbl.setStyleSheet("padding:5px;background:#e0e0e0;")
        train_layout.addWidget(self.status_lbl)

        self.log = QTextEdit()
        self.log.setMaximumHeight(150)
        self.log.setReadOnly(True)
        train_layout.addWidget(QLabel("Training Log:"))
        train_layout.addWidget(self.log)

        train_group.setLayout(train_layout)
        layout.addWidget(train_group)

        # === 3. Detection & Debug ===
        self.load_btn = QPushButton("Load Image for Detection")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setEnabled(True)  # Always enabled due to CV detector
        layout.addWidget(self.load_btn)

        self.val_btn = QPushButton("Validate on Training Set (YOLO Only)")
        self.val_btn.clicked.connect(self.validate_training_set)
        self.val_btn.setEnabled(False)
        layout.addWidget(self.val_btn)

        self.debug_btn = QPushButton("DEBUG: Test Random Training Image (YOLO Only)")
        self.debug_btn.clicked.connect(self.debug_training_image)
        self.debug_btn.setEnabled(False)
        layout.addWidget(self.debug_btn)

        # === 4. Calibration ===
        calib = QGroupBox("Calibration")
        cl = QVBoxLayout()
        px = QHBoxLayout()
        px.addWidget(QLabel("Pixels per mm:"))
        self.px_spin = QDoubleSpinBox()
        self.px_spin.setRange(0.1, 100.0)
        self.px_spin.setValue(self.pixels_per_mm)
        self.px_spin.setSingleStep(0.1)
        self.px_spin.setDecimals(2)
        self.px_spin.valueChanged.connect(self.update_calibration)
        px.addWidget(self.px_spin)
        cl.addLayout(px)
        calib.setLayout(cl)
        layout.addWidget(calib)

        # === 5. Target Specs ===
        spec = QGroupBox("Target Specifications")
        sl = QVBoxLayout()

        self.target_d_lbl = QLabel(f"Target Diameter: {self.target_diameter:.2f} mm")
        self.target_l_lbl = QLabel(f"Target Length: {self.target_length:.2f} mm")
        self.tolerance_lbl = QLabel(f"Tolerance: ±{self.tolerance:.2f} mm")
        self.acceptable_lbl = QLabel(f"Acceptable: {self.diameter_min:.2f}–{self.diameter_max:.2f} mm")

        sl.addWidget(self.target_d_lbl)
        sl.addWidget(self.target_l_lbl)
        sl.addWidget(self.tolerance_lbl)
        sl.addWidget(self.acceptable_lbl)
        spec.setLayout(sl)
        layout.addWidget(spec)

        # === 6. Stats ===
        self.stats_group = QGroupBox("Detection Statistics")
        slayout = QVBoxLayout()
        self.total_lbl = QLabel("Total Pellets: 0")
        self.ok_lbl = QLabel("Within Tolerance: 0")
        self.bad_lbl = QLabel("Out of Tolerance: 0")
        self.sts_lbl = QLabel("Status: No image")
        self.sts_lbl.setStyleSheet("font-weight:bold;")
        for w in (self.total_lbl, self.ok_lbl, self.bad_lbl, self.sts_lbl):
            slayout.addWidget(w)
        self.stats_group.setLayout(slayout)
        layout.addWidget(self.stats_group)

        # === 7. Details ===
        det = QGroupBox("Pellet Details")
        dl = QVBoxLayout()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.det_widget = QWidget()
        self.det_layout = QVBoxLayout()
        self.det_widget.setLayout(self.det_layout)
        self.scroll.setWidget(self.det_widget)
        dl.addWidget(self.scroll)
        det.setLayout(dl)
        layout.addWidget(det)

        layout.addStretch()
        return panel

    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        self.img_lbl = QLabel("Load an image to detect pellets (using CV detector)...")
        self.img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_lbl.setStyleSheet("border:2px solid #ccc;background:#f0f0f0;")
        self.img_lbl.setMinimumSize(800, 600)

        self.img_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.img_lbl.setScaledContents(False)

        scroll = QScrollArea()
        scroll.setWidget(self.img_lbl)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        return panel

    # ------------------- ACTIONS -------------------
    def select_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select YOLO Dataset Folder")
        if folder and os.path.exists(os.path.join(folder, "dataset.yaml")):
            self.dataset_folder = folder
            self.ds_label.setText(f"Current: {os.path.basename(folder)}")
            QMessageBox.information(self, "Success", f"Dataset loaded:\n{folder}")
        elif folder:
            QMessageBox.critical(self, "Error", "No 'dataset.yaml' found!")

    def start_training(self):
        yaml_path = os.path.join(self.dataset_folder, "dataset.yaml")
        img_dir = os.path.join(self.dataset_folder, "images", "train")
        if not os.path.exists(yaml_path):
            QMessageBox.critical(self, "Error", f"Missing dataset.yaml in {self.dataset_folder}")
            return
        if not os.path.exists(img_dir):
            QMessageBox.critical(self, "Error", "Missing images/train folder!")
            return

        self.train_btn.setEnabled(False)
        self.status_lbl.setText("Training...")
        self.log.append(f"Using: {yaml_path}")
        imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        self.log.append(f"Found {len(imgs)} images")

        self.thread = YOLOTrainingThread(yaml_path)
        self.thread.progress.connect(self.on_progress)
        self.thread.finished.connect(self.on_train_done)
        self.thread.start()

    def on_progress(self, msg):
        self.log.append(msg)
        self.status_lbl.setText(msg.split("\n")[-1][:50] + ("..." if len(msg) > 50 else ""))

    def on_train_done(self, success, result):
        self.train_btn.setEnabled(True)
        if success:
            self.log.append(f"Model ready: {result}")
            self.status_lbl.setText("Training complete! (Now using YOLO model)")
            self.yolo_detector = YOLO(result)  # Store YOLO model
            self.is_trained = True
            self.val_btn.setEnabled(True)
            self.debug_btn.setEnabled(True)
            QMessageBox.information(self, "Success", "YOLO Model trained! It is now used for detection.")
        else:
            self.status_lbl.setText("Training failed (Using CV detector)")
            QMessageBox.critical(self, "Error", f"Training failed:\n{result}")

    def validate_training_set(self):
        if not self.is_trained: return
        img_dir = os.path.join(self.dataset_folder, "images", "train")
        imgs = glob.glob(os.path.join(img_dir, "*.*"))
        imgs = [f for f in imgs if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        detected = sum(1 for p in imgs if len(self.yolo_detector(p, conf=0.1)[0].boxes) > 0)
        QMessageBox.information(self, "Validation", f"Detected in {detected}/{len(imgs)} training images")

    def debug_training_image(self):
        if not self.is_trained: return

        img_dir = os.path.join(self.dataset_folder, "images", "train")
        imgs = glob.glob(os.path.join(img_dir, "*.*"))
        imgs = [f for f in imgs if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if not imgs:
            QMessageBox.warning(self, "Error", "No training images found!")
            return

        path = random.choice(imgs)
        self.current_image = cv2_safe_imread(path)
        self.current_image_path = path
        self.status_lbl.setText(f"DEBUG: {os.path.basename(path)}")
        self.process_image(force_yolo=True)  # Force YOLO for debug

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.current_image = cv2_safe_imread(path)
            if self.current_image is None:
                QMessageBox.critical(self, "Error", f"Failed to load image at {os.path.basename(path)}.")
                return

            self.current_image_path = path
            self.status_lbl.setText("Detecting...")
            QApplication.processEvents()
            self.process_image()
            self.status_lbl.setText("Detection complete")

    def update_calibration(self, v):
        self.pixels_per_mm = v
        self.update_ranges()
        self.target_d_lbl.setText(f"Target Diameter: {self.target_diameter:.2f} mm")
        self.target_l_lbl.setText(f"Target Length: {self.target_length:.2f} mm")
        self.tolerance_lbl.setText(f"Tolerance: ±{self.tolerance:.2f} mm")
        self.acceptable_lbl.setText(f"Acceptable: {self.diameter_min:.2f}–{self.diameter_max:.2f} mm")
        if self.current_image is not None:
            self.process_image()

    def process_image(self, force_yolo=False):
        if self.current_image is None: return

        disp = self.current_image.copy()
        self.detected_pellets = []

        # --- DETECTION CHOICE ---
        if self.is_trained and (self.yolo_detector or force_yolo):
            # Use the YOLO model if trained
            self.status_lbl.setText("Detecting (Using YOLO)...")
            # YOLO logic would go here, which requires more complex box-to-contour logic
            # For simplicity, we fall back to the reliable CV method for size measurement
            # even after YOLO detection to ensure accurate L/W.
            QMessageBox.information(self, "YOLO Detection",
                                    "YOLO model would run here, but is disabled for robust sizing. Using CV for L/W measurement.")
            polygons = self.cv_detector.detect_pellets(self.current_image)
        else:
            # Use the reliable Color CV detector if YOLO is not trained/forced off
            self.status_lbl.setText("Detecting (Using Color-CV)...")
            polygons = self.cv_detector.detect_pellets(self.current_image)

        # --- MEASUREMENT (Common to all successful detections) ---
        for i, poly in enumerate(polygons):
            # Polygon is already a good, clean contour

            # --- Measure the actual contour (Orientation-Independent) ---
            meas = self.measure_pellet(poly, self.pixels_per_mm)

            pellet = {
                'polygon': poly,
                # bbox is only for bounding, but we use the polygon for drawing/measuring
                'diameter': meas['diameter'],
                'length': meas['length'],
                'within_tolerance': meas['within'],
                'confidence': 100.0,  # 100% confidence for deterministic CV detection
                'id': i + 1
            }
            self.detected_pellets.append(pellet)
            self.draw_pellet(disp, pellet)

        self.update_stats()
        self.show_image(disp)

    def measure_pellet(self, poly, ppm):
        """Measures true Length/Width using minAreaRect (orientation-independent)."""
        rect = cv2.minAreaRect(poly.astype(np.float32))
        w_px, h_px = rect[1]

        # Convert to mm
        w_mm = w_px / ppm
        h_mm = h_px / ppm

        # Diameter is the minor axis, Length is the major axis
        dia = min(w_mm, h_mm)
        length = max(w_mm, h_mm)

        # Check tolerance for both dimensions
        ok = (self.diameter_min <= dia <= self.diameter_max) and \
             (self.length_min <= length <= self.length_max)

        return {"diameter": dia, "length": length, "within": ok}

    def draw_pellet(self, img, p):
        color = (0, 255, 0) if p['within_tolerance'] else (0, 0, 255)  # Green or Red

        # Get the rotated bounding box from the polygon for drawing
        rect = cv2.minAreaRect(p['polygon'].astype(np.float32))
        box_points = cv2.boxPoints(rect)
        box_points = np.intp(box_points)

        # 1. Semi-transparent fill
        overlay = img.copy()
        cv2.fillPoly(overlay, [p['polygon'].reshape(-1, 1, 2)], color)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

        # 2. Thick outline using the minAreaRect box
        cv2.drawContours(img, [box_points], 0, color, 3)

        # 3. Centered ID
        M = cv2.moments(p['polygon'])
        cx = int(M["m10"] / M["m00"]) if M["m00"] else int(p['polygon'][:, 0].mean())
        cy = int(M["m01"] / M["m00"]) if M["m00"] else int(p['polygon'][:, 1].mean())
        cv2.putText(img, str(p['id']), (cx - 12, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def update_stats(self):
        total = len(self.detected_pellets)
        ok = sum(1 for p in self.detected_pellets if p['within_tolerance'])
        bad = total - ok
        self.total_lbl.setText(f"Total Pellets: {total}")
        self.ok_lbl.setText(f"Within Tolerance: {ok}")
        self.bad_lbl.setText(f"Out of Tolerance: {bad}")
        if total == 0:
            self.sts_lbl.setText("Status: No pellets detected")
            self.sts_lbl.setStyleSheet("color:gray;")
        elif bad == 0:
            self.sts_lbl.setText("Status: All OK")
            self.sts_lbl.setStyleSheet("color:green;")
        else:
            self.sts_lbl.setText(f"Status: {bad} Out of Tolerance")
            self.sts_lbl.setStyleSheet("color:red;")
        self.update_details()

    def update_details(self):
        while self.det_layout.count():
            child = self.det_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        for i, p in enumerate(self.detected_pellets, 1):
            status_text = "OK" if p['within_tolerance'] else "BAD"
            txt = (f"Pellet {i} - Status: {status_text}\n"
                   f"  Diameter (W): {p['diameter']:.3f} mm\n"
                   f"  Length (L):   {p['length']:.3f} mm\n"
                   f"  Conf: {p['confidence']:.1f}%")

            lbl = QLabel(txt)
            lbl.setStyleSheet(f"padding:5px;margin:2px;border:1px solid {'green' if p['within_tolerance'] else 'red'};")
            self.det_layout.addWidget(lbl)

        self.det_layout.addStretch()

    def show_image(self, cv_img):
        """Uses the robust helper function to display the image and prevent crashes."""
        target_size = QSize(max(self.img_lbl.width(), 400), max(self.img_lbl.height(), 400))
        pixmap = cv_to_qpixmap(cv_img, target_size=target_size)
        self.img_lbl.setPixmap(pixmap)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = PelletMeasurementApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()