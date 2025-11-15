# --------------------------------------------------------------
#  Pellet Size Measurement System – YOLOv8 version
# --------------------------------------------------------------
import sys
import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSpinBox, QDoubleSpinBox, QGroupBox, QScrollArea,
                             QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

# -------------------------- YOLO wrapper -----------------------
from ultralytics import YOLO


class YoloPelletDetector:
    """Very thin wrapper around YOLOv8 that mimics the old API."""

    def __init__(self, model_path: str = "runs/detect/train/weights/best.pt"):
        self.model_path = Path(model_path)
        self.model = None
        self.is_trained = False

    # ----------------------------------------------------------
    # 1. Convert CSV → YOLO txt files (once, at training time)
    # ----------------------------------------------------------
    @staticmethod
    def _csv_to_yolo_txt(csv_path: str, img_folder: str, out_folder: str):
        df = pd.read_csv(csv_path, header=None,
                         names=["label", "x", "y", "w", "h", "img_name", "img_w", "img_h"])

        os.makedirs(out_folder, exist_ok=True)

        # YOLO format:  class_id  cx_norm  cy_norm  w_norm  h_norm
        for img_name, group in df.groupby("img_name"):
            img_path = Path(img_folder) / img_name
            if not img_path.exists():
                continue

            txt_path = Path(out_folder) / (img_path.stem + ".txt")
            with open(txt_path, "w") as f:
                for _, row in group.iterrows():
                    # class id = 0 (only one class – pellet)
                    cx = (row["x"] + row["w"] / 2) / row["img_w"]
                    cy = (row["y"] + row["h"] / 2) / row["img_h"]
                    wn = row["w"] / row["img_w"]
                    hn = row["h"] / row["img_h"]
                    f.write(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n")

    # ----------------------------------------------------------
    # 2. Train (or just load a pre-trained model)
    # ----------------------------------------------------------
    def train(self, csv_path: str, img_folder: str, epochs: int = 30, imgsz: int = 640):
        """Convert CSV → YOLO txt → train a fresh model."""
        label_dir = "labels_yolo"
        self._csv_to_yolo_txt(csv_path, img_folder, label_dir)

        # create a minimal data.yaml
        yaml_content = f"""
path: .
train: {img_folder}
val: {img_folder}
nc: 1
names: ['pellet']
"""
        Path("data.yaml").write_text(yaml_content)

        # Ultralytics will create the `runs/` folder automatically
        model = YOLO("yolov8n.pt")               # start from nano (fast)
        model.train(data="data.yaml",
                    epochs=epochs,
                    imgsz=imgsz,
                    project="runs",
                    name="detect",
                    exist_ok=True)

        # keep the best weights
        self.model_path = Path("runs/detect/train/weights/best.pt")
        self.load_model()
        self.is_trained = True

    # ----------------------------------------------------------
    # 3. Load a ready model (skip training if you already have one)
    # ----------------------------------------------------------
    def load_model(self):
        if self.model_path.exists():
            self.model = YOLO(str(self.model_path))
            self.is_trained = True
        else:
            raise FileNotFoundError(f"Model not found: {self.model_path}")

    # ----------------------------------------------------------
    # 4. Inference – returns the same dict format as the old detector
    # ----------------------------------------------------------
    def detect_pellets(self, image: np.ndarray):
        if not self.is_trained:
            return []

        results = self.model(image, conf=0.25, iou=0.45, imgsz=640)[0]   # first image

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            # YOLO gives a rectangle → treat it as polygon (4 points)
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

            detections.append({
                'bbox': (x1, y1, x2 - x1, y2 - y1),
                'polygon': poly,
                'confidence': conf * 100,          # keep the old scale (0-100)
                'method': 'yolo'
            })
        return detections


# ==============================================================
# <<<--- GUI (unchanged except the detector initialisation) ----
# ==============================================================
class PelletMeasurementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Size Measurement System – YOLOv8")
        self.setGeometry(100, 100, 1400, 800)

        # ---- configuration -------------------------------------------------
        self.pixels_per_mm = 10.0
        self.target_diameter = 3.0
        self.target_length = 3.0
        self.tolerance = 0.5

        # ---- data -----------------------------------------------------------
        self.current_image = None
        self.current_image_path = None
        self.detected_pellets = []

        # ---- detector -------------------------------------------------------
        self.detector = YoloPelletDetector()
        self.is_trained = False

        self.init_ui()
        self.prepare_training()

    # --------------------------------------------------------------------- #
    # <<<--- NEW: training from CSV (called once at start) -------------- #
    # --------------------------------------------------------------------- #
    def prepare_training(self):
        csv_path = "labels_pellet.csv"
        img_folder = "pellet_training"

        if not Path(csv_path).exists():
            QMessageBox.warning(self, "Warning", f"{csv_path} not found!")
            return
        if not Path(img_folder).is_dir():
            QMessageBox.critical(self, "Error",
                                 f"Image folder '{img_folder}' not found!")
            return

        self.progress_label.setText("Converting CSV → YOLO format …")
        QApplication.processEvents()

        try:
            # ---- 1. Train a fresh model (you can comment this out and use a pre-trained one)
            self.detector.train(csv_path, img_folder, epochs=35, imgsz=640)

            # ---- 2. Or simply load an already trained model:
            # self.detector.load_model()

            self.is_trained = True
            self.progress_label.setText("YOLO model ready")
            self.load_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Training error", str(e))

    # --------------------------------------------------------------------- #
    # <<<--- exact measurement from mask (unchanged) ------------------- #
    # --------------------------------------------------------------------- #
    def rotated_rect_dimensions(self, polygon):
        rect = cv2.minAreaRect(polygon.astype(np.float32))
        return rect[1]                     # (w, h) in pixels

    def measure_pellet(self, polygon, pixels_per_mm):
        w_px, h_px = self.rotated_rect_dimensions(polygon)
        width_mm = w_px / pixels_per_mm
        height_mm = h_px / pixels_per_mm
        diameter = min(width_mm, height_mm)
        length = max(width_mm, height_mm)

        within = (
            (self.target_diameter - self.tolerance <= diameter <= self.target_diameter + self.tolerance) and
            (self.target_length - self.tolerance <= length <= self.target_length + self.tolerance)
        )
        return {"diameter": diameter, "length": length, "within": within}

    # --------------------------------------------------------------------- #
    # UI (exactly the same as before – only the detector changed)
    # --------------------------------------------------------------------- #
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

        # training status
        self.progress_label = QLabel("Initializing …")
        self.progress_label.setStyleSheet("padding: 5px; background-color: #e0e0e0;")
        layout.addWidget(self.progress_label)

        # load image
        self.load_btn = QPushButton("Load Pellet Image for Detection")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setStyleSheet("QPushButton { padding: 10px; font-size: 14px; }")
        self.load_btn.setEnabled(False)
        layout.addWidget(self.load_btn)

        # calibration
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

        # target specs
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

        # statistics
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

        # pellet details (scrollable)
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

    # --------------------------------------------------------------------- #
    def load_image(self):
        if not self.is_trained:
            QMessageBox.warning(self, "Not Trained", "Train the detector first!")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Pellet Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.current_image_path = path
            self.current_image = cv2.imread(path)
            if self.current_image is not None:
                self.progress_label.setText("Detecting pellets …")
                QApplication.processEvents()
                self.process_image()
                self.progress_label.setText("Detection complete")
            else:
                self.image_label.setText("Error loading image")

    def update_calibration(self, value):
        self.pixels_per_mm = value
        if self.current_image is not None:
            self.process_image()

    # --------------------------------------------------------------------- #
    def process_image(self):
        if self.current_image is None or not self.is_trained:
            return

        detections = self.detector.detect_pellets(self.current_image)
        display_img = self.current_image.copy()
        self.detected_pellets = []

        for idx, det in enumerate(detections, start=1):
            polygon = det['polygon']

            meas = self.measure_pellet(polygon, self.pixels_per_mm)

            pellet = {
                'polygon': polygon,
                'bbox': det['bbox'],
                'diameter': meas['diameter'],
                'length': meas['length'],
                'within_tolerance': meas['within'],
                'confidence': det['confidence'],
                'id': idx
            }
            self.detected_pellets.append(pellet)
            self.draw_pellet(display_img, pellet)

        self.update_statistics()
        self.display_image(display_img)

    # --------------------------------------------------------------------- #
    def draw_pellet(self, image, pellet):
        poly = pellet['polygon']
        pid = pellet['id']
        ok = pellet['within_tolerance']
        colour = (0, 255, 0) if ok else (0, 0, 255)

        overlay = image.copy()
        cv2.fillPoly(overlay, [poly.reshape(-1, 1, 2)], colour)
        cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)

        cv2.polylines(image, [poly.reshape(-1, 1, 2)], True, colour, 2)

        M = cv2.moments(poly)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else int(poly[:, 0].mean())
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else int(poly[:, 1].mean())
        cv2.putText(image, str(pid), (cx - 12, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # --------------------------------------------------------------------- #
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
                   f"  Confidence: {p['confidence']:.0f}\n"
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


# --------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = PelletMeasurementApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()