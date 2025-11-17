import sys
import os
import cv2
import numpy as np
import glob
import random
import traceback
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QDoubleSpinBox, QGroupBox, QScrollArea,
                             QMessageBox, QTextEdit, QSizePolicy, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont
from ultralytics import YOLO


# --- ROBUST PIXMAP CONVERSION ---
def cv_to_qpixmap(cv_img, target_size=None):
    if cv_img is None:
        return QPixmap()
    if len(cv_img.shape) == 3:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        format = QImage.Format.Format_RGB888
        data = rgb
    elif len(cv_img.shape) == 2:
        h, w = cv_img.shape
        bytes_per_line = w
        format = QImage.Format.Format_Grayscale8
        data = cv_img
    else:
        return QPixmap()

    contiguous = np.ascontiguousarray(data)
    qimg = QImage(contiguous.data, w, h, bytes_per_line, format)
    pixmap = QPixmap.fromImage(qimg)

    if target_size and not target_size.isNull():
        return pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
    return pixmap


def cv2_safe_imread(path):
    try:
        with open(path, 'rb') as f:
            data = f.read()
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Read error: {e}")
        return None


# --- COLOR DETECTOR (WIDE HSV) ---
class ColorPelletDetector:
    LOWER_BLUE = np.array([85, 90, 30])
    UPPER_BLUE = np.array([155, 255, 255])
    MIN_AREA = 200
    MAX_AREA = 50000

    def detect_pellets(self, image):
        if image is None: return []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWER_BLUE, self.UPPER_BLUE)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.MIN_AREA <= area <= self.MAX_AREA:
                rect = cv2.minAreaRect(c)
                w, h = rect[1]
                aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                if 1.0 <= aspect <= 6.0:
                    valid.append(c)
        return valid


# --- TRAINING THREAD ---
class YOLOTrainingThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, yaml_path):
        super().__init__()
        self.yaml_path = yaml_path

    def run(self):
        try:
            self.progress.emit("Loading YOLOv8n...")
            model = YOLO('yolov8n.pt')
            self.progress.emit("Training...")
            results = model.train(
                data=self.yaml_path, epochs=200, imgsz=1280, batch=4,
                name='pellet_detector', patience=50, save=True,
                augment=True, pretrained=True, optimizer='AdamW',
                lr0=0.01, cos_lr=True, close_mosaic=10, device='cpu',
                exist_ok=True, plots=True
            )
            path = "runs/detect/pellet_detector/weights/best.pt"
            self.finished.emit(True, path)
        except Exception as e:
            self.finished.emit(False, traceback.format_exc())


# --- MAIN APP ---
class PelletMeasurementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Inspector - Full-Res Flexible")
        self.setGeometry(100, 100, 1600, 1000)
        self.setMinimumSize(1200, 800)

        # Config
        self.pixels_per_mm = 10.0
        self.target_d = 3.0
        self.target_l = 3.0
        self.tolerance = 0.5
        self.update_ranges()

        # Data
        self.current_image = None
        self.current_image_path = None
        self.detected_pellets = []
        self.yolo_detector = None
        self.is_trained = False
        self.cv_detector = ColorPelletDetector()
        self.dataset_folder = "pellet_label_yolo"

        self.init_ui()
        self.load_existing_model()

    def update_ranges(self):
        self.d_min = self.target_d - self.tolerance
        self.d_max = self.target_d + self.tolerance
        self.l_min = self.target_l - self.tolerance
        self.l_max = self.target_l + self.tolerance

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.left_panel())
        splitter.addWidget(self.right_panel())
        splitter.setSizes([400, 1200])
        main_layout.addWidget(splitter)

    def left_panel(self):
        w = QWidget()
        l = QVBoxLayout()
        w.setLayout(l)
        font = QFont("Segoe UI", 10)
        bold = QFont("Segoe UI", 10, QFont.Weight.Bold)

        # Dataset
        g = QGroupBox("Dataset")
        g.setFont(bold)
        gl = QVBoxLayout()
        self.ds_btn = QPushButton("Select YOLO Folder")
        self.ds_btn.setFont(font)
        self.ds_btn.clicked.connect(self.select_dataset)
        gl.addWidget(self.ds_btn)
        self.ds_lbl = QLabel(f"Current: {os.path.basename(self.dataset_folder)}")
        self.ds_lbl.setFont(font)
        self.ds_lbl.setStyleSheet("background:#f8f8f8;padding:6px;")
        gl.addWidget(self.ds_lbl)
        g.setLayout(gl)
        l.addWidget(g)

        # Model
        g = QGroupBox("Model")
        g.setFont(bold)
        gl = QVBoxLayout()
        self.train_btn = QPushButton("Train Model")
        self.train_btn.setFont(font)
        self.train_btn.clicked.connect(self.start_training)
        gl.addWidget(self.train_btn)
        self.status_lbl = QLabel("Status: Loading...")
        self.status_lbl.setFont(font)
        self.status_lbl.setStyleSheet("background:#e0e0e0;padding:6px;")
        gl.addWidget(self.status_lbl)
        self.log = QTextEdit()
        self.log.setMaximumHeight(160)
        self.log.setFont(QFont("Consolas", 9))
        self.log.setReadOnly(True)
        gl.addWidget(QLabel("Log:"))
        gl.addWidget(self.log)
        g.setLayout(gl)
        l.addWidget(g)

        # Actions
        self.load_btn = QPushButton("Load Image (Any Size)")
        self.load_btn.setFont(font)
        self.load_btn.clicked.connect(self.load_image)
        l.addWidget(self.load_btn)

        self.val_btn = QPushButton("Validate Training Set")
        self.val_btn.setFont(font)
        self.val_btn.clicked.connect(self.validate_set)
        self.val_btn.setEnabled(False)
        l.addWidget(self.val_btn)

        # Calibration
        g = QGroupBox("Calibration")
        g.setFont(bold)
        gl = QVBoxLayout()
        h = QHBoxLayout()
        h.addWidget(QLabel("px/mm:"))
        self.px_spin = QDoubleSpinBox()
        self.px_spin.setRange(0.1, 500)
        self.px_spin.setValue(self.pixels_per_mm)
        self.px_spin.setDecimals(3)
        self.px_spin.setFont(font)
        self.px_spin.valueChanged.connect(self.calib_changed)
        h.addWidget(self.px_spin)
        gl.addLayout(h)
        g.setLayout(gl)
        l.addWidget(g)

        # Specs
        g = QGroupBox("Target Specs")
        g.setFont(bold)
        gl = QVBoxLayout()
        self.d_lbl = QLabel(f"Target D: {self.target_d:.2f} mm")
        self.l_lbl = QLabel(f"Target L: {self.target_l:.2f} mm")
        self.tol_lbl = QLabel(f"Tolerance: ±{self.tolerance:.2f} mm")
        self.acc_lbl = QLabel(f"Acceptable: {self.d_min:.2f}–{self.d_max:.2f} mm")
        for lbl in (self.d_lbl, self.l_lbl, self.tol_lbl, self.acc_lbl):
            lbl.setFont(font)
            gl.addWidget(lbl)
        g.setLayout(gl)
        l.addWidget(g)

        # Stats
        g = QGroupBox("Statistics")
        g.setFont(bold)
        gl = QVBoxLayout()
        self.total_lbl = QLabel("Total: 0")
        self.ok_lbl = QLabel("OK: 0")
        self.bad_lbl = QLabel("BAD: 0")
        self.sts_lbl = QLabel("Status: —")
        self.sts_lbl.setFont(bold)
        for lbl in (self.total_lbl, self.ok_lbl, self.bad_lbl, self.sts_lbl):
            lbl.setFont(font)
            gl.addWidget(lbl)
        g.setLayout(gl)
        l.addWidget(g)

        # Details
        g = QGroupBox("Pellet Details")
        g.setFont(bold)
        gl = QVBoxLayout()
        self.scroll_det = QScrollArea()
        self.scroll_det.setWidgetResizable(True)
        self.det_w = QWidget()
        self.det_l = QVBoxLayout()
        self.det_w.setLayout(self.det_l)
        self.scroll_det.setWidget(self.det_w)
        gl.addWidget(self.scroll_det)
        g.setLayout(gl)
        l.addWidget(g)

        l.addStretch()
        return w

    def right_panel(self):
        w = QWidget()
        l = QVBoxLayout()
        w.setLayout(l)

        self.img_scroll = QScrollArea()
        self.img_scroll.setWidgetResizable(True)
        self.img_lbl = QLabel()
        self.img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_lbl.setStyleSheet("background:#ffffff;")
        self.img_lbl.setMinimumSize(600, 600)
        self.img_scroll.setWidget(self.img_lbl)
        l.addWidget(self.img_scroll)

        return w

    # --- MODEL LOAD ---
    def load_existing_model(self):
        path = "runs/detect/pellet_detector5/weights/best.pt"
        self.log.append(f"Checking: {path}")
        if os.path.exists(path):
            try:
                self.yolo_detector = YOLO(path)
                self.is_trained = True
                self.status_lbl.setText("YOLO Model Loaded")
                self.val_btn.setEnabled(True)
                self.train_btn.setEnabled(False)
                self.train_btn.setText("Model Active")
                self.train_btn.setStyleSheet("background:#2e7d32;color:white;")
                self.log.append("YOLO ready for any image size")
            except Exception as e:
                self.log.append(f"Load error: {e}")
                self.status_lbl.setText("YOLO failed → CV mode")
        else:
            self.log.append("No model found")
            self.status_lbl.setText("Using Color-CV")

    # --- ACTIONS ---
    def select_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select YOLO Dataset")
        if folder and os.path.exists(os.path.join(folder, "dataset.yaml")):
            self.dataset_folder = folder
            self.ds_lbl.setText(f"Current: {os.path.basename(folder)}")

    def start_training(self):
        yaml = os.path.join(self.dataset_folder, "dataset.yaml")
        if not os.path.exists(yaml):
            QMessageBox.critical(self, "Error", "No dataset.yaml")
            return
        self.thread = YOLOTrainingThread(yaml)
        self.thread.progress.connect(lambda m: self.log.append(m))
        self.thread.finished.connect(self.train_done)
        self.thread.start()

    def train_done(self, ok, msg):
        if ok:
            self.yolo_detector = YOLO(msg)
            self.is_trained = True
            self.status_lbl.setText("Training complete")

    def validate_set(self):
        if not self.is_trained: return
        imgs = glob.glob(os.path.join(self.dataset_folder, "images", "train", "*.*"))
        imgs = [f for f in imgs if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        detected = sum(1 for f in imgs if len(self.yolo_detector(f, conf=0.05, imgsz=1280)[0].boxes) > 0)
        QMessageBox.information(self, "Validation", f"Detected: {detected}/{len(imgs)}")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path: return
        img = cv2_safe_imread(path)
        if img is None:
            QMessageBox.critical(self, "Error", "Cannot read image")
            return
        self.current_image = img
        self.current_image_path = path
        h, w = img.shape[:2]
        self.log.append(f"Loaded: {w}×{h} px")
        self.process_image()

    def calib_changed(self, v):
        self.pixels_per_mm = v
        self.update_ranges()
        self.acc_lbl.setText(f"Acceptable: {self.d_min:.2f}–{self.d_max:.2f} mm")
        if self.current_image is not None:
            self.process_image()

    # --- DETECTION (FULL RES) ---
    def process_image(self):
        if self.current_image is None: return
        disp = self.current_image.copy()
        self.detected_pellets = []
        polygons = []
        confidences = []

        if self.is_trained and self.yolo_detector:
            try:
                results = self.yolo_detector.predict(
                    source=self.current_image,
                    conf=0.05,
                    iou=0.45,
                    imgsz=1280,        # High-res input
                    max_det=300,       # Allow many pellets
                    device='cpu',
                    verbose=False
                )
                r = results[0]
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf.item()
                    # YOLO debug box
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(disp, f"YOLO {conf:.2f}", (x1, y1-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                    roi = self.current_image[y1:y2, x1:x2]
                    if roi.size == 0: continue
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, self.cv_detector.LOWER_BLUE, self.cv_detector.UPPER_BLUE)
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not cnts: continue
                    c = max(cnts, key=cv2.contourArea)
                    area = cv2.contourArea(c)
                    if area < self.cv_detector.MIN_AREA: continue
                    c += np.array([x1, y1])
                    polygons.append(c)
                    confidences.append(conf * 100)
            except Exception as e:
                self.log.append(f"YOLO error: {e}")
                polygons = self.cv_detector.detect_pellets(self.current_image)
                confidences = [100.0] * len(polygons)
        else:
            polygons = self.cv_detector.detect_pellets(self.current_image)
            confidences = [100.0] * len(polygons)

        # Measure & draw
        for i, (poly, conf) in enumerate(zip(polygons, confidences), 1):
            meas = self.measure(poly, self.pixels_per_mm)
            pellet = {
                'polygon': poly, 'diameter': meas['d'], 'length': meas['l'],
                'within': meas['ok'], 'confidence': conf, 'id': i
            }
            self.detected_pellets.append(pellet)
            self.draw_pellet(disp, pellet)

        self.update_stats()
        self.show_image(disp)
        self.status_lbl.setText(f"Detected: {len(self.detected_pellets)} pellets")

    def measure(self, poly, ppm):
        rect = cv2.minAreaRect(poly.astype(np.float32))
        w, h = rect[1]
        d = min(w, h) / ppm
        l = max(w, h) / ppm
        ok = (self.d_min <= d <= self.d_max) and (self.l_min <= l <= self.l_max)
        return {'d': d, 'l': l, 'ok': ok}

    def draw_pellet(self, img, p):
        color = (0, 255, 0) if p['within'] else (0, 0, 255)
        rect = cv2.minAreaRect(p['polygon'].astype(np.float32))
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        overlay = img.copy()
        cv2.fillPoly(overlay, [p['polygon'].reshape(-1,1,2)], color)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        cv2.drawContours(img, [box], 0, color, 4)

        M = cv2.moments(p['polygon'])
        cx = int(M["m10"]/M["m00"]) if M["m00"] else int(p['polygon'][:,0].mean())
        cy = int(M["m01"]/M["m00"]) if M["m00"] else int(p['polygon'][:,1].mean())
        cv2.putText(img, str(p['id']), (cx-20, cy+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

    def update_stats(self):
        total = len(self.detected_pellets)
        ok = sum(1 for p in self.detected_pellets if p['within'])
        bad = total - ok
        self.total_lbl.setText(f"Total: {total}")
        self.ok_lbl.setText(f"OK: {ok}")
        self.bad_lbl.setText(f"BAD: {bad}")
        if total == 0:
            self.sts_lbl.setText("No pellets")
            self.sts_lbl.setStyleSheet("color:gray;")
        elif bad == 0:
            self.sts_lbl.setText("All OK")
            self.sts_lbl.setStyleSheet("color:green;")
        else:
            self.sts_lbl.setText(f"{bad} Out of Spec")
            self.sts_lbl.setStyleSheet("color:red;")
        self.update_details()

    def update_details(self):
        for i in reversed(range(self.det_l.count())):
            w = self.det_l.itemAt(i).widget()
            if w: w.deleteLater()
        for i, p in enumerate(self.detected_pellets, 1):
            txt = (f"<b>Pellet {i}</b> → {'<font color=green>OK</font>' if p['within'] else '<font color=red>BAD</font>'}<br>"
                   f"D: <b>{p['diameter']:.3f}</b> mm<br>"
                   f"L: <b>{p['length']:.3f}</b> mm<br>"
                   f"Conf: <b>{p['confidence']:.1f}</b>%")
            lbl = QLabel(txt)
            lbl.setFont(QFont("Segoe UI", 10))
            lbl.setStyleSheet(f"padding:8px;margin:3px;border:2px solid {'green' if p['within'] else 'red'};border-radius:6px;")
            self.det_l.addWidget(lbl)
        self.det_l.addStretch()

    def show_image(self, cv_img):
        h, w = cv_img.shape[:2]
        self.log.append(f"Displaying: {w}×{h} px (scroll to zoom)")
        pixmap = cv_to_qpixmap(cv_img)
        self.img_lbl.setPixmap(pixmap)
        self.img_lbl.resize(pixmap.size())
        self.img_scroll.verticalScrollBar().setValue(0)
        self.img_scroll.horizontalScrollBar().setValue(0)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont("Segoe UI", 10))
    win = PelletMeasurementApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()