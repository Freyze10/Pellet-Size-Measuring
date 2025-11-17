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
                             QMessageBox, QTextEdit, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage
from ultralytics import YOLO


# --- ROBUST IMAGE CONVERSION (CRASH FIX) ---
def cv_to_qpixmap(cv_img, target_size=None):
    if cv_img is None:
        return QPixmap()
    if len(cv_img.shape) == 3:
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        format = QImage.Format.Format_RGB888
        data_to_pass = rgb_image
    elif len(cv_img.shape) == 2:
        h, w = cv_img.shape
        bytes_per_line = w
        format = QImage.Format.Format_Grayscale8
        data_to_pass = cv_img
    else:
        return QPixmap()
    contiguous_data = data_to_pass.copy(order='C')
    qimg = QImage(contiguous_data.data, w, h, bytes_per_line, format)
    qpixmap = QPixmap.fromImage(qimg)
    if target_size and not target_size.isNull():
        return qpixmap.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
    return qpixmap


def cv2_safe_imread(path):
    try:
        with open(path, 'rb') as f:
            data = f.read()
        np_arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Read error: {e}")
        return None


# --- COLOR DETECTOR (WIDER RANGE) ---
class ColorPelletDetector:
    LOWER_BLUE = np.array([90, 100, 30])    # Widened for lighting variation
    UPPER_BLUE = np.array([150, 255, 255])
    MIN_PIXEL_AREA = 300
    MAX_PIXEL_AREA = 20000

    def detect_pellets(self, image):
        if image is None: return []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWER_BLUE, self.UPPER_BLUE)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.MIN_PIXEL_AREA <= area <= self.MAX_PIXEL_AREA:
                rect = cv2.minAreaRect(c)
                w, h = rect[1]
                aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                if 1.0 <= aspect <= 5.0:
                    valid.append(c)
        return valid


# --- TRAINING THREAD (OPTIONAL) ---
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
            self.progress.emit("Training 200 epochs...")
            results = model.train(
                data=self.yaml_path, epochs=200, imgsz=640, batch=4,
                name='pellet_detector', patience=50, save=True,
                augment=True, pretrained=True, optimizer='AdamW',
                lr0=0.01, cos_lr=True, close_mosaic=10, device='cpu',
                exist_ok=True, plots=True
            )
            path = "runs/detect/pellet_detector/weights/best.pt"
            self.progress.emit("Done!")
            self.finished.emit(True, path)
        except Exception as e:
            self.finished.emit(False, traceback.format_exc())


# --- MAIN APP ---
class PelletMeasurementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Inspector - YOLO + CV (Debug Mode)")
        self.setGeometry(100, 100, 1450, 900)

        self.pixels_per_mm = 10.0
        self.target_diameter = 3.0
        self.target_length = 3.0
        self.tolerance = 0.5
        self.update_ranges()

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
        self.d_min = self.target_diameter - self.tolerance
        self.d_max = self.target_diameter + self.tolerance
        self.l_min = self.target_length - self.tolerance
        self.l_max = self.target_length + self.tolerance

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout()
        central.setLayout(layout)

        layout.addWidget(self.left_panel(), 1)
        layout.addWidget(self.right_panel(), 3)

    def left_panel(self):
        w = QWidget()
        l = QVBoxLayout()
        w.setLayout(l)

        # Dataset
        g = QGroupBox("YOLO Dataset")
        gl = QVBoxLayout()
        self.ds_btn = QPushButton("Select Folder")
        self.ds_btn.clicked.connect(self.select_dataset)
        gl.addWidget(self.ds_btn)
        self.ds_lbl = QLabel(f"Current: {os.path.basename(self.dataset_folder)}")
        self.ds_lbl.setStyleSheet("background:#f8f8f8;padding:5px;")
        gl.addWidget(self.ds_lbl)
        g.setLayout(gl)
        l.addWidget(g)

        # Train
        g = QGroupBox("Model")
        gl = QVBoxLayout()
        self.train_btn = QPushButton("Train New Model")
        self.train_btn.clicked.connect(self.start_training)
        gl.addWidget(self.train_btn)
        self.status_lbl = QLabel("Status: Initializing...")
        self.status_lbl.setStyleSheet("background:#e0e0e0;padding:5px;")
        gl.addWidget(self.status_lbl)
        self.log = QTextEdit()
        self.log.setMaximumHeight(140)
        self.log.setReadOnly(True)
        gl.addWidget(QLabel("Log:"))
        gl.addWidget(self.log)
        g.setLayout(gl)
        l.addWidget(g)

        # Actions
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        l.addWidget(self.load_btn)

        self.val_btn = QPushButton("Validate Training Set")
        self.val_btn.clicked.connect(self.validate_set)
        self.val_btn.setEnabled(False)
        l.addWidget(self.val_btn)

        self.debug_btn = QPushButton("Debug Random Train Img")
        self.debug_btn.clicked.connect(self.debug_train_img)
        self.debug_btn.setEnabled(False)
        l.addWidget(self.debug_btn)

        # Calibration
        g = QGroupBox("Calibration")
        gl = QVBoxLayout()
        h = QHBoxLayout()
        h.addWidget(QLabel("px/mm:"))
        self.px_spin = QDoubleSpinBox()
        self.px_spin.setRange(0.1, 200)
        self.px_spin.setValue(self.pixels_per_mm)
        self.px_spin.setDecimals(2)
        self.px_spin.valueChanged.connect(self.calib_changed)
        h.addWidget(self.px_spin)
        gl.addLayout(h)
        g.setLayout(gl)
        l.addWidget(g)

        # Specs
        g = QGroupBox("Target Specs")
        gl = QVBoxLayout()
        self.d_lbl = QLabel(f"Target D: {self.target_diameter:.2f} mm")
        self.l_lbl = QLabel(f"Target L: {self.target_length:.2f} mm")
        self.tol_lbl = QLabel(f"Tolerance: ±{self.tolerance:.2f} mm")
        self.acc_lbl = QLabel(f"Acceptable: {self.d_min:.2f}–{self.d_max:.2f} mm")
        for lbl in (self.d_lbl, self.l_lbl, self.tol_lbl, self.acc_lbl):
            gl.addWidget(lbl)
        g.setLayout(gl)
        l.addWidget(g)

        # Stats
        g = QGroupBox("Stats")
        gl = QVBoxLayout()
        self.total_lbl = QLabel("Total: 0")
        self.ok_lbl = QLabel("OK: 0")
        self.bad_lbl = QLabel("BAD: 0")
        self.sts_lbl = QLabel("Status: —")
        self.sts_lbl.setStyleSheet("font-weight:bold;")
        for lbl in (self.total_lbl, self.ok_lbl, self.bad_lbl, self.sts_lbl):
            gl.addWidget(lbl)
        g.setLayout(gl)
        l.addWidget(g)

        # Details
        g = QGroupBox("Pellet Details")
        gl = QVBoxLayout()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.det_w = QWidget()
        self.det_l = QVBoxLayout()
        self.det_w.setLayout(self.det_l)
        self.scroll.setWidget(self.det_w)
        gl.addWidget(self.scroll)
        g.setLayout(gl)
        l.addWidget(g)

        l.addStretch()
        return w

    def right_panel(self):
        w = QWidget()
        l = QVBoxLayout()
        w.setLayout(l)

        # ---- QLabel that will always show the FULL WIDTH of the image ----
        self.img_lbl = QLabel("Load image...")
        self.img_lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.img_lbl.setStyleSheet("border:2px solid #ccc;background:#f0f0f0;")
        # start with a reasonable minimum width – it will be resized later
        self.img_lbl.setMinimumWidth(800)
        self.img_lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        # ---- Scroll area that only scrolls vertically --------------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)  # we control the size ourselves
        scroll.setWidget(self.img_lbl)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        l.addWidget(scroll)
        return w

    # --- MODEL LOAD ---
    def load_existing_model(self):
        path = "runs/detect/pellet_detector5/weights/best.pt"
        self.log.append(f"Checking model: {path}")
        if os.path.exists(path):
            try:
                self.yolo_detector = YOLO(path)
                self.is_trained = True
                self.status_lbl.setText("YOLO model loaded")
                self.val_btn.setEnabled(True)
                self.debug_btn.setEnabled(True)
                self.train_btn.setEnabled(False)
                self.train_btn.setText("Model Loaded")
                self.train_btn.setStyleSheet("background:#777;color:#fff;")
                self.log.append("SUCCESS: YOLO ready")
                QMessageBox.information(self, "Model Loaded", f"YOLO active:\n{path}")
            except Exception as e:
                self.log.append(f"LOAD FAIL: {e}")
                self.status_lbl.setText("YOLO failed → CV mode")
        else:
            self.log.append("NO MODEL FOUND")
            self.status_lbl.setText("No model → using CV")

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
        self.train_btn.setEnabled(False)

    def train_done(self, ok, msg):
        self.train_btn.setEnabled(True)
        if ok:
            self.yolo_detector = YOLO(msg)
            self.is_trained = True
            self.status_lbl.setText("Training done!")
        else:
            self.status_lbl.setText("Training failed")

    def validate_set(self):
        if not self.is_trained: return
        imgs = glob.glob(os.path.join(self.dataset_folder, "images", "train", "*.*"))
        imgs = [f for f in imgs if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        detected = sum(1 for f in imgs if len(self.yolo_detector(f, conf=0.05)[0].boxes) > 0)
        QMessageBox.information(self, "Validation", f"Detected: {detected}/{len(imgs)}")

    def debug_train_img(self):
        if not self.is_trained: return
        imgs = glob.glob(os.path.join(self.dataset_folder, "images", "train", "*.*"))
        imgs = [f for f in imgs if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if not imgs: return
        path = random.choice(imgs)
        self.current_image = cv2_safe_imread(path)
        self.current_image_path = path
        self.process_image()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path: return
        self.current_image = cv2_safe_imread(path)
        if self.current_image is None:
            QMessageBox.critical(self, "Error", "Failed to load image")
            return
        self.current_image_path = path
        self.process_image()

    def calib_changed(self, v):
        self.pixels_per_mm = v
        self.update_ranges()
        for lbl, val in [(self.d_lbl, self.target_diameter),
                         (self.l_lbl, self.target_length),
                         (self.tol_lbl, self.tolerance),
                         (self.acc_lbl, f"{self.d_min:.2f}–{self.d_max:.2f}")]:
            if "Target D" in lbl.text():
                lbl.setText(f"Target D: {val:.2f} mm")
            elif "Target L" in lbl.text():
                lbl.setText(f"Target L: {val:.2f} mm")
            elif "Tolerance" in lbl.text():
                lbl.setText(f"Tolerance: ±{val:.2f} mm")
            else:
                lbl.setText(f"Acceptable: {self.d_min:.2f}–{self.d_max:.2f} mm")
        if self.current_image is not None:
            self.process_image()

    # --- DETECTION ---
    def process_image(self):
        if self.current_image is None: return
        disp = self.current_image.copy()
        self.detected_pellets = []
        polygons = []
        confidences = []

        # === YOLO (with debug boxes) ===
        if self.is_trained and self.yolo_detector:
            try:
                results = self.yolo_detector.predict(
                    source=self.current_image,
                    conf=0.05,      # LOW THRESHOLD
                    iou=0.45,
                    imgsz=640,
                    device='cpu',
                    verbose=False
                )
                r = results[0]
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf.item()
                    # Draw YOLO box in YELLOW for debugging
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(disp, f"YOLO {conf:.2f}", (x1, max(y1-10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    roi = self.current_image[y1:y2, x1:x2]
                    if roi.size == 0: continue
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, self.cv_detector.LOWER_BLUE, self.cv_detector.UPPER_BLUE)
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not cnts: continue
                    c = max(cnts, key=cv2.contourArea)
                    area = cv2.contourArea(c)
                    if area < self.cv_detector.MIN_PIXEL_AREA: continue
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

        # === MEASURE & DRAW FINAL ===
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
        self.status_lbl.setText(f"Detected: {len(self.detected_pellets)}")

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

        # semi-transparent fill
        overlay = img.copy()
        cv2.fillPoly(overlay, [p['polygon'].reshape(-1,1,2)], color)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

        # thicker box
        cv2.drawContours(img, [box], 0, color, 3)

        # ----- LARGER ID TEXT -----
        M = cv2.moments(p['polygon'])
        cx = int(M["m10"]/M["m00"]) if M["m00"] else int(p['polygon'][:,0].mean())
        cy = int(M["m01"]/M["m00"]) if M["m00"] else int(p['polygon'][:,1].mean())

        font_scale = 1.1
        thickness = 3
        cv2.putText(img, str(p['id']),
                    (cx-20, cy+12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255,255,255), thickness)

        # ----- OPTIONAL: CONFIDENCE (bigger) -----
        if p['confidence'] < 100.0:
            conf_text = f"{p['confidence']:.0f}%"
            cv2.putText(img, conf_text,
                        (cx-30, cy-20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,255,255), 2)

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
            self.sts_lbl.setText(f"{bad} BAD")
            self.sts_lbl.setStyleSheet("color:red;")
        self.update_details()

    def update_details(self):
        for i in reversed(range(self.det_l.count())):
            w = self.det_l.itemAt(i).widget()
            if w: w.deleteLater()
        for i, p in enumerate(self.detected_pellets, 1):
            txt = (f"Pellet {i} - {'OK' if p['within'] else 'BAD'}\n"
                   f"  D: {p['diameter']:.3f} mm\n"
                   f"  L: {p['length']:.3f} mm\n"
                   f"  Conf: {p['confidence']:.1f}%")
            lbl = QLabel(txt)
            lbl.setStyleSheet(f"padding:5px;margin:2px;border:1px solid {'green' if p['within'] else 'red'};")
            self.det_l.addWidget(lbl)
        self.det_l.addStretch()

    def show_image(self, cv_img):
        target = self.img_lbl.size()
        pix = cv_to_qpixmap(cv_img, target)
        self.img_lbl.setPixmap(pix)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = PelletMeasurementApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()