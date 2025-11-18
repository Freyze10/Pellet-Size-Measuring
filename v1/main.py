from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QGroupBox, QScrollArea,
                             QDoubleSpinBox, QMessageBox, QFrame, QSpacerItem, QSizePolicy)
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon

# --- Helper functions ---
def cv_to_qpixmap(cv_img, target_size=None):
    if cv_img is None:
        return QPixmap()
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    if target_size:
        pixmap = pixmap.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
    return pixmap

def cv2_safe_imread(path):
    try:
        with open(path, 'rb') as f:
            data = f.read()
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except:
        return None

# --- Color fallback detector ---
class ColorPelletDetector:
    LOWER_BLUE = np.array([90, 100, 30])
    UPPER_BLUE = np.array([150, 255, 255])
    MIN_PIXEL_AREA = 300
    MAX_PIXEL_AREA = 20000

    def detect_pellets(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWER_BLUE, self.UPPER_BLUE)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.MIN_PIXEL_AREA <= area <= self.MAX_PIXEL_AREA:
                w, h = cv2.minAreaRect(c)[1]
                if min(w, h) > 0 and 1.0 <= max(w, h) / min(w, h) <= 5.0:
                    valid.append(c)
        return valid

# --- Main Application ---
class PelletMeasurementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Inspector Pro")
        self.setGeometry(100, 100, 1200, 750)
        self.setStyleSheet("""
            QMainWindow { background-color: #f8f9fc; }
            QLabel { color: #2d3748; font-size: 13px; }
            QPushButton {
                background-color: #3b82f6; color: white; border: none; padding: 10px 16px;
                border-radius: 8px; font-weight: bold; font-size: 14px;
            }
            QPushButton:hover { background-color: #2563eb; }
            QPushButton:pressed { background-color: #1d4ed8; }
            QGroupBox {
                font-weight: bold; font-size: 14px; color: #1e40af;
                border: 2px solid #e2e8f0; border-radius: 10px; margin-top: 10px; padding-top: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 10px; }
            QDoubleSpinBox { padding: 6px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px; }
            QScrollArea { border: none; background: transparent; }
        """)

        self.pixels_per_mm = 25.4
        self.target_diameter = 3.0
        self.target_length = 3.0
        self.tolerance = 0.5
        self.update_ranges()

        self.current_image = None
        self.detected_pellets = []
        self.yolo_detector = None
        self.cv_detector = ColorPelletDetector()

        self.init_ui()
        self.load_model("trained_model/best.pt")

    def update_ranges(self):
        self.d_min = self.target_diameter - self.tolerance
        self.d_max = self.target_diameter + self.tolerance
        self.l_min = self.target_length - self.tolerance
        self.l_max = self.target_length + self.tolerance

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Left Panel - Controls
        left_panel = self.create_left_panel()
        left_panel.setMaximumWidth(380)
        left_panel.setStyleSheet("background: white; border-radius: 12px; border: 1px solid #e2e8f0;")
        main_layout.addWidget(left_panel)

        # Right Panel - Image Display
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)

    def create_left_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Pellet Inspector Pro")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #1e40af;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Load Image Button
        self.load_btn = QPushButton("Load Image")
        self.load_btn.setStyleSheet("background-color: #10b981; padding: 14px; font-size: 15px;")
        self.load_btn.clicked.connect(self.load_image)
        layout.addWidget(self.load_btn)

        # Calibration
        calib_group = QGroupBox("Calibration & Target Size")
        calib_layout = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel("Pixels per mm:"))
        self.px_spin = QDoubleSpinBox()
        self.px_spin.setValue(self.pixels_per_mm)
        self.px_spin.setSingleStep(0.5)
        self.px_spin.valueChanged.connect(lambda v: setattr(self, 'pixels_per_mm', v))
        row.addWidget(self.px_spin)
        calib_layout.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Target Ø:"))
        self.diam_spin = QDoubleSpinBox()
        self.diam_spin.setValue(3.0)
        self.diam_spin.setSingleStep(0.1)
        self.diam_spin.valueChanged.connect(lambda v: setattr(self, 'target_diameter', v) or self.update_ranges())
        row2.addWidget(self.diam_spin)
        row2.addWidget(QLabel("mm"))

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Target L:"))
        self.len_spin = QDoubleSpinBox()
        self.len_spin.setValue(3.0)
        self.len_spin.setSingleStep(0.1)
        self.len_spin.valueChanged.connect(lambda v: setattr(self, 'target_length', v) or self.update_ranges())
        row3.addWidget(self.len_spin)
        row3.addWidget(QLabel("mm"))

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Tolerance: ±"))
        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setValue(0.5)
        self.tol_spin.setSingleStep(0.1)
        self.tol_spin.valueChanged.connect(lambda v: setattr(self, 'tolerance', v) or self.update_ranges())
        row4.addWidget(self.tol_spin)
        row4.addWidget(QLabel("mm"))

        calib_layout.addLayout(row2)
        calib_layout.addLayout(row3)
        calib_layout.addLayout(row4)
        calib_group.setLayout(calib_layout)
        layout.addWidget(calib_group)

        # Statistics
        stats_group = QGroupBox("Inspection Results")
        stats_layout = QVBoxLayout()
        self.total_lbl = QLabel("Total Pellets: —")
        self.ok_lbl = QLabel("OK: —")
        self.bad_lbl = QLabel("Defective: —")
        for lbl in [self.total_lbl, self.ok_lbl, self.bad_lbl]:
            lbl.setStyleSheet("font-size: 15px; padding: 4px;")
        self.ok_lbl.setStyleSheet(self.ok_lbl.styleSheet() + "color: #16a34a; font-weight: bold;")
        self.bad_lbl.setStyleSheet(self.bad_lbl.styleSheet() + "color: #dc2626; font-weight: bold;")
        stats_layout.addWidget(self.total_lbl)
        stats_layout.addWidget(self.ok_lbl)
        stats_layout.addWidget(self.bad_lbl)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Pellet Details
        details_group = QGroupBox("Detected Pellets")
        details_layout = QVBoxLayout()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setMaximumHeight(300)
        self.detail_container = QWidget()
        self.detail_layout = QVBoxLayout(self.detail_container)
        self.detail_layout.addStretch()
        self.scroll.setWidget(self.detail_container)
        details_layout.addWidget(self.scroll)
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        layout.addStretch()
        return widget

    def create_right_panel(self):
        widget = QWidget()
        widget.setStyleSheet("background: white; border-radius: 12px; border: 1px solid #e2e8f0;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(15, 15, 15, 15)

        self.img_label = QLabel("Click 'Load Image' to begin inspection")
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet("""
            color: #64748b; font-size: 16px; background: #f1f5f9; border: 2px dashed #cbd5e1;
            border-radius: 12px; min-height: 500px;
        """)
        self.img_label.setMinimumSize(600, 500)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.img_label)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        layout.addWidget(scroll)
        return widget

    def load_model(self, path):
        if os.path.exists(path):
            try:
                self.yolo_detector = YOLO(path)
                print(f"Model loaded: {path}")
            except Exception as e:
                print("Model load failed:", e)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        img = cv2_safe_imread(path)
        if img is None:
            QMessageBox.critical(self, "Error", "Could not load image.")
            return
        self.current_image = img
        self.process_image()

    def process_image(self):
        if self.current_image is None:
            return
        img_disp = self.current_image.copy()
        self.detected_pellets = []

        polygons, confidences = [], []
        try:
            if self.yolo_detector:
                results = self.yolo_detector.predict(self.current_image, conf=0.05, imgsz=640, device='cpu', verbose=False)
                r = results[0]
                temp = []
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf.item() * 100
                    roi = self.current_image[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    cnts = self.cv_detector.detect_pellets(roi)
                    if not cnts:
                        continue
                    c = max(cnts, key=cv2.contourArea) + np.array([x1, y1])
                    M = cv2.moments(c)
                    cx = int(M["m10"] / M["m00"]) if M["m00"] else int(c[:,0].mean())
                    cy = int(M["m01"] / M["m00"]) if M["m00"] else int(c[:,1].mean())
                    temp.append({'poly': c, 'conf': conf, 'center': (cx, cy)})

                # Remove duplicates
                filtered = []
                for p in sorted(temp, key=lambda x: -x['conf']):
                    if not any(np.hypot(p['center'][0]-f['center'][0], p['center'][1]-f['center'][1]) < 12 for f in filtered):
                        filtered.append(p)
                polygons = [p['poly'] for p in filtered]
                confidences = [p['conf'] for p in filtered]
            else:
                polygons = self.cv_detector.detect_pellets(self.current_image)
                confidences = [100] * len(polygons)
        except:
            polygons = self.cv_detector.detect_pellets(self.current_image)
            confidences = [100] * len(polygons)

        # Measure and draw
        for i, (poly, conf) in enumerate(zip(polygons, confidences), 1):
            rect = cv2.minAreaRect(poly.astype(np.float32))
            w, h = rect[1]
            diameter = min(w, h) / self.pixels_per_mm
            length = max(w, h) / self.pixels_per_mm
            ok = self.d_min <= diameter <= self.d_max and self.l_min <= length <= self.l_max

            pellet = {
                'polygon': poly, 'diameter': diameter, 'length': length,
                'within': ok, 'confidence': conf, 'id': i
            }
            self.detected_pellets.append(pellet)
            self.draw_pellet(img_disp, pellet)

        self.update_stats()
        self.display_image(img_disp)

    def draw_pellet(self, img, p):
        color = (0, 200, 0) if p['within'] else (0, 0, 220)
        overlay = img.copy()
        cv2.fillPoly(overlay, [p['polygon'].reshape(-1, 1, 2)], color)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        box = np.intp(cv2.boxPoints(cv2.minAreaRect(p['polygon'].astype(np.float32))))
        cv2.drawContours(img, [box], 0, color, 3)

        M = cv2.moments(p['polygon'])
        cx = int(M["m10"]/M["m00"]) if M["m00"] else int(p['polygon'][:,0].mean())
        cy = int(M["m01"]/M["m00"]) if M["m00"] else int(p['polygon'][:,1].mean())

        cv2.putText(img, str(p['id']), (cx-15, cy+10), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 3)
        cv2.putText(img, str(p['id']), (cx-15, cy+10), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,0), 2)

        if p['confidence'] < 100:
            cv2.putText(img, f"{p['confidence']:.0f}%", (cx-35, cy-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    def update_stats(self):
        total = len(self.detected_pellets)
        ok = sum(1 for p in self.detected_pellets if p['within'])
        bad = total - ok
        self.total_lbl.setText(f"Total Pellets: {total}")
        self.ok_lbl.setText(f"OK: {ok}")
        self.bad_lbl.setText(f"Defective: {bad}")

        # Clear and update detail list
        for i in reversed(range(self.detail_layout.count())):
            child = self.detail_layout.itemAt(i).widget()
            if child: child.deleteLater()

        for p in self.detected_pellets:
            status = "OK" if p['within'] else "DEFECT"
            color = "#16a34a" if p['within'] else "#dc2626"
            text = f"Pellet {p['id']:2d} → {status} | Ø {p['diameter']:.3f} mm | L {p['length']:.3f} mm"
            lbl = QLabel(text)
            lbl.setStyleSheet(f"""
                background: {color}15; color: {color}; padding: 10px; border-radius: 8px;
                border-left: 5px solid {color}; font-family: Segoe UI; font-size: 13px;
            """)
            self.detail_layout.insertWidget(self.detail_layout.count()-1, lbl)

    def display_image(self, cv_img):
        if cv_img is None:
            return
        h, w = cv_img.shape[:2]
        scale = min(800 / w, 600 / h, 1.0)
        target = QSize(int(w * scale), int(h * scale))
        pixmap = cv_to_qpixmap(cv_img, target)
        self.img_label.setPixmap(pixmap)
        self.img_label.setStyleSheet("background: white; border-radius: 12px;")

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = PelletMeasurementApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()