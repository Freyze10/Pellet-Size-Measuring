from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QGroupBox, QScrollArea,
                             QDoubleSpinBox, QMessageBox, QFrame, QSpacerItem, QSizePolicy)
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont, QPalette, QColor

# --- Helper functions ---
def cv_to_qpixmap(cv_img, target_size=None):
    if cv_img is None:
        return QPixmap()
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_line, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    if target_size:
        pix = pix.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    return pix

def cv2_safe_imread(path):
    try:
        data = open(path, 'rb').read()
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
            a = cv2.contourArea(c)
            if self.MIN_PIXEL_AREA <= a <= self.MAX_PIXEL_AREA:
                w, h = cv2.minAreaRect(c)[1]
                if min(w,h) > 0 and 1.0 <= max(w,h)/min(w,h) <= 5.0:
                    valid.append(c)
        return valid

# --- Main App with Modern Professional UI ---
class PelletMeasurementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Inspector Pro")
        self.setGeometry(100, 100, 1200, 800)  # Compact professional size
        self.setStyleSheet("""
            QMainWindow { background-color: #f8f9fc; }
            QLabel { color: #2d3748; font-size: 12px; }
            QGroupBox {
                font-weight: bold; 
                border: 1px solid #e2e8f0; 
                border-radius: 8px; 
                margin: 10px; 
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                background-color: white;
                color: #4a5568;
            }
            QPushButton {
                background-color: #4299e1; 
                color: white; 
                border: none; 
                padding: 10px 20px; 
                border-radius: 6px; 
                font-weight: bold;
            }
            QPushButton:hover { background-color: #3182ce; }
            QPushButton:pressed { background-color: #2b6cb0; }
            QDoubleSpinBox {
                padding: 6px; 
                border: 1px solid #cbd5e0; 
                border-radius: 6px;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
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
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # Left Panel
        left = self.create_left_panel()
        left.setMaximumWidth(380)
        left.setStyleSheet("background-color: white; border-radius: 10px; border: 1px solid #e2e8f0;")
        main_layout.addWidget(left, 1)

        # Right Panel - Image Display
        right = self.create_right_panel()
        main_layout.addWidget(right, 3)

    def create_left_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Pellet Inspector Pro")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2d3748; border: none;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Load Button
        self.load_btn = QPushButton("Load Image")
        self.load_btn.setMinimumHeight(45)
        self.load_btn.clicked.connect(self.load_image)
        layout.addWidget(self.load_btn)

        # Calibration
        calib_box = QGroupBox("Calibration")
        calib_layout = QVBoxLayout()
        h = QHBoxLayout()
        h.addWidget(QLabel("Pixels per mm:"))
        self.px_spin = QDoubleSpinBox()
        self.px_spin.setRange(1.0, 200.0)
        self.px_spin.setSingleStep(0.5)
        self.px_spin.setValue(self.pixels_per_mm)
        self.px_spin.setDecimals(2)
        self.px_spin.valueChanged.connect(lambda v: setattr(self, 'pixels_per_mm', v))
        h.addWidget(self.px_spin)
        calib_layout.addLayout(h)
        calib_box.setLayout(calib_layout)
        layout.addWidget(calib_box)

        # Stats
        stats_box = QGroupBox("Detection Summary")
        stats_layout = QVBoxLayout()
        self.total_lbl = QLabel("Total Pellets: 0")
        self.ok_lbl = QLabel("OK: 0")
        self.bad_lbl = QLabel("Out of Spec: 0")
        for lbl in [self.total_lbl, self.ok_lbl, self.bad_lbl]:
            lbl.setStyleSheet("font-size: 14px; padding: 4px;")
        stats_layout.addWidget(self.total_lbl)
        stats_layout.addWidget(self.ok_lbl)
        stats_layout.addWidget(self.bad_lbl)
        stats_box.setLayout(stats_layout)
        layout.addWidget(stats_box)

        # Details List
        details_box = QGroupBox("Pellet Measurements")
        details_layout = QVBoxLayout()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(250)
        self.detail_container = QWidget()
        self.detail_layout = QVBoxLayout(self.detail_container)
        self.detail_layout.addStretch()
        self.scroll.setWidget(self.detail_container)
        details_layout.addWidget(self.scroll)
        details_box.setLayout(details_layout)
        layout.addWidget(details_box)

        layout.addStretch()
        return widget

    def create_right_panel(self):
        widget = QWidget()
        widget.setStyleSheet("background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)

        self.img_label = QLabel("No image loaded")
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet("""
            QLabel {
                background-color: #f7fafc;
                border: 2px dashed #cbd5e0;
                border-radius: 8px;
                color: #718096;
                font-size: 16px;
            }
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
                QMessageBox.information(self, "Success", "YOLO model loaded successfully!")
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Could not load YOLO model:\n{e}")
        else:
            QMessageBox.information(self, "Info", "No YOLO model found. Using color detection only.")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self.current_image = cv2_safe_imread(path)
        if self.current_image is None:
            QMessageBox.critical(self, "Error", "Failed to load image.")
            return
        self.process_image()

    def process_image(self):
        if self.current_image is None:
            return
        img_disp = self.current_image.copy()
        self.detected_pellets = []

        polygons, confidences = [], []
        try:
            if self.yolo_detector:
                results = self.yolo_detector.predict(
                    self.current_image, conf=0.05, imgsz=640, device='cpu', verbose=False
                )
                r = results[0]
                temp_pellets = []
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf.item() * 100
                    roi = self.current_image[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    cnts = self.cv_detector.detect_pellets(roi)
                    if not cnts:
                        continue
                    c = max(cnts, key=cv2.contourArea)
                    c += np.array([x1, y1])
                    M = cv2.moments(c)
                    cx = int(M["m10"] / M["m00"]) if M["m00"] else int(c[:, 0].mean())
                    cy = int(M["m01"] / M["m00"]) if M["m00"] else int(c[:, 1].mean())
                    temp_pellets.append({'polygon': c, 'confidence': conf, 'center': (cx, cy)})

                # Filter duplicates
                filtered = []
                threshold = 10
                for p in sorted(temp_pellets, key=lambda x: -x['confidence']):
                    if not any(np.hypot(p['center'][0] - f['center'][0], p['center'][1] - f['center'][1]) < threshold for f in filtered):
                        filtered.append(p)

                polygons = [p['polygon'] for p in filtered]
                confidences = [p['confidence'] for p in filtered]
            else:
                polygons = self.cv_detector.detect_pellets(self.current_image)
                confidences = [100] * len(polygons)
        except Exception as e:
            print("Detection error:", e)
            polygons = self.cv_detector.detect_pellets(self.current_image)
            confidences = [100] * len(polygons)

        for i, (poly, conf) in enumerate(zip(polygons, confidences), 1):
            rect = cv2.minAreaRect(poly.astype(np.float32))
            w, h = rect[1]
            d = min(w, h) / self.pixels_per_mm
            l = max(w, h) / self.pixels_per_mm
            ok = self.d_min <= d <= self.d_max and self.l_min <= l <= self.l_max
            pellet = {
                'polygon': poly, 'diameter': d, 'length': l,
                'within': ok, 'confidence': conf, 'id': i
            }
            self.detected_pellets.append(pellet)
            self.draw_pellet(img_disp, pellet)

        self.update_stats()
        self.show_image(img_disp)

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
        cv2.putText(img, str(p['id']), (cx-15, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        if p['confidence'] < 100:
            cv2.putText(img, f"{p['confidence']:.0f}%", (cx-35, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    def update_stats(self):
        total = len(self.detected_pellets)
        ok = sum(1 for p in self.detected_pellets if p['within'])
        bad = total - ok

        self.total_lbl.setText(f"Total Pellets: <b>{total}</b>")
        self.ok_lbl.setText(f"OK: <b style='color:green'>{ok}</b>")
        self.bad_lbl.setText(f"Out of Spec: <b style='color:red'>{bad}</b>")

        # Clear previous details
        for i in reversed(range(self.detail_layout.count())):
            child = self.detail_layout.itemAt(i).widget()
            if child:
                child.deleteLater()

        for p in self.detected_pellets:
            status = "OK" if p['within'] else "Out of Spec"
            color = "#d4edda" if p['within'] else "#f8d7da"
            text_color = "#155724" if p['within'] else "#721c24"
            txt = f"Pellet {p['id']} → {status}<br>→ Diameter: {p['diameter']:.3f} mm | Length: {p['length']:.3f} mm"
            if p['confidence'] < 100:
                txt += f" | Conf: {p['confidence']:.1f}%"
            lbl = QLabel(txt)
            lbl.setStyleSheet(f"""
                background-color: {color}; 
                color: {text_color}; 
                padding: 10px; 
                margin: 3px; 
                border-radius: 6px; 
                border: 1px solid {'#c3e6cb' if p['within'] else '#f5c6cb'};
                font-size: 12px;
            """)
            self.detail_layout.insertWidget(self.detail_layout.count()-1, lbl)

    def show_image(self, cv_img):
        h, w = cv_img.shape[:2]
        max_w = self.img_label.width() - 20
        max_h = self.img_label.height() - 20
        scale = min(max_w / w, max_h / h, 1.0)
        new_size = QSize(int(w * scale), int(h * scale))
        pixmap = cv_to_qpixmap(cv_img, new_size)
        self.img_label.setPixmap(pixmap)
        self.img_label.setStyleSheet("background-color: white; border: none;")

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Clean modern look
    win = PelletMeasurementApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()