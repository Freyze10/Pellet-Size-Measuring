from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
import glob
import random
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QGroupBox, QScrollArea, QTextEdit,
                             QDoubleSpinBox, QMessageBox, QSizePolicy, QFrame)
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont, QPalette, QColor

# --- Helper functions ---
def cv_to_qpixmap(cv_img, target_size=None):
    if cv_img is None:
        return QPixmap()
    if len(cv_img.shape) == 3:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_line, QImage.Format.Format_RGB888)
    else:
        h, w = cv_img.shape
        bytes_line = w
        qimg = QImage(cv_img.data, w, h, bytes_line, QImage.Format.Format_Grayscale8)
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

# --- Main app ---
class PelletMeasurementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Inspector Pro")
        self.setGeometry(100, 100, 1200, 750)  # Clean professional size
        self.setStyleSheet("""
            QMainWindow { background-color: #f8f9fc; }
            QLabel { color: #2d3748; font-size: 13px; }
            QPushButton {
                background-color: #3b82f6; color: white; border: none;
                padding: 10px 16px; border-radius: 8px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2563eb; }
            QPushButton:pressed { background-color: #1d4ed8; }
            QGroupBox {
                font-weight: bold; border: 2px solid #e2e8f0; border-radius: 10px;
                margin: 10px; padding-top: 10px; background-color: white;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 10px; }
            QDoubleSpinBox { padding: 6px; border: 1px solid #cbd5e0; border-radius: 6px; }
        """)

        self.pixels_per_mm = 25.4
        self.target_diameter = 3.0
        self.target_length = 3.0
        self.tolerance = 0.5
        self.update_ranges()

        self.current_image = None
        self.raw_image = None  # Store original
        self.processed_image = None  # Store annotated
        self.showing_overlay = True
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
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        # Left Panel
        main_layout.addWidget(self.create_left_panel(), 1)
        # Right Panel (Image)
        main_layout.addWidget(self.create_right_panel(), 3)

    def create_left_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(380)
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Title
        title = QLabel("Pellet Inspector Pro")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #1e40af; margin: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Load Image Button
        self.load_btn = QPushButton("Load Image")
        self.load_btn.setStyleSheet("background-color: #10b981; padding: 14px;")
        self.load_btn.clicked.connect(self.load_image)
        layout.addWidget(self.load_btn)

        # Toggle Overlay Button
        self.toggle_btn = QPushButton("Show Raw Image")
        self.toggle_btn.setStyleSheet("background-color: #6366f1;")
        self.toggle_btn.clicked.connect(self.toggle_overlay)
        layout.addWidget(self.toggle_btn)

        # Calibration
        calib_box = QGroupBox("Calibration")
        calib_layout = QVBoxLayout()
        h = QHBoxLayout()
        h.addWidget(QLabel("Pixels per mm:"))
        self.px_spin = QDoubleSpinBox()
        self.px_spin.setRange(1.0, 200.0)
        self.px_spin.setDecimals(2)
        self.px_spin.setValue(self.pixels_per_mm)
        self.px_spin.valueChanged.connect(lambda v: setattr(self, 'pixels_per_mm', v) or self.reprocess())
        h.addWidget(self.px_spin)
        calib_layout.addLayout(h)
        calib_box.setLayout(calib_layout)
        layout.addWidget(calib_box)

        # Statistics
        stats_box = QGroupBox("Detection Summary")
        stats_layout = QVBoxLayout()
        self.total_lbl = QLabel("Total: 0")
        self.ok_lbl = QLabel("OK: 0")
        self.bad_lbl = QLabel("BAD: 0")
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.total_lbl.setFont(font)
        self.ok_lbl.setFont(font)
        self.bad_lbl.setFont(font)
        self.ok_lbl.setStyleSheet("color: #16a34a;")
        self.bad_lbl.setStyleSheet("color: #dc2626;")
        stats_layout.addWidget(self.total_lbl)
        stats_layout.addWidget(self.ok_lbl)
        stats_layout.addWidget(self.bad_lbl)
        stats_box.setLayout(stats_layout)
        layout.addWidget(stats_box)

        # Pellet Details
        details_box = QGroupBox("Pellet Details")
        details_layout = QVBoxLayout()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; }")
        self.detail_container = QWidget()
        self.detail_layout = QVBoxLayout()
        self.detail_container.setLayout(self.detail_layout)
        self.scroll.setWidget(self.detail_container)
        details_layout.addWidget(self.scroll)
        details_box.setLayout(details_layout)
        layout.addWidget(details_box)

        layout.addStretch()
        return panel

    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        self.img_lbl = QLabel("Load an image to begin...")
        self.img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_lbl.setStyleSheet("""
            background-color: white; 
            border: 3px dashed #cbd5e0; 
            border-radius: 12px;
            font-size: 18px; 
            color: #94a3b8;
        """)
        self.img_lbl.setMinimumSize(600, 500)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.img_lbl)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        layout.addWidget(scroll_area)
        return panel

    def load_model(self, path):
        if os.path.exists(path):
            try:
                self.yolo_detector = YOLO(path)
                QMessageBox.information(self, "Success", "YOLO model loaded successfully!")
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Could not load YOLO model:\n{e}")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        img = cv2_safe_imread(path)
        if img is None:
            QMessageBox.critical(self, "Error", "Failed to load image.")
            return

        self.raw_image = img.copy()
        self.current_image = img.copy()
        self.process_image()

    def toggle_overlay(self):
        self.showing_overlay = not self.showing_overlay
        self.toggle_btn.setText("Show Raw Image" if self.showing_overlay else "Show Detection")
        self.show_image(self.raw_image if not self.showing_overlay else self.processed_image)

    def reprocess(self):
        if self.raw_image is not None:
            self.current_image = self.raw_image.copy()
            self.process_image()

    def process_image(self):
        if self.current_image is None:
            return

        img_disp = self.current_image.copy()
        self.detected_pellets = []

        # --- Detection Logic (unchanged) ---
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

        # --- Measurement & Drawing ---
        for i, (poly, conf) in enumerate(zip(polygons, confidences), 1):
            rect = cv2.minAreaRect(poly.astype(np.float32))
            w, h = rect[1]
            d = min(w, h) / self.pixels_per_mm
            l = max(w, h) / self.pixels_per_mm
            ok = self.d_min <= d <= self.d_max and self.l_min <= l <= self.l_max
            pellet = {'polygon': poly, 'diameter': d, 'length': l, 'within': ok, 'confidence': conf, 'id': i}
            self.detected_pellets.append(pellet)
            self.draw_pellet(img_disp, pellet)

        self.processed_image = img_disp
        self.update_stats()
        self.show_image(img_disp)

    def draw_pellet(self, img, p):
        color = (0, 255, 0) if p['within'] else (0, 0, 255)
        overlay = img.copy()
        cv2.fillPoly(overlay, [p['polygon'].reshape(-1, 1, 2)], color)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        box = np.intp(cv2.boxPoints(cv2.minAreaRect(p['polygon'].astype(np.float32))))
        cv2.drawContours(img, [box], 0, color, 4)

        M = cv2.moments(p['polygon'])
        cx = int(M["m10"]/M["m00"]) if M["m00"] else int(p['polygon'][:,0].mean())
        cy = int(M["m01"]/M["m00"]) if M["m00"] else int(p['polygon'][:,1].mean())

        # Big readable number
        cv2.putText(img, str(p['id']), (cx - 30, cy + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 7, cv2.LINE_AA)
        cv2.putText(img, str(p['id']), (cx - 30, cy + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 0, 0), 3, cv2.LINE_AA)

        if p['confidence'] < 100:
            cv2.putText(img, f"{p['confidence']:.0f}%", (cx - 40, cy - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    def update_stats(self):
        total = len(self.detected_pellets)
        ok = sum(1 for p in self.detected_pellets if p['within'])
        bad = total - ok
        self.total_lbl.setText(f"Total Detected: {total}")
        self.ok_lbl.setText(f"OK: {ok}")
        self.bad_lbl.setText(f"Out of Spec: {bad}")

        # Clear and update details
        for i in reversed(range(self.detail_layout.count())):
            widget = self.detail_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        for p in self.detected_pellets:
            status = "OK" if p['within'] else "BAD"
            color = "#16a34a" if p['within'] else "#dc2626"
            text = f"Pellet {p['id']:2d} → {status} | D: {p['diameter']:.3f} mm | L: {p['length']:.3f} mm | Conf: {p['confidence']:.1f}%"
            lbl = QLabel(text)
            lbl.setStyleSheet(f"""
                background-color: white; padding: 10px; margin: 4px; 
                border-left: 6px solid {color}; border-radius: 6px;
                font-family: Segoe UI; font-size: 13px;
            """)
            self.detail_layout.addWidget(lbl)
        self.detail_layout.addStretch()

    def show_image(self, cv_img):
        if cv_img is None:
            return
        h, w = cv_img.shape[:2]
        avail_w = self.img_lbl.width() - 20
        if avail_w < 100:
            avail_w = 800
        scale = min(avail_w / w, 680 / h)
        target_size = QSize(int(w * scale), int(h * scale))
        pixmap = cv_to_qpixmap(cv_img, target_size)
        self.img_lbl.setPixmap(pixmap)
        self.img_lbl.setFixedSize(target_size)

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Clean modern look
    win = PelletMeasurementApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()