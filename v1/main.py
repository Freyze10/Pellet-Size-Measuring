from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QGroupBox, QScrollArea, QTextEdit,
                             QDoubleSpinBox, QMessageBox, QSizePolicy)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont

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

# --- Main App ---
class PelletMeasurementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Inspector Pro")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #f8f9fa;")

        self.pixels_per_mm = 25.4
        self.target_diameter = 3.0
        self.target_length = 3.0
        self.tolerance = 0.5
        self.update_ranges()

        self.current_image = None
        self.raw_image = None
        self.annotated_image = None
        self.detected_pellets = []
        self.yolo_detector = None
        self.cv_detector = ColorPelletDetector()
        self.showing_annotated = True

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
        panel.setStyleSheet("background-color: white; border-radius: 10px; margin: 10px; padding: 10px;")
        panel.setFixedWidth(340)
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Title
        title = QLabel("Pellet Inspector Pro")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; padding: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Load Image Button
        self.load_btn = QPushButton("Load Image")
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db; color: white; border-radius: 8px; padding: 12px;
                font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        self.load_btn.clicked.connect(self.load_image)
        layout.addWidget(self.load_btn)

        # Toggle Overlay Button
        self.toggle_btn = QPushButton("Show Raw Image")
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6; color: white; border-radius: 8px; padding: 10px;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #7f8c8d; }
        """)
        self.toggle_btn.clicked.connect(self.toggle_image_view)
        layout.addWidget(self.toggle_btn)

        # Calibration
        calib_group = QGroupBox("Calibration")
        calib_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2c3e50; }")
        calib_layout = QVBoxLayout()
        h = QHBoxLayout()
        h.addWidget(QLabel("Pixels per mm:"))
        self.px_spin = QDoubleSpinBox()
        self.px_spin.setRange(0.1, 200)
        self.px_spin.setValue(self.pixels_per_mm)
        self.px_spin.setStyleSheet("padding: 5px; font-size: 13px;")
        self.px_spin.valueChanged.connect(lambda v: setattr(self, 'pixels_per_mm', v))
        h.addWidget(self.px_spin)
        calib_layout.addLayout(h)
        calib_group.setLayout(calib_layout)
        layout.addWidget(calib_group)

        # Stats
        stats_group = QGroupBox("Detection Summary")
        stats_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2c3e50; }")
        stats_layout = QVBoxLayout()
        self.total_lbl = QLabel("Total: 0")
        self.ok_lbl = QLabel("OK: 0")
        self.bad_lbl = QLabel("BAD: 0")
        for lbl in [self.total_lbl, self.ok_lbl, self.bad_lbl]:
            lbl.setStyleSheet("font-size: 15px; padding: 4px;")
        stats_layout.addWidget(self.total_lbl)
        stats_layout.addWidget(self.ok_lbl)
        stats_layout.addWidget(self.bad_lbl)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Details
        details_group = QGroupBox("Pellet Measurements")
        details_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2c3e50; }")
        details_layout = QVBoxLayout()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border: none;")
        self.det_w = QWidget()
        self.det_l = QVBoxLayout()
        self.det_w.setLayout(self.det_l)
        self.scroll.setWidget(self.det_w)
        details_layout.addWidget(self.scroll)
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        layout.addStretch()
        return panel

    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        self.img_lbl = QLabel("Load an image to begin...")
        self.img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_lbl.setStyleSheet("""
            QLabel {
                background-color: #ecf0f1;
                border: 2px dashed #bdc3c7;
                border-radius: 12px;
                font-size: 18px;
                color: #7f8c8d;
            }
        """)
        self.img_lbl.setMinimumSize(600, 500)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.img_lbl)
        scroll_area.setStyleSheet("border: none; background: transparent;")
        layout.addWidget(scroll_area)
        return panel

    def load_model(self, path):
        if os.path.exists(path):
            try:
                self.yolo_detector = YOLO(path)
                print(f"Model loaded: {path}")
            except Exception as e:
                print("YOLO load error:", e)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        img = cv2_safe_imread(path)
        if img is None:
            QMessageBox.critical(self, "Error", "Failed to load image")
            return

        self.raw_image = img.copy()
        self.current_image = img.copy()
        self.process_image()

    def toggle_image_view(self):
        if self.raw_image is None:
            return
        self.showing_annotated = not self.showing_annotated
        if self.showing_annotated and self.annotated_image is not None:
            self.show_image(self.annotated_image)
            self.toggle_btn.setText("Show Raw Image")
        else:
            self.show_image(self.raw_image)
            self.toggle_btn.setText("Show Annotated")

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

                filtered = []
                threshold = 8
                for p in sorted(temp_pellets, key=lambda x: -x['confidence']):
                    if not any(np.hypot(p['center'][0] - f['center'][0], p['center'][1] - f['center'][1]) < threshold for f in filtered):
                        filtered.append(p)

                polygons = [p['polygon'] for p in filtered]
                confidences = [p['confidence'] for p in filtered]
        except:
            polygons = self.cv_detector.detect_pellets(self.current_image)
            confidences = [100] * len(polygons)

        # --- Measure & Draw ---
        for i, (poly, conf) in enumerate(zip(polygons, confidences), 1):
            rect = cv2.minAreaRect(poly.astype(np.float32))
            w, h = rect[1]
            d = min(w, h) / self.pixels_per_mm
            l = max(w, h) / self.pixels_per_mm
            ok = self.d_min <= d <= self.d_max and self.l_min <= l <= self.l_max
            pellet = {'polygon': poly, 'diameter': d, 'length': l, 'within': ok, 'confidence': conf, 'id': i}
            self.detected_pellets.append(pellet)
            self.draw_pellet(img_disp, pellet)

        self.annotated_image = img_disp
        self.update_stats()
        self.show_image(img_disp)
        self.toggle_btn.setText("Show Raw Image")
        self.showing_annotated = True

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

        # Big, bold, readable number
        cv2.putText(img, str(p['id']), (cx - 30, cy + 15),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, (255, 255, 255), 7, cv2.LINE_AA)
        cv2.putText(img, str(p['id']), (cx - 30, cy + 15),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, (0, 0, 0), 3, cv2.LINE_AA)

        if p['confidence'] < 100:
            cv2.putText(img, f"{p['confidence']:.0f}%", (cx - 40, cy - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)

    def update_stats(self):
        total = len(self.detected_pellets)
        ok = sum(1 for p in self.detected_pellets if p['within'])
        bad = total - ok
        self.total_lbl.setText(f"Total: <b>{total}</b>")
        self.ok_lbl.setText(f"OK: <b style='color:green'>{ok}</b>")
        self.bad_lbl.setText(f"BAD: <b style='color:red'>{bad}</b>")

        for i in reversed(range(self.det_l.count())):
            widget = self.det_l.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        for i, p in enumerate(self.detected_pellets, 1):
            status = "OK" if p['within'] else "BAD"
            color = "#27ae60" if p['within'] else "#e74c3c"
            txt = f"<b>Pellet {i}</b> — <span style='color:{color}; font-weight:bold'>{status}</span><br>" \
                  f" Diameter: {p['diameter']:.3f} mm<br>" \
                  f" Length:    {p['length']:.3f} mm<br>" \
                  f" Confidence: {p['confidence']:.1f}%"
            lbl = QLabel(txt)
            lbl.setStyleSheet(f"""
                background: #ffffff; padding: 10px; margin: 3px; border-left: 6px solid {color};
                border-radius: 6px; font-size: 13px;
            """)
            lbl.setWordWrap(True)
            self.det_l.addWidget(lbl)
        self.det_l.addStretch()

    def show_image(self, cv_img):
        if cv_img is None:
            return
        h, w = cv Img.shape[:2]
        max_w = self.img_lbl.width() - 20 if self.img_lbl.width() > 40 else 1000
        scale = min(1.0, max_w / w)
        target_size = QSize(int(w * scale), int(h * scale))
        pixmap = cv_to_qpixmap(cv_img, target_size)
        self.img_lbl.setPixmap(pixmap)
        self.img_lbl.setStyleSheet("background-color: #ffffff; border: 1px solid #ddd; border-radius: 10px;")

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Clean modern look
    win = PelletMeasurementApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()