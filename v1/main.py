from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QGroupBox, QScrollArea,
                             QDoubleSpinBox, QMessageBox, QFrame)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QFont, QPalette, QColor

# --- Helper functions
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
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img if img is not None else None
    except:
        return None

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

class PelletMeasurementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Inspector Pro")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow { background-color: #f8f9fc; }
            QLabel { color: #2c3e50; font-family: Segoe UI; }
            QPushButton {
                background-color: #3498db; color: white; border: none;
                padding: 10px 16px; border-radius: 8px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton#toggleBtn:checked { background-color: #27ae60; }
            QGroupBox {
                font-weight: bold; border: 2px solid #ddd; border-radius: 10px;
                margin-top: 10px; padding-top: 10px; background: white;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 8px; }
            QScrollArea { border: none; background: transparent; }
            QDoubleSpinBox { padding: 6px; border: 1px solid #ccc; border-radius: 6px; }
        """)

        self.pixels_per_mm = 25.4
        self.target_diameter = 3.0
        self.target_length = 3.0
        self.tolerance = 0.5
        self.update_ranges()

        self.raw_image = None
        self.processed_image = None
        self.detected_pellets = []
        self.yolo_detector = None
        self.cv_detector = ColorPelletDetector()
        self.showing_processed = True

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

        # Left Panel
        left = self.create_left_panel()
        main_layout.addWidget(left, 1)

        # Right Panel - Image View
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Toggle Button
        self.toggle_btn = QPushButton("Hide Overlay (Show Raw)")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setFixedHeight(40)
        self.toggle_btn.setObjectName("toggleBtn")
        self.toggle_btn.clicked.connect(self.toggle_view)
        right_layout.addWidget(self.toggle_btn)

        # Image Label
        self.img_label = QLabel("Load an image to begin...")
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet("""
            QLabel {
                background: white;
                border: 3px solid #ddd;
                border-radius: 12px;
                padding: 10px;
            }
        """)
        self.img_label.setMinimumSize(600, 600)
        right_layout.addWidget(self.img_label)

        main_layout.addWidget(right, 3)

    def create_left_panel(self):
        panel = QWidget()
        panel.setFixedWidth(340)
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        # Title
        title = QLabel("Pellet Inspector Pro")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #2c3e50; margin: 10px;")
        layout.addWidget(title)

        # Load Button
        load_btn = QPushButton("Load Image")
        load_btn.setFixedHeight(50)
        load_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        load_btn.clicked.connect(self.load_image)
        layout.addWidget(load_btn)

        # Calibration
        cal_group = QGroupBox("Calibration")
        cal_layout = QVBoxLayout()
        h = QHBoxLayout()
        h.addWidget(QLabel("Pixels per mm:"))
        self.px_spin = QDoubleSpinBox()
        self.px_spin.setRange(1, 200)
        self.px_spin.setSingleStep(0.5)
        self.px_spin.setValue(self.pixels_per_mm)
        self.px_spin.setFont(QFont("Segoe UI", 10))
        self.px_spin.valueChanged.connect(lambda v: setattr(self, 'pixels_per_mm', v))
        h.addWidget(self.px_spin)
        cal_layout.addLayout(h)
        cal_group.setLayout(cal_layout)
        layout.addWidget(cal_group)

        # Stats
        stats = QGroupBox("Detection Summary")
        stats_layout = QVBoxLayout()
        self.total_lbl = QLabel("Total Pellets: 0")
        self.ok_lbl = QLabel("OK: 0")
        self.bad_lbl = QLabel("Out of Spec: 0")
        for lbl in [self.total_lbl, self.ok_lbl, self.bad_lbl]:
            lbl.setFont(QFont("Segoe UI", 11))
            lbl.setStyleSheet("padding: 5px;")
            stats_layout.addWidget(lbl)
        stats.setLayout(stats_layout)
        layout.addWidget(stats)

        # Details List
        details = QGroupBox("Pellet Details")
        details_layout = QVBoxLayout()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.detail_container = QWidget()
        self.detail_layout = QVBoxLayout(self.detail_container)
        self.detail_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.detail_container)
        details_layout.addWidget(self.scroll)
        details.setLayout(details_layout)
        layout.addWidget(details)

        layout.addStretch()
        return panel

    def load_model(self, path):
        if os.path.exists(path):
            try:
                self.yolo_detector = YOLO(path)
                print("Model loaded successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Model Error", f"Could not load model:\n{e}")
                self.yolo_detector = None

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return

        img = cv2_imread(path)
        if img is None:
            QMessageBox.critical(self, "Error", "Cannot read image file.")
            return

        self.raw_image = img
        self.process_image()

    def process_image(self):
        if self.raw_image is None:
            return

        img = self.raw_image.copy()
        self.detected_pellets = []

        polygons, confidences = [], []

        try:
            if self.yolo_detector:
                results = self.yolo_detector.predict(
                    img, conf=0.05, imgsz=640, device='cpu', verbose=False
                )
                r = results[0]

                temp = []
                for box in r.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf.item() * 100
                    roi = img[y1:y2, x1:x2]
                    if roi.size == 0: continue
                    cnts = self.cv_detector.detect_pellets(roi)
                    if not cnts: continue
                    c = max(cnts, key=cv2.contourArea)
                    c += np.array([x1, y1])
                    M = cv2.moments(c)
                    cx = int(M["m10"]/M["m00"]) if M["m00"] else int(c[:,0].mean())
                    cy = int(M["m01"]/M["m00"]) if M["m00"] else int(c[:,1].mean())
                    temp.append({'poly': c, 'conf': conf, 'center': (cx, cy)})

                # Remove duplicates
                filtered = []
                for p in sorted(temp, key=lambda x: -x['conf']):
                    if not any(np.hypot(p['center'][0]-f['center'][0], p['center'][1]-f['center'][1]) < 10 for f in filtered):
                        filtered.append(p)

                polygons = [p['poly'] for p in filtered]
                confidences = [p['conf'] for p in filtered]
            else:
                polygons = self.cv_detector.detect_pellets(img)
                confidences = [100.0] * len(polygons)
        except Exception as e:
            print("Detection error:", e)
            polygons = self.cv_detector.detect_pellets(img)
            confidences = [100.0] * len(polygons)

        # Draw and measure
        self.processed_image = img.copy()
        for i, (poly, conf) in enumerate(zip(polygons, confidences), 1):
            rect = cv2.minAreaRect(poly.astype(np.float32))
            w, h = rect[1]
            diameter = min(w, h) / self.pixels_per_mm
            length = max(w, h) / self.pixels_per_mm
            ok = self.d_min <= diameter <= self.d_max and self.l_min <= length <= self.l_max

            pellet = {
                'id': i, 'poly': poly, 'diameter': diameter, 'length': length,
                'within': ok, 'confidence': conf
            }
            self.detected_pellets.append(pellet)
            self.draw_pellet(self.processed_image, pellet)

        self.update_ui()
        self.display_current_image()

    def draw_pellet(self, img, p):
        color = (0, 200, 0) if p['within'] else (0, 0, 230)
        overlay = img.copy()
        cv2.fillPoly(overlay, [p['poly']], color)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        box = np.int0(cv2.boxPoints(cv2.minAreaRect(p['poly'].astype(np.float32))))
        cv2.drawContours(img, [box], 0, color, 4)

        M = cv2.moments(p['poly'])
        cx = int(M["m10"]/M["m00"]) if M["m00"] else int(p['poly'][:,0].mean())
        cy = int(M["m01"]/M["m00"]) if M["m00"] else int(p['poly'][:,1].mean())

        # Large bold number
        cv2.putText(img, str(p['id']), (cx-25, cy+15),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(img, str(p['id']), (cx-25, cy+15),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 0), 2, cv2.LINE_AA)

        if p['confidence'] < 99.5:
            cv2.putText(img, f"{p['confidence']:.0f}%", (cx-40, cy-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    def toggle_view(self):
        self.showing_processed = not self.toggle_btn.isChecked()
        self.toggle_btn.setText("Show Overlay" if self.showing_processed else "Hide Overlay (Show Raw)")
        self.display_current_image()

    def display_current_image(self):
        img = self.processed_image if self.showing_processed else self.raw_image
        if img is None:
            return
        h, w = img.shape[:2]
        max_w = self.img_label.width() - 20
        max_h = self.img_label.height() - 20
        scale = min(max_w / w, max_h / h, 1.0)
        new_size = QSize(int(w * scale), int(h * scale))
        pixmap = cv_to_qpixmap(img, new_size)
        self.img_label.setPixmap(pixmap)

    def update_ui(self):
        total = len(self.detected_pellets)
        ok = sum(1 for p in self.detected_pellets if p['within'])
        bad = total - ok

        self.total_lbl.setText(f"Total Pellets: {total}")
        self.ok_lbl.setText(f"OK: {ok}")
        self.bad_lbl.setText(f"Out of Spec: {bad}")
        self.ok_lbl.setStyleSheet("color: #27ae60; font-weight: bold;" if ok else "")
        self.bad_lbl.setStyleSheet("color: #e74c3c; font-weight: bold;" if bad else "")

        # Clear old details
        for i in reversed(range(self.detail_layout.count())):
            widget = self.detail_layout.itemAt(i).widget()
            if widget: widget.setParent(None)

        # Add new
        for p in self.detected_pellets:
            status = "OK" if p['within'] else "BAD"
            color = "#2ecc71" if p['within'] else "#e74c3c"
            text = f"<b>Pellet {p['id']} — <span style='color:{color}'>{status}</span></b><br>"
            text += f"  • Diameter: {p['diameter']:.3f} mm<br>"
            text += f"  • Length:   {p['length']:.3f} mm<br>"
            text += f"  • Confidence: {p['confidence']:.1f}%"

            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"""
                background: white; border-left: 6px solid {color};
                padding: 12px; margin: 4px; border-radius: 6px;
                font-family: Segoe UI; font-size: 10pt;
            """)
            self.detail_layout.addWidget(lbl)

        self.detail_layout.addStretch()

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = PelletMeasurementApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()