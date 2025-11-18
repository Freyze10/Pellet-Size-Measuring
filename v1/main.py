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
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont


# --- Helper functions ---
def cv_to_qpixmap(cv_img, target_size=None):
    if cv_img is None: return QPixmap()
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
    if target_size: pix = pix.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio,
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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        for c in contours:
            a = cv2.contourArea(c)
            if self.MIN_PIXEL_AREA <= a <= self.MAX_PIXEL_AREA:
                w, h = cv2.minAreaRect(c)[1]
                if min(w, h) > 0 and 1.0 <= max(w, h) / min(w, h) <= 5.0:
                    valid.append(c)
        return valid


# --- Main app ---
class PelletMeasurementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Inspector - YOLO Scanner")
        self.setGeometry(100, 100, 1200, 700)

        self.pixels_per_mm = 25.4
        self.target_diameter = 3.0
        self.target_length = 3.0
        self.tolerance = 0.5
        self.update_ranges()

        self.current_image = None
        self.processed_image = None
        self.detected_pellets = []
        self.show_processed = True
        self.yolo_detector = None
        self.cv_detector = ColorPelletDetector()
        self.dataset_folder = "pellet_label_yolo"

        self.setup_styles()
        self.init_ui()
        self.load_model("trained_model/best.pt")

    def setup_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
                border-radius: 5px;
                min-height: 35px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton#toggleBtn {
                background-color: #2196F3;
            }
            QPushButton#toggleBtn:hover {
                background-color: #0b7dda;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
                color: #2c3e50;
            }
            QLabel {
                font-size: 12px;
                color: #333;
            }
            QDoubleSpinBox {
                padding: 5px;
                border: 2px solid #e0e0e0;
                border-radius: 4px;
                background-color: white;
                font-size: 12px;
            }
            QDoubleSpinBox:focus {
                border: 2px solid #4CAF50;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

    def update_ranges(self):
        self.d_min = self.target_diameter - self.tolerance
        self.d_max = self.target_diameter + self.tolerance
        self.l_min = self.target_length - self.tolerance
        self.l_max = self.target_length + self.tolerance

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        central.setLayout(layout)

        layout.addWidget(self.left_panel(), 1)
        layout.addWidget(self.right_panel(), 3)

    def left_panel(self):
        w = QWidget()
        l = QVBoxLayout()
        l.setSpacing(12)
        w.setLayout(l)

        # Title
        title = QLabel("📊 Pellet Inspector")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #2c3e50; padding: 10px; background-color: white; border-radius: 8px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l.addWidget(title)

        # Load Image button
        self.load_btn = QPushButton("📁 Load Image")
        self.load_btn.clicked.connect(self.load_image)
        l.addWidget(self.load_btn)

        # Toggle View button
        self.toggle_btn = QPushButton("🔄 Show Original Image")
        self.toggle_btn.setObjectName("toggleBtn")
        self.toggle_btn.clicked.connect(self.toggle_view)
        self.toggle_btn.setEnabled(False)
        l.addWidget(self.toggle_btn)

        # Calibration
        g = QGroupBox("⚙️ Calibration")
        gl = QVBoxLayout()
        gl.setSpacing(8)
        h = QHBoxLayout()
        h.addWidget(QLabel("Pixels per mm:"))
        self.px_spin = QDoubleSpinBox()
        self.px_spin.setRange(0.1, 200)
        self.px_spin.setValue(self.pixels_per_mm)
        self.px_spin.valueChanged.connect(lambda v: setattr(self, 'pixels_per_mm', v))
        h.addWidget(self.px_spin)
        gl.addLayout(h)
        g.setLayout(gl)
        l.addWidget(g)

        # Stats
        g = QGroupBox("📈 Detection Statistics")
        gl = QVBoxLayout()
        gl.setSpacing(8)

        self.total_lbl = QLabel("Total Pellets: 0")
        self.total_lbl.setStyleSheet("font-size: 13px; padding: 5px; color: #2c3e50; font-weight: bold;")

        self.ok_lbl = QLabel("✓ Passed: 0")
        self.ok_lbl.setStyleSheet("font-size: 13px; padding: 5px; color: #27ae60; font-weight: bold;")

        self.bad_lbl = QLabel("✗ Failed: 0")
        self.bad_lbl.setStyleSheet("font-size: 13px; padding: 5px; color: #e74c3c; font-weight: bold;")

        gl.addWidget(self.total_lbl)
        gl.addWidget(self.ok_lbl)
        gl.addWidget(self.bad_lbl)
        g.setLayout(gl)
        l.addWidget(g)

        # Details
        g = QGroupBox("📋 Pellet Details")
        gl = QVBoxLayout()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.det_w = QWidget()
        self.det_w.setStyleSheet("background-color: #fafafa;")
        self.det_l = QVBoxLayout()
        self.det_l.setSpacing(8)
        self.det_w.setLayout(self.det_l)
        self.scroll.setWidget(self.det_w)
        gl.addWidget(self.scroll)
        g.setLayout(gl)
        l.addWidget(g, stretch=1)

        l.addStretch()
        return w

    def right_panel(self):
        w = QWidget()
        l = QVBoxLayout()
        l.setContentsMargins(0, 0, 0, 0)
        w.setLayout(l)

        self.img_lbl = QLabel("📷 Load an image to begin inspection...")
        self.img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_lbl.setStyleSheet("""
            border: 3px dashed #bdc3c7;
            background: white;
            border-radius: 8px;
            color: #7f8c8d;
            font-size: 14px;
            padding: 20px;
        """)

        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setWidget(self.img_lbl)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: white;
                border-radius: 8px;
            }
        """)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        l.addWidget(scroll)
        return w

    def load_model(self, path):
        if os.path.exists(path):
            try:
                self.yolo_detector = YOLO(path)
                QMessageBox.information(self, "✓ Model Loaded", f"YOLO model successfully loaded from:\n{path}")
            except Exception as e:
                print("YOLO load error:", e)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path: return
        self.current_image = cv2_safe_imread(path)
        if self.current_image is None:
            QMessageBox.critical(self, "Error", "Failed to load image")
            return
        self.show_processed = True
        self.toggle_btn.setText("🔄 Show Original Image")
        self.toggle_btn.setEnabled(True)
        self.process_image()

    def toggle_view(self):
        if self.current_image is None: return
        self.show_processed = not self.show_processed
        if self.show_processed:
            self.toggle_btn.setText("🔄 Show Original Image")
            self.show_image(self.processed_image)
        else:
            self.toggle_btn.setText("🔄 Show Processed Image")
            self.show_image(self.current_image)

    def process_image(self):
        if self.current_image is None: return
        img_disp = self.current_image.copy()
        self.detected_pellets = []

        # --- YOLO detection ---
        polygons, confidences, bboxes = [], [], []
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
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    roi = self.current_image[y1:y2, x1:x2]
                    if roi.size == 0: continue
                    cnts = self.cv_detector.detect_pellets(roi)
                    if not cnts: continue
                    c = max(cnts, key=cv2.contourArea)
                    c += np.array([x1, y1])

                    temp_pellets.append({
                        'polygon': c,
                        'confidence': conf,
                        'center': (cx, cy),
                        'bbox': (x1, y1, x2, y2)
                    })

                # --- FILTER OVERLAPPING BOXES BY IOU + HIGHEST CONF ---
                filtered = []

                def calculate_iou(box1, box2):
                    x1_1, y1_1, x2_1, y2_1 = box1
                    x1_2, y1_2, x2_2, y2_2 = box2

                    xi1 = max(x1_1, x1_2)
                    yi1 = max(y1_1, y1_2)
                    xi2 = min(x2_1, x2_2)
                    yi2 = min(y2_1, y2_2)
                    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

                    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
                    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
                    union_area = box1_area + box2_area - inter_area

                    return inter_area / union_area if union_area > 0 else 0

                iou_threshold = 0.3
                for p in sorted(temp_pellets, key=lambda x: -x['confidence']):
                    is_duplicate = False
                    for f in filtered:
                        iou = calculate_iou(p['bbox'], f['bbox'])
                        if iou > iou_threshold:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        filtered.append(p)

                polygons = [p['polygon'] for p in filtered]
                confidences = [p['confidence'] for p in filtered]
                bboxes = [p['bbox'] for p in filtered]

        except:
            polygons = self.cv_detector.detect_pellets(self.current_image)
            confidences = [100] * len(polygons)
            bboxes = [None] * len(polygons)

        # --- Measurement & Drawing ---
        for i, (poly, conf, bbox) in enumerate(zip(polygons, confidences, bboxes), 1):
            rect = cv2.minAreaRect(poly.astype(np.float32))
            w, h = rect[1]
            d = min(w, h) / self.pixels_per_mm
            l = max(w, h) / self.pixels_per_mm
            ok = self.d_min <= d <= self.d_max and self.l_min <= l <= self.l_max
            pellet = {'polygon': poly, 'diameter': d, 'length': l, 'within': ok, 'confidence': conf, 'id': i,
                      'bbox': bbox}
            self.detected_pellets.append(pellet)
            self.draw_pellet(img_disp, pellet)

        self.processed_image = img_disp
        self.update_stats()
        self.show_image(img_disp)

    def draw_pellet(self, img, p):
        color = (0, 255, 0) if p['within'] else (0, 0, 255)

        # Draw YOLO bounding box if available
        if p['bbox'] is not None:
            x1, y1, x2, y2 = p['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 165, 0), 2)  # Orange bbox

        # Draw filled polygon
        overlay = img.copy()
        cv2.fillPoly(overlay, [p['polygon'].reshape(-1, 1, 2)], color)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

        # Draw rotated rectangle around polygon
        box = np.intp(cv2.boxPoints(cv2.minAreaRect(p['polygon'].astype(np.float32))))
        cv2.drawContours(img, [box], 0, color, 3)

        # Draw pellet ID
        M = cv2.moments(p['polygon'])
        cx = int(M["m10"] / M["m00"]) if M["m00"] else int(p['polygon'][:, 0].mean())
        cy = int(M["m01"] / M["m00"]) if M["m00"] else int(p['polygon'][:, 1].mean())

        # Larger, more readable label
        cv2.putText(img, str(p['id']), (cx - 25, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        cv2.putText(img, str(p['id']), (cx - 25, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

        # Draw confidence on bbox if available
        if p['confidence'] < 100 and p['bbox'] is not None:
            x1, y1, x2, y2 = p['bbox']
            cv2.putText(img, f"{p['confidence']:.0f}%", (x1 + 5, y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

    def update_stats(self):
        total = len(self.detected_pellets)
        ok = sum(1 for p in self.detected_pellets if p['within'])
        bad = total - ok
        self.total_lbl.setText(f"Total Pellets: {total}")
        self.ok_lbl.setText(f"✓ Passed: {ok}")
        self.bad_lbl.setText(f"✗ Failed: {bad}")

        for i in reversed(range(self.det_l.count())):
            w = self.det_l.itemAt(i).widget()
            if w: w.deleteLater()

        for i, p in enumerate(self.detected_pellets, 1):
            status = "✓ PASSED" if p['within'] else "✗ FAILED"
            color = "#d5f4e6" if p['within'] else "#fadbd8"
            border_color = "#27ae60" if p['within'] else "#e74c3c"

            txt = f"<b>Pellet {i}</b> - {status}<br>" \
                  f"<span style='color: #555;'>Diameter: {p['diameter']:.3f} mm | " \
                  f"Length: {p['length']:.3f} mm | Confidence: {p['confidence']:.1f}%</span>"

            lbl = QLabel(txt)
            lbl.setStyleSheet(f"""
                padding: 10px;
                margin: 2px;
                border-left: 4px solid {border_color};
                background-color: {color};
                border-radius: 4px;
                font-size: 12px;
            """)
            lbl.setWordWrap(True)
            self.det_l.addWidget(lbl)

        self.det_l.addStretch()

    def show_image(self, cv_img):
        avail_w = self.img_lbl.parent().width() - 30
        h, w = cv_img.shape[:2]
        scale = avail_w / w
        pix = cv_to_qpixmap(cv_img, QSize(int(w * scale), int(h * scale)))
        self.img_lbl.setFixedWidth(int(w * scale))
        self.img_lbl.setMinimumHeight(int(h * scale))
        self.img_lbl.setPixmap(pix)
        self.img_lbl.setStyleSheet("border: none; background: white;")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = PelletMeasurementApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()