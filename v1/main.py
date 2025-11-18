from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QGroupBox, QScrollArea,
                             QDoubleSpinBox, QMessageBox, QSizePolicy)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont


# --- Helper functions ---
def cv_to_qpixmap(cv_img, target_size=None):
    if cv_img is None:
        return QPixmap()
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
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


# --- Color fallback detector (for refining YOLO detections) ---
class ColorPelletDetector:
    LOWER_BLUE = np.array([90, 100, 30])
    UPPER_BLUE = np.array([130, 255, 255])
    MIN_AREA = 300
    MAX_AREA = 25000

    def detect_pellets(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWER_BLUE, self.UPPER_BLUE)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.MIN_AREA <= area <= self.MAX_AREA:
                rect = cv2.minAreaRect(c)
                w, h = rect[1]
                if w > 0 and h > 0:
                    aspect = max(w, h) / min(w, h)
                    if 1.0 <= aspect <= 6.0:
                        valid.append(c)
        return valid


# --- Main Application ---
class PelletMeasurementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Inspector - YOLOv11 @ 1275×1754")
        self.setGeometry(100, 100, 1400, 900)

        # --- Critical: Exact size your model was trained on ---
        self.TARGET_W = 1275
        self.TARGET_H = 1754
        self.TARGET_SIZE = (self.TARGET_W, self.TARGET_H)

        # Calibration (150 DPI → ~25.4 px/mm on short side)
        self.pixels_per_mm = 25.4

        # Target specs
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

        self.setup_styles()
        self.init_ui()
        self.load_model("trained_model/best.pt")

    def setup_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f8f9fa; }
            QPushButton {
                background-color: #2ecc71; color: white; border: none;
                padding: 12px 24px; font-size: 14px; font-weight: bold;
                border-radius: 8px; min-height: 40px;
            }
            QPushButton:hover { background-color: #27ae60; }
            QPushButton#toggleBtn { background-color: #3498db; }
            QPushButton#toggleBtn:hover { background-color: #2980b9; }
            QGroupBox {
                font-weight: bold; font-size: 14px; border: 2px solid #ddd;
                border-radius: 10px; margin-top: 12px; padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 8px; }
            QLabel { font-size: 13px; }
            QDoubleSpinBox { padding: 8px; border: 2px solid #ddd; border-radius: 6px; }
        """)

    def update_ranges(self):
        self.d_min = self.target_diameter - self.tolerance
        self.d_max = self.target_diameter + self.tolerance
        self.l_min = self.target_length - self.tolerance
        self.l_max = self.target_length + self.tolerance

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        central.setLayout(main_layout)

        main_layout.addWidget(self.left_panel(), 1)
        main_layout.addWidget(self.right_panel(), 3)

    def left_panel(self):
        w = QWidget()
        l = QVBoxLayout()
        l.setSpacing(15)
        w.setLayout(l)

        title = QLabel("Pellet Inspector\nYOLOv11 @ 1275×1754")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #2c3e50; background: white; padding: 15px; border-radius: 10px;")
        l.addWidget(title)

        self.load_btn = QPushButton("Load Scanned Image")
        self.load_btn.clicked.connect(self.load_image)
        l.addWidget(self.load_btn)

        self.toggle_btn = QPushButton("Show Original")
        self.toggle_btn.setObjectName("toggleBtn")
        self.toggle_btn.clicked.connect(self.toggle_view)
        self.toggle_btn.setEnabled(False)
        l.addWidget(self.toggle_btn)

        # Calibration
        g = QGroupBox("Calibration (px/mm)")
        gl = QVBoxLayout()
        h = QHBoxLayout()
        h.addWidget(QLabel("Pixels per mm:"))
        self.px_spin = QDoubleSpinBox()
        self.px_spin.setRange(1.0, 100.0)
        self.px_spin.setDecimals(3)
        self.px_spin.setValue(self.pixels_per_mm)
        self.px_spin.valueChanged.connect(lambda v: setattr(self, 'pixels_per_mm', v))
        h.addWidget(self.px_spin)
        gl.addLayout(h)
        g.setLayout(gl)
        l.addWidget(g)

        # Stats
        g = QGroupBox("Detection Statistics")
        gl = QVBoxLayout()
        self.total_lbl = QLabel("Total: 0")
        self.ok_lbl = QLabel("Passed: 0")
        self.bad_lbl = QLabel("Failed: 0")
        for lbl in (self.total_lbl, self.ok_lbl, self.bad_lbl):
            lbl.setStyleSheet("font-size: 14px; padding: 6px;")
        self.ok_lbl.setStyleSheet(self.ok_lbl.styleSheet() + "color: #27ae60; font-weight: bold;")
        self.bad_lbl.setStyleSheet(self.bad_lbl.styleSheet() + "color: #e74c3c; font-weight: bold;")
        gl.addWidget(self.total_lbl)
        gl.addWidget(self.ok_lbl)
        gl.addWidget(self.bad_lbl)
        g.setLayout(gl)
        l.addWidget(g)

        # Details list
        g = QGroupBox("Pellet Details")
        gl = QVBoxLayout()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.detail_container = QWidget()
        self.detail_layout = QVBoxLayout()
        self.detail_container.setLayout(self.detail_layout)
        self.scroll.setWidget(self.detail_container)
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

        self.img_label = QLabel("Load a 1275×1754 scanned image to start...")
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet("""
            background: white; border: 4px dashed #95a5a6;
            border-radius: 12px; font-size: 16px; color: #7f8c8d;
        """)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.img_label)
        scroll.setStyleSheet("border: none; background: white;")
        l.addWidget(scroll)
        return w

    def load_model(self, path):
        if not os.path.exists(path):
            QMessageBox.warning(self, "Model Not Found", f"Model not found:\n{path}")
            return
        try:
            self.yolo_detector = YOLO(path)
            QMessageBox.information(self, "Success", f"YOLOv11 model loaded!\nUsing exact size: 1275×1754")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Scanned Image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
        )
        if not path:
            return

        img = cv2_safe_imread(path)
        if img is None:
            QMessageBox.critical(self, "Error", "Cannot read image")
            return

        # Resize exactly to training size (1275×1754) — this is what your model expects
        self.current_image = cv2.resize(img, self.TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        self.show_processed = True
        self.toggle_btn.setText("Show Original")
        self.toggle_btn.setEnabled(True)
        self.process_image()

    def toggle_view(self):
        if self.current_image is None:
            return
        self.show_processed = not self.show_processed
        self.toggle_btn.setText("Show Processed" if self.show_processed else "Show Original")
        img_to_show = self.processed_image if self.show_processed else self.current_image
        self.show_image(img_to_show)

    def process_image(self):
        if self.current_image is None:
            return

        img = self.current_image.copy()
        self.detected_pellets = []

        # YOLO inference at EXACT training resolution
        try:
            results = self.yolo_detector.predict(
                source=self.current_image,
                imgsz=[self.TARGET_H, self.TARGET_W],   # ← Critical: [H, W] format!
                conf=0.1,
                iou=0.45,
                device='cpu',
                verbose=False,
                rect=False,        # No padding
                augment=False
            )
            r = results[0]

            candidates = []
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf.item() * 100

                roi = self.current_image[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                contours = self.cv_detector.detect_pellets(roi)
                if not contours:
                    continue

                best_cnt = max(contours, key=cv2.contourArea)
                best_cnt += np.array([x1, y1])  # offset back

                candidates.append({
                    'contour': best_cnt,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })

            # Simple NMS by confidence
            candidates = sorted(candidates, key=lambda x: -x['confidence'])
            final = []
            iou_thresh = 0.3

            for cand in candidates:
                if any(self._iou(cand['bbox'], f['bbox']) > iou_thresh for f in final):
                    continue
                final.append(cand)

            for i, item in enumerate(final, 1):
                cnt = item['contour']
                conf = item['confidence']
                bbox = item['bbox']

                rect = cv2.minAreaRect(cnt.astype(np.float32))
                width_px, height_px = rect[1]
                diameter_mm = min(width_px, height_px) / self.pixels_per_mm
                length_mm = max(width_px, height_px) / self.pixels_per_mm
                passed = self.d_min <= diameter_mm <= self.d_max and self.l_min <= length_mm <= self.l_max

                pellet = {
                    'id': i,
                    'contour': cnt,
                    'bbox': bbox,
                    'diameter': diameter_mm,
                    'length': length_mm,
                    'confidence': conf,
                    'passed': passed
                }
                self.detected_pellets.append(pellet)
                self.draw_pellet(img, pellet)

        except Exception as e:
            print("YOLO failed, falling back to color:", e)
            contours = self.cv_detector.detect_pellets(self.current_image)
            for i, cnt in enumerate(contours, 1):
                rect = cv2.minAreaRect(cnt.astype(np.float32))
                w, h = rect[1]
                d = min(w, h) / self.pixels_per_mm
                l = max(w, h) / self.pixels_per_mm
                passed = self.d_min <= d <= self.d_max and self.l_min <= l <= self.l_max
                pellet = {
                    'id': i, 'contour': cnt, 'bbox': None,
                    'diameter': d, 'length': l, 'confidence': 100.0, 'passed': passed
                }
                self.detected_pellets.append(pellet)
                self.draw_pellet(img, pellet)

        self.processed_image = img
        self.update_stats()
        self.show_image(img)

    def _iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1b, y1b, x2b, y2b = box2
        xi1 = max(x1, x1b)
        yi1 = max(y1, y1b)
        xi2 = min(x2, x2b)
        yi2 = min(y2, y2b)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2b - x1b) * (y2b - y1b)
        return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0

    def draw_pellet(self, img, p):
        color = (0, 255, 0) if p['passed'] else (0, 0, 255)
        alpha = 0.3

        # Fill contour
        overlay = img.copy()
        cv2.fillPoly(overlay, [p['contour']], color)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Draw min area rectangle
        box = cv2.boxPoints(cv2.minAreaRect(p['contour'].astype(np.float32)))
        box = np.intp(box)
        cv2.drawContours(img, [box], 0, color, 3)

        # YOLO bbox (orange)
        if p['bbox']:
            x1, y1, x2, y2 = p['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(img, f"{p['confidence']:.0f}%", (x1 + 8, y1 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

        # ID + size
        M = cv2.moments(p['contour'])
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = int(p['contour'][:, 0].mean())
            cy = int(p['contour'][:, 1].mean())

        cv2.putText(img, f"{p['id']}", (cx - 20, cy + 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 4)
        cv2.putText(img, f"{p['id']}", (cx - 20, cy + 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)

        size_text = f"{p['diameter']:.2f}x{p['length']:.2f}"
        cv2.putText(img, size_text, (cx - 50, cy + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(img, size_text, (cx - 50, cy + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    def update_stats(self):
        total = len(self.detected_pellets)
        passed = sum(1 for p in self.detected_pellets if p['passed'])
        failed = total - passed

        self.total_lbl.setText(f"Total Pellets: {total}")
        self.ok_lbl.setText(f"Passed: {passed}")
        self.bad_lbl.setText(f"Failed: {failed}")

        # Clear old details
        for i in reversed(range(self.detail_layout.count())):
            child = self.detail_layout.itemAt(i).widget()
            if child:
                child.deleteLater()

        for p in self.detected_pellets:
            status = "PASSED" if p['passed'] else "FAILED"
            color_bg = "#d5f5e6" if p['passed'] else "#fadbd8"
            color_border = "#27ae60" if p['passed'] else "#e74c3c"

            text = (f"<b>Pellet {p['id']}</b> — <span style='color:{color_border};'>{status}</span><br>"
                    f"Diameter: <b>{p['diameter']:.3f} mm</b> | "
                    f"Length: <b>{p['length']:.3f} mm</b><br>"
                    f"Confidence: {p['confidence']:.1f}%")

            lbl = QLabel(text)
            lbl.setStyleSheet(f"""
                background: {color_bg}; border-left: 6px solid {color_border};
                padding: 12px; margin: 4px; border-radius: 8px; font-size: 13px;
            """)
            lbl.setWordWrap(True)
            self.detail_layout.addWidget(lbl)

        self.detail_layout.addStretch()

    def show_image(self, cv_img):
        if cv_img is None:
            return
        pixmap = cv_to_qpixmap(cv_img)
        self.img_label.setPixmap(pixmap)
        self.img_label.setFixedSize(pixmap.size())


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = PelletMeasurementApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()