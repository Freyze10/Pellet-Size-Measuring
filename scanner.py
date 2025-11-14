import sys
import cv2
import numpy as np
import json
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QWidget, QFileDialog, QSlider, QGridLayout,
    QGroupBox, QDoubleSpinBox, QTextEdit, QScrollArea, QFrame,
    QComboBox
)
from PyQt6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent
from PyQt6.QtCore import Qt, QMimeData
from typing import List, Dict, Tuple


# ----------------------------------------------------------------------
# Global Configuration
# ----------------------------------------------------------------------
TARGET_DIAMETER = 3.0
TARGET_LENGTH = 3.0
TOLERANCE = 0.5
EXCLUSION_THRESHOLD = 200.0

MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 10000


class PelletInspectorCOCO(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Inspector - COCO Label Matching")
        self.setGeometry(100, 100, 1500, 900)

        self.image = None
        self.pixmap = None
        self.pixels_per_mm = 6.0
        self.pellets: List[Dict] = []
        self.coco_data = None
        self.categories = {}
        self.images = {}
        self.annotations = []

        self.init_ui()
        self.load_coco_labels("pellets_label.json")

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --------------------- Left Panel ---------------------
        left_panel = QGroupBox("Controls & Analysis")
        left_panel.setFixedWidth(420)
        left_layout = QVBoxLayout(left_panel)

        # Upload
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        left_layout.addWidget(self.upload_btn)

        self.drop_label = QLabel("Or drag & drop image here")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setStyleSheet("QLabel { border: 2px dashed #aaa; padding: 25px; margin: 5px; }")
        left_layout.addWidget(self.drop_label)

        # Calibration
        calib_group = QGroupBox("Scale Calibration")
        calib_layout = QGridLayout(calib_group)
        calib_layout.addWidget(QLabel("Pixels per mm:"), 0, 0)
        self.px_spin = QDoubleSpinBox()
        self.px_spin.setRange(0.1, 100.0)
        self.px_spin.setSingleStep(0.1)
        self.px_spin.setValue(self.pixels_per_mm)
        self.px_spin.valueChanged.connect(self.on_calibration_change)
        calib_layout.addWidget(self.px_spin, 0, 1)

        self.px_slider = QSlider(Qt.Orientation.Horizontal)
        self.px_slider.setRange(1, 1000)
        self.px_slider.setValue(int(self.pixels_per_mm * 10))
        self.px_slider.valueChanged.connect(self.on_slider_change)
        calib_layout.addWidget(self.px_slider, 1, 0, 1, 2)
        left_layout.addWidget(calib_group)

        # COCO Info
        self.coco_info = QLabel("No COCO labels loaded")
        self.coco_info.setWordWrap(True)
        left_layout.addWidget(QGroupBox("Label File", self.coco_info))

        # Statistics
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        self.stats_display.setFixedHeight(300)
        left_layout.addWidget(QGroupBox("Pellet Analysis", self.stats_display))

        # --------------------- Right Panel ---------------------
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_label = QLabel("No image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        self.image_scroll.setWidget(self.image_label)

        # Enable drag drop
        self.setAcceptDrops(True)
        self.drop_label.setAcceptDrops(True)
        self.drop_label.dragEnterEvent = self.dragEnterEvent
        self.drop_label.dropEvent = self.dropEvent

        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.image_scroll, 1)

    # ------------------------------------------------------------------
    # Drag & Drop
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.load_image(files[0])

    # ------------------------------------------------------------------
    # Image & COCO Loading
    # ------------------------------------------------------------------
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path: str):
        self.image = cv2.imread(file_path)
        if self.image is None:
            self.stats_display.setText("Error: Could not load image.")
            return
        self.process_and_display()
        self.update_stats()

    def load_coco_labels(self, json_path: str):
        if not os.path.exists(json_path):
            self.coco_info.setText(f"<span style='color: red;'>Not found:</span> {json_path}")
            return

        try:
            with open(json_path, 'r') as f:
                self.coco_data = json.load(f)

            self.categories = {cat['id']: cat['name'] for cat in self.coco_data.get('categories', [])}
            self.images = {img['id']: img for img in self.coco_data.get('images', [])}
            self.annotations = self.coco_data.get('annotations', [])

            cat_names = ', '.join(self.categories.values()) or "None"
            self.coco_info.setText(
                f"<b>Loaded:</b> {len(self.annotations)} annotations<br>"
                f"<b>Categories:</b> {cat_names}<br>"
                f"<b>Images:</b> {len(self.images)}"
            )
        except Exception as e:
            self.coco_info.setText(f"<span style='color: red;'>Error:</span> {e}")

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def on_calibration_change(self, value):
        self.pixels_per_mm = value
        self.px_slider.blockSignals(True)
        self.px_slider.setValue(int(value * 10))
        self.px_slider.blockSignals(False)
        self.process_and_display()
        self.update_stats()

    def on_slider_change(self, value):
        self.pixels_per_mm = value / 10.0
        self.px_spin.blockSignals(True)
        self.px_spin.setValue(self.pixels_per_mm)
        self.px_spin.blockSignals(False)
        self.process_and_display()
        self.update_stats()

    # ------------------------------------------------------------------
    # COCO Matching
    # ------------------------------------------------------------------
    def get_closest_annotation(self, cx: int, cy: int) -> Dict:
        """Find closest COCO annotation by centroid distance."""
        min_dist = float('inf')
        best = None
        for ann in self.annotations:
            if 'bbox' not in ann:
                continue
            x, y, w, h = ann['bbox']
            ann_cx, ann_cy = x + w // 2, y + h // 2
            dist = np.hypot(cx - ann_cx, cy - ann_cy)
            if dist < min_dist:
                min_dist = dist
                best = ann
        return best if min_dist < 80 else None  # threshold

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect_pellets(self, img) -> List[Dict]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pellets = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA):
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            diameter_px = min(w, h)
            length_px = max(w, h)

            diameter_mm = diameter_px / self.pixels_per_mm
            length_mm = length_px / self.pixels_per_mm

            if not (TARGET_DIAMETER - EXCLUSION_THRESHOLD <= diameter_mm <= TARGET_DIAMETER + EXCLUSION_THRESHOLD):
                continue

            # Centroid
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2

            # Match to COCO label
            ann = self.get_closest_annotation(cx, cy)
            category_name = self.categories.get(ann['category_id'], "Unknown") if ann else "Unlabeled"

            within_tol = (
                TARGET_DIAMETER - TOLERANCE <= diameter_mm <= TARGET_DIAMETER + TOLERANCE and
                TARGET_LENGTH - TOLERANCE <= length_mm <= TARGET_LENGTH + TOLERANCE
            )

            pellets.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'diameter': round(diameter_mm, 3),
                'length': round(length_mm, 3),
                'within_tolerance': within_tol,
                'category': category_name,
                'annotation': ann,
                'contour': cnt
            })
        return pellets

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def process_and_display(self):
        if self.image is None:
            return

        display_img = self.image.copy()
        self.pellets = self.detect_pellets(display_img)

        # Draw
        for p in self.pellets:
            x, y, w, h = p['x'], p['y'], p['w'], p['h']
            color = (0, 255, 0) if p['within_tolerance'] else (0, 0, 255)
            cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 2)

            # Draw COCO polygon if exists
            if p['annotation'] and 'segmentation' in p['annotation']:
                for seg in p['annotation']['segmentation']:
                    poly = np.array(seg, np.int32).reshape(-1, 2)
                    cv2.polylines(display_img, [poly], True, (255, 255, 0), 2)
                    overlay = display_img.copy()
                    cv2.fillPoly(overlay, [poly], (255, 255, 0))
                    cv2.addWeighted(overlay, 0.2, display_img, 0.8, 0, display_img)

            # Label box
            label = f"{p['category']}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            bg_y = y - 60
            cv2.rectangle(display_img, (x, bg_y), (x + label_size[0] + 10, bg_y + 25), (0, 0, 0), -1)
            cv2.putText(display_img, label, (x + 5, bg_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Measurements
            bg_y2 = max(y - 55, 0)
            cv2.rectangle(display_img, (x, bg_y2), (x + 130, y - 5), (0, 0, 0), -1)
            cv2.putText(display_img, f"D: {p['diameter']}", (x + 5, bg_y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(display_img, f"L: {p['length']}", (x + 5, bg_y2 + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            if not p['within_tolerance']:
                cv2.circle(display_img, (x + w - 12, y + 12), 9, (0, 0, 255), -1)
                cv2.putText(display_img, "!", (x + w - 16, y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Convert to QPixmap
        height, width, _ = display_img.shape
        q_img = QImage(display_img.data, width, height, 3 * width, QImage.Format.Format_BGR888)
        self.pixmap = QPixmap.fromImage(q_img)
        scaled = self.pixmap.scaled(
            self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap:
            scaled = self.pixmap.scaled(
                self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def update_stats(self):
        if not self.pellets:
            self.stats_display.setHtml("<p style='color: gray;'>No pellets detected.</p>")
            return

        total = len(self.pellets)
        within = sum(1 for p in self.pellets if p['within_tolerance'])
        out = total - within

        from collections import Counter
        cat_count = Counter(p['category'] for p in self.pellets)

        html = f"""
        <b style="font-size: 13px;">Pellet Summary</b><br>
        <b>Total:</b> {total}  <b style="color: green;">In Tolerance:</b> {within}  <b style="color: red;">Out:</b> {out}<br>
        <b>Calibration:</b> {self.pixels_per_mm:.2f} px/mm<br>
        <hr>
        <b>By Category:</b><br>
        """
        for cat, count in cat_count.items():
            color = "green" if cat != "Unlabeled" else "gray"
            html += f"• <span style='color: {color};'><b>{cat}:</b> {count}</span><br>"

        html += "<hr><b>Individual Pellets:</b><br>"
        for i, p in enumerate(self.pellets):
            status = "OK" if p['within_tolerance'] else "BAD"
            color = "green" if p['within_tolerance'] else "red"
            html += f"<span style='color: {color};'>#{i+1} [{p['category']}] D:{p['diameter']} L:{p['length']} <b>{status}</b></span><br>"

        self.stats_display.setHtml(html)


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = PelletInspectorCOCO()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()