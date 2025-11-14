import sys
import cv2
import numpy as np
import json
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QWidget, QFileDialog, QSlider, QGridLayout,
    QGroupBox, QDoubleSpinBox, QTextEdit, QFrame, QScrollArea
)
from PyQt6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent
from PyQt6.QtCore import Qt
from typing import List, Dict, Optional


# ----------------------------------------------------------------------
# Global Configuration
# ----------------------------------------------------------------------
TARGET_DIAMETER = 3.0
TARGET_LENGTH   = 3.0
TOLERANCE       = 0.5
EXCLUSION_THRESHOLD = 200.0
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 10000


class PelletInspector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Inspector with COCO Reference Polygons")
        self.setGeometry(100, 100, 1500, 900)

        self.image = None
        self.image_path = None
        self.pixels_per_mm = 6.0
        self.pellets: List[Dict] = []
        self.reference_polygons = []
        self.coco_images = {}

        self.init_ui()
        self.update_calibration_ranges()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --------------------- Left Panel ---------------------
        left_panel = QGroupBox("Controls & Analysis")
        left_panel.setFixedWidth(420)
        left_layout = QVBoxLayout(left_panel)

        # Upload Image Button
        self.upload_img_btn = QPushButton("Upload Image")
        self.upload_img_btn.clicked.connect(self.upload_image)
        left_layout.addWidget(self.upload_img_btn)

        # Upload JSON Button
        self.upload_json_btn = QPushButton("Load COCO Labels (pellets_label.json)")
        self.upload_json_btn.clicked.connect(self.load_coco_json)
        left_layout.addWidget(self.upload_json_btn)

        # Drag & Drop Area
        self.drop_label = QLabel("Or drag & drop image / JSON here")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setStyleSheet("QLabel { border: 2px dashed #aaa; padding: 20px; margin: 10px; }")
        left_layout.addWidget(self.drop_label)

        # Calibration Group
        calib_group = QGroupBox("Calibration")
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

        # Reference Info
        self.ref_info = QLabel("No reference polygons loaded")
        self.ref_info.setStyleSheet("color: #555; font-style: italic;")
        left_layout.addWidget(self.ref_info)

        # Statistics
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        self.stats_display.setFixedHeight(200)
        self.stats_display.setStyleSheet("font-family: Consolas; font-size: 11px;")
        stats_group = QGroupBox("Pellet Analysis")
        stats_group_layout = QVBoxLayout(stats_group)
        stats_group_layout.addWidget(self.stats_display)
        left_layout.addWidget(stats_group)

        left_layout.addStretch()

        # --------------------- Right Panel: Scrollable Image ---------------------
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setMinimumSize(800, 600)
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f8f8f8;")
        self.image_scroll.setWidget(self.image_label)

        # Enable drag & drop
        self.setAcceptDrops(True)
        self.drop_label.setAcceptDrops(True)
        self.drop_label.dragEnterEvent = self.dragEnterEvent
        self.drop_label.dropEvent = self.dropEvent

        # Layout
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
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                self.load_image(f)
                break
            elif f.lower().endswith('.json'):
                self.load_coco_json(f)
                break

    # ------------------------------------------------------------------
    # Load Image
    # ------------------------------------------------------------------
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path: str):
        self.image_path = file_path
        self.image = cv2.imread(file_path)
        if self.image is None:
            self.show_error("Could not load image.")
            return
        self.process_and_display()
        self.update_stats()

    # ------------------------------------------------------------------
    # Load COCO JSON
    # ------------------------------------------------------------------
    def load_coco_json(self, json_path=None):
        if json_path is None:
            json_path, _ = QFileDialog.getOpenFileName(
                self, "Open COCO JSON", "", "JSON Files (*.json)"
            )
        if not json_path or not os.path.exists(json_path):
            self.show_error("JSON file not found.")
            return

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            self.coco_images = {img['id']: img for img in data.get('images', [])}
            annotations = data.get('annotations', [])
            self.reference_polygons = []

            for ann in annotations:
                if 'segmentation' in ann and ann['segmentation']:
                    for seg in ann['segmentation']:
                        if len(seg) < 6:
                            continue
                        polygon = np.array(seg).reshape(-1, 2).astype(np.int32)
                        self.reference_polygons.append({
                            'polygon': polygon,
                            'bbox': ann.get('bbox', []),
                            'category_id': ann.get('category_id', 1)
                        })

            self.ref_info.setText(f"{len(self.reference_polygons)} reference polygons loaded")
            print(f"Loaded {len(self.reference_polygons)} polygons from {os.path.basename(json_path)}")

            if self.image is not None:
                self.process_and_display()
                self.update_stats()

        except Exception as e:
            self.show_error(f"Error loading JSON: {e}")

    def show_error(self, msg):
        self.stats_display.setHtml(f"<p style='color: red;'><b>Error:</b> {msg}</p>")

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def update_calibration_ranges(self):
        self.d_min = TARGET_DIAMETER - TOLERANCE
        self.d_max = TARGET_DIAMETER + TOLERANCE
        self.l_min = TARGET_LENGTH - TOLERANCE
        self.l_max = TARGET_LENGTH + TOLERANCE
        self.ex_d_min = TARGET_DIAMETER - EXCLUSION_THRESHOLD
        self.ex_d_max = TARGET_DIAMETER + EXCLUSION_THRESHOLD
        self.ex_l_min = TARGET_LENGTH - EXCLUSION_THRESHOLD
        self.ex_l_max = TARGET_LENGTH + EXCLUSION_THRESHOLD

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
    # Match Contour to Reference Polygon
    # ------------------------------------------------------------------
    def match_to_reference(self, contour) -> Optional[Dict]:
        if not self.reference_polygons:
            return None

        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        min_dist = float('inf')
        closest = None

        for poly in self.reference_polygons:
            ref_M = cv2.moments(poly['polygon'])
            if ref_M["m00"] == 0:
                continue
            ref_cx = int(ref_M["m10"] / ref_M["m00"])
            ref_cy = int(ref_M["m01"] / ref_M["m00"])
            dist = np.sqrt((cx - ref_cx)**2 + (cy - ref_cy)**2)
            if dist < min_dist:
                min_dist = dist
                closest = poly

        return closest if min_dist < 60 else None

    # ------------------------------------------------------------------
    # Detection + Drawing
    # ------------------------------------------------------------------
    def detect_and_draw(self, img) -> List[Dict]:
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
            d_px = min(w, h)
            l_px = max(w, h)
            diameter = d_px / self.pixels_per_mm
            length = l_px / self.pixels_per_mm

            if not (self.ex_d_min <= diameter <= self.ex_d_max and
                    self.ex_l_min <= length <= self.ex_l_max):
                continue

            within_tol = (self.d_min <= diameter <= self.d_max and
                          self.l_min <= length <= self.l_max)

            ref_poly = self.match_to_reference(cnt)

            pellets.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'diameter': round(diameter, 3),
                'length': round(length, 3),
                'within_tolerance': within_tol,
                'reference_polygon': ref_poly
            })

            # Draw reference polygon (if matched)
            if ref_poly:
                poly = ref_poly['polygon']
                overlay = img.copy()
                cv2.fillPoly(overlay, [poly], (255, 255, 0))
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                cv2.polylines(img, [poly], True, (0, 255, 255), 2)

            # Bounding box
            color = (0, 255, 0) if within_tol else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # Label
            bg_y = max(y - 50, 0)
            cv2.rectangle(img, (x, bg_y), (x + 130, y - 5), (0, 0, 0), -1)
            cv2.putText(img, f"D: {diameter:.3f}", (x + 5, bg_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, f"L: {length:.3f}", (x + 5, bg_y + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if not within_tol:
                cv2.circle(img, (x + w - 12, y + 12), 10, (0, 0, 255), -1)
                cv2.putText(img, "!", (x + w - 18, y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return pellets

    def process_and_display(self):
        if self.image is None:
            return

        display_img = self.image.copy()
        self.pellets = self.detect_and_draw(display_img)

        # Convert to QPixmap
        h, w, ch = display_img.shape
        bytes_per_line = ch * w
        q_img = QImage(display_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)

        # Scale to fit
        label_size = self.image_label.size()
        scaled = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.image_label.setMinimumSize(1, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'image_label') and self.image_label.pixmap():
            pixmap = self.image_label.pixmap()
            scaled = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled)

    # ------------------------------------------------------------------
    # Update Statistics
    # ------------------------------------------------------------------
    def update_stats(self):
        if not self.pellets:
            self.stats_display.setHtml("<p style='color: gray;'>No pellets detected.</p>")
            return

        total = len(self.pellets)
        within = sum(1 for p in self.pellets if p['within_tolerance'])
        out = total - within
        matched = sum(1 for p in self.pellets if p['reference_polygon'])

        html = f"""
        <b style="color: #222;">Pellet Analysis</b><br>
        <b>Total Detected:</b> {total}<br>
        <b style="color: green;">Within Tolerance:</b> {within}<br>
        <b style="color: red;">Out of Tolerance:</b> {out}<br>
        <b>Matched to Reference:</b> {matched}/{len(self.reference_polygons)}<br>
        <hr>
        <small>
        <b>Target:</b> D={TARGET_DIAMETER}±{TOLERANCE}mm, L={TARGET_LENGTH}±{TOLERANCE}mm<br>
        <b>Calibration:</b> {self.pixels_per_mm:.2f} px/mm
        </small>
        <hr>
        """

        for i, p in enumerate(self.pellets):
            status = "OK" if p['within_tolerance'] else "BAD"
            color = "green" if p['within_tolerance'] else "red"
            ref = " (Ref)" if p['reference_polygon'] else ""
            html += f"<span style='color: {color};'>#{i+1}: D={p['diameter']} L={p['length']} [{status}]{ref}</span><br>"

        self.stats_display.setHtml(html)


# ----------------------------------------------------------------------
# Run App
# ----------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = PelletInspector()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()