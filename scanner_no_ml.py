import sys
import json
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSpinBox, QDoubleSpinBox, QGroupBox, QScrollArea,
                             QProgressBar, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage


class PelletDetector:
    """Contour-based detector using geometric heuristics and filtering."""

    def __init__(self, min_area=500, max_area=15000, aspect_ratio_max=4.0):
        # Removed SIFT and trained_samples - no training is done.
        # Set default geometric limits for filtering contours
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_max = aspect_ratio_max

    # --------------------------------------------------------------------- #
    # <<<--- REMOVED: train_from_coco and feature_detector/trained_samples ---
    # --------------------------------------------------------------------- #
    def train_from_coco(self, coco_data, images_folder=""):
        """Dummy method - no training is performed in this version."""
        print("Detector uses heuristics, no training required.")
        return True

    def detect_pellets(self, image):
        """Detect pellets in a new image using contour and shape analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Now only relies on the contour method
        detections = self.detect_by_contours(image, gray)

        # Non-max suppression still useful to clean up multiple hits
        detections = self.non_max_suppression(detections)
        return detections

    def detect_by_contours(self, image, gray):
        detections = []

        # Image pre-processing for better contour finding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtering contours based on heuristic geometric limits
        for contour in contours:
            area = cv2.contourArea(contour)

            # 1. Area filtering (must be within reasonable limits for a pellet)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 0

                # 2. Aspect Ratio filtering (pellets are typically not long and thin)
                if 1.0 <= aspect <= self.aspect_ratio_max:
                    # Since we removed the ML-based comparison, we use a fixed, high confidence score
                    match_score = 0.95

                    detections.append({
                        'bbox': (x, y, w, h),
                        'polygon': contour.reshape(-1, 2),
                        'confidence': match_score * 100,
                        'method': 'geometric_contour'
                    })
        return detections

    # --------------------------------------------------------------------- #
    # <<<--- REMOVED: compare_with_samples (depended on trained data) ---
    # --------------------------------------------------------------------- #
    def compare_with_samples(self, contour):
        """Dummy method for compatibility, returns fixed confidence."""
        return 0.95

    # non_max_suppression remains unchanged
    def non_max_suppression(self, detections, overlap_threshold=0.5):
        if not detections:
            return []
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []
        for det in detections:
            x1, y1, w1, h1 = det['bbox']
            duplicate = False
            for kept in keep:
                x2, y2, w2, h2 = kept['bbox']
                xi1 = max(x1, x2);
                yi1 = max(y1, y2)
                xi2 = min(x1 + w1, x2 + w2);
                yi2 = min(y1 + h1, y2 + h2)
                inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                union = w1 * h1 + w2 * h2 - inter
                iou = inter / union if union > 0 else 0
                if iou > overlap_threshold:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(det)
        return keep


# =============================================================================
class PelletMeasurementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Size Measurement System - Heuristic Contour")
        self.setGeometry(100, 100, 1400, 800)

        # ---- configuration -------------------------------------------------
        self.pixels_per_mm = 6.0
        self.target_diameter = 3.0
        self.target_length = 3.0
        self.tolerance = 0.5

        # ---- data -----------------------------------------------------------
        self.current_image = None
        self.current_image_path = None
        self.detected_pellets = []
        # Removed: self.coco_data, self.images_folder

        # ---- detector -------------------------------------------------------
        # Initialize detector with heuristic parameters
        self.detector = PelletDetector(min_area=500, max_area=15000, aspect_ratio_max=4.0)
        self.is_trained = True  # Always True, as no training is needed

        self.init_ui()
        # Removed: self.load_and_train() - direct initialization

    # Rotated and measure methods remain unchanged as they are purely geometry-based
    def rotated_rect_dimensions(self, polygon):
        """Return (width_px, height_px) of the rotated min-area rectangle."""
        rect = cv2.minAreaRect(polygon.astype(np.float32))  # ((cx,cy),(w,h),angle)
        return rect[1]  # (w, h)

    def measure_pellet(self, polygon, pixels_per_mm):
        """Return diameter/length in mm using the exact mask."""
        w_px, h_px = self.rotated_rect_dimensions(polygon)

        width_mm = w_px / pixels_per_mm
        height_mm = h_px / pixels_per_mm

        diameter = min(width_mm, height_mm)
        length = max(width_mm, height_mm)

        within = (
                (self.target_diameter - self.tolerance <= diameter <= self.target_diameter + self.tolerance) and
                (self.target_length - self.tolerance <= length <= self.target_length + self.tolerance)
        )
        return {"diameter": diameter, "length": length, "within": within}

    # --------------------------------------------------------------------- #

    # --------------------------------------------------------------------- #
    # <<<--- REMOVED: load_and_train (no training data used) ---
    # --------------------------------------------------------------------- #

    # --------------------------------------------------------------------- #
    # UI (Modified to reflect non-training status)
    # --------------------------------------------------------------------- #
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        left = self.create_left_panel()
        right = self.create_right_panel()
        main_layout.addWidget(left, 1)
        main_layout.addWidget(right, 3)

    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # ---- status label (modified) ----------------------------------------
        self.progress_label = QLabel("Detector initialized with geometric heuristics.")
        self.progress_label.setStyleSheet("padding: 5px; background-color: #e0e0e0;")
        layout.addWidget(self.progress_label)

        # ---- load image (enabled by default) --------------------------------
        self.load_btn = QPushButton("Load Pellet Image for Detection")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setStyleSheet("QPushButton { padding: 10px; font-size: 14px; }")
        self.load_btn.setEnabled(True)  # Always enabled now
        layout.addWidget(self.load_btn)

        # ---- calibration ----------------------------------------------------
        calib = QGroupBox("Calibration")
        calib_l = QVBoxLayout()
        px_l = QHBoxLayout()
        px_l.addWidget(QLabel("Pixels per mm:"))
        self.px_spinbox = QDoubleSpinBox()
        self.px_spinbox.setRange(0.1, 100.0)
        self.px_spinbox.setValue(self.pixels_per_mm)
        self.px_spinbox.setSingleStep(0.1)
        self.px_spinbox.setDecimals(2)
        self.px_spinbox.valueChanged.connect(self.update_calibration)
        px_l.addWidget(self.px_spinbox)
        calib_l.addLayout(px_l)
        calib.setLayout(calib_l)
        layout.addWidget(calib)

        # ---- target specs (unchanged) ---------------------------------------
        spec = QGroupBox("Target Specifications")
        spec_l = QVBoxLayout()
        spec_l.addWidget(QLabel(f"Target Diameter: {self.target_diameter} mm"))
        spec_l.addWidget(QLabel(f"Target Length: {self.target_length} mm"))
        spec_l.addWidget(QLabel(f"Tolerance: ±{self.tolerance} mm"))
        spec_l.addWidget(QLabel(
            f"Acceptable: {self.target_diameter - self.tolerance:.1f}-"
            f"{self.target_diameter + self.tolerance:.1f} mm"))
        spec.setLayout(spec_l)
        layout.addWidget(spec)

        # ---- statistics (unchanged) -----------------------------------------
        self.stats_group = QGroupBox("Detection Statistics")
        self.stats_layout = QVBoxLayout()
        self.total_label = QLabel("Total Pellets: 0")
        self.within_label = QLabel("Within Tolerance: 0")
        self.out_label = QLabel("Out of Tolerance: 0")
        self.status_label = QLabel("Status: Detector ready")
        self.status_label.setStyleSheet("font-weight: bold; padding: 5px;")
        for w in (self.total_label, self.within_label, self.out_label, self.status_label):
            self.stats_layout.addWidget(w)
        self.stats_group.setLayout(self.stats_layout)
        layout.addWidget(self.stats_group)

        # ---- pellet details (scrollable) (unchanged) ------------------------
        details = QGroupBox("Pellet Details")
        details_l = QVBoxLayout()
        self.details_scroll = QScrollArea()
        self.details_scroll.setWidgetResizable(True)
        self.details_widget = QWidget()
        self.details_widget_layout = QVBoxLayout()
        self.details_widget.setLayout(self.details_widget_layout)
        self.details_scroll.setWidget(self.details_widget)
        details_l.addWidget(self.details_scroll)
        details.setLayout(details_l)
        layout.addWidget(details)

        layout.addStretch()
        return panel

    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        self.image_label = QLabel("Load an image to start detection")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.image_label.setMinimumSize(800, 600)

        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        return panel

    # --------------------------------------------------------------------- #
    def load_image(self):
        # The check for is_trained is now mostly redundant but harmless
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Pellet Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.current_image_path = path
            self.current_image = cv2.imread(path)
            if self.current_image is not None:
                self.progress_label.setText("Detecting pellets using contour analysis...")
                QApplication.processEvents()
                self.process_image()
                self.progress_label.setText("Detection complete")
            else:
                self.image_label.setText("Error loading image")

    def update_calibration(self, value):
        self.pixels_per_mm = value
        if self.current_image is not None:
            self.process_image()

    def process_image(self):
        if self.current_image is None:
            return

        detections = self.detector.detect_pellets(self.current_image)
        display_img = self.current_image.copy()
        self.detected_pellets = []

        for idx, det in enumerate(detections, start=1):
            polygon = det['polygon']

            # ---- exact measurement from mask --------------------------------
            meas = self.measure_pellet(polygon, self.pixels_per_mm)

            pellet = {
                'polygon': polygon,
                'bbox': det['bbox'],
                'diameter': meas['diameter'],
                'length': meas['length'],
                'within_tolerance': meas['within'],
                'confidence': det['confidence'],
                'id': idx
            }
            self.detected_pellets.append(pellet)
            self.draw_pellet(display_img, pellet)

        self.update_statistics()
        self.display_image(display_img)

    # draw_pellet, update_statistics, update_pellet_details, display_image are unchanged.
    def draw_pellet(self, image, pellet):
        """Draw only the outline + centered ID number."""
        poly = pellet['polygon']
        pid = pellet['id']
        ok = pellet['within_tolerance']
        colour = (0, 255, 0) if ok else (0, 0, 255)

        # semi-transparent fill
        overlay = image.copy()
        cv2.fillPoly(overlay, [poly.reshape(-1, 1, 2)], colour)
        cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)

        # thick outline
        cv2.polylines(image, [poly.reshape(-1, 1, 2)], True, colour, 2)

        # centered number
        M = cv2.moments(poly)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else int(poly[:, 0].mean())
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else int(poly[:, 1].mean())
        cv2.putText(image, str(pid), (cx - 12, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def update_statistics(self):
        total = len(self.detected_pellets)
        within = sum(1 for p in self.detected_pellets if p['within_tolerance'])
        out = total - within

        self.total_label.setText(f"Total Pellets: {total}")
        self.within_label.setText(f"Within Tolerance: {within}")
        self.out_label.setText(f"Out of Tolerance: {out}")

        if total == 0:
            self.status_label.setText("Status: No pellets detected")
            self.status_label.setStyleSheet("font-weight: bold; padding: 5px; color: gray;")
        elif out == 0:
            self.status_label.setText("Status: All Within Tolerance")
            self.status_label.setStyleSheet("font-weight: bold; padding: 5px; color: green;")
        else:
            self.status_label.setText(f"Status: {out} Out of Tolerance")
            self.status_label.setStyleSheet("font-weight: bold; padding: 5px; color: red;")

        self.update_pellet_details()

    def update_pellet_details(self):
        while self.details_widget_layout.count():
            child = self.details_widget_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for i, p in enumerate(self.detected_pellets, 1):
            txt = (f"Pellet {i}:\n"
                   f"  Diameter: {p['diameter']:.2f} mm\n"
                   f"  Length:   {p['length']:.2f} mm\n"
                   f"  Confidence: {p['confidence']:.0f}\n"
                   f"  Status: {'OK' if p['within_tolerance'] else 'Out'}")
            lbl = QLabel(txt)
            lbl.setStyleSheet(
                f"padding:5px;margin:2px;border:1px solid {'green' if p['within_tolerance'] else 'red'};")
            self.details_widget_layout.addWidget(lbl)
        self.details_widget_layout.addStretch()

    def display_image(self, cv_image):
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.image_label.size(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.image_label.setMinimumSize(1, 1)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = PelletMeasurementApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()