import sys
import json
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSpinBox, QDoubleSpinBox, QGroupBox, QScrollArea,
                             QProgressBar, QMessageBox, QSizePolicy)  # QSizePolicy added
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage


# --- ROBUST IMAGE LOADING HELPER FUNCTION (For crash prevention) ---
def cv2_safe_imread(path):
    """
    Reads an image using cv2.imdecode to handle file path/encoding issues that
    can occur with cv2.imread and user file dialog paths, especially with JPEGs.
    """
    try:
        # 1. Read the file data as a raw byte array
        with open(path, 'rb') as f:
            data = f.read()
        # 2. Decode the byte array into an OpenCV image (NumPy array)
        np_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error during safe image read: {e}")
        return None


# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Core Heuristic Parameters from Camera Script
# ----------------------------------------------------------------------
DEFAULT_PIXELS_PER_MM = 10.0
DEFAULT_TARGET_DIAMETER = 3.0
DEFAULT_TARGET_LENGTH = 3.0
DEFAULT_TOLERANCE = 0.05  # Reduced tolerance for the 3.0mm specification
MIN_CONTOUR_AREA_HEURISTIC = 100
MAX_CONTOUR_AREA_HEURISTIC = 10000

# HSV Color Range for the Blue Pellets (Optimized for the sample image)
LOWER_BLUE = np.array([100, 150, 50])
UPPER_BLUE = np.array([140, 255, 255])


class PelletDetector:
    """Contour-based detector using geometric heuristics and **color filtering**."""

    def __init__(self, min_area=MIN_CONTOUR_AREA_HEURISTIC, max_area=MAX_CONTOUR_AREA_HEURISTIC, aspect_ratio_max=4.0):
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_max = aspect_ratio_max

    def train_from_coco(self, coco_data, images_folder=""):
        """Dummy method - no training is performed in this version."""
        print("Detector uses heuristics, no training required.")
        return True

    def detect_pellets(self, image):
        """
        Detect pellets using robust HSV color thresholding instead of generic
        adaptive thresholding to better handle the scanned image background.
        """
        # 1. Convert to HSV for color-based segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 2. Create a mask for the blue pellets
        mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

        # 3. Clean up the mask using morphology (Closing operation)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        processed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 4. Find contours
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        # Filtering contours based on size and shape limits
        for contour in contours:
            area = cv2.contourArea(contour)

            # 1. Area filtering
            if self.min_area <= area <= self.max_area:

                # Get the minimum area bounding box for accurate L/W regardless of rotation
                rect = cv2.minAreaRect(contour)
                w, h = rect[1]

                # Aspect Ratio filtering (L/W)
                aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 0

                if 1.0 <= aspect <= self.aspect_ratio_max:
                    # Use the standard bounding rect for the bbox output (simpler for display)
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    match_score = 0.95

                    detections.append({
                        'bbox': (x, y, w_box, h_box),
                        'polygon': contour.reshape(-1, 2),
                        'confidence': match_score * 100,
                        'method': 'color_contour'
                    })

        # Non-max suppression still useful to clean up multiple hits
        detections = self.non_max_suppression(detections)
        return detections

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
        self.setWindowTitle("Pellet Size Measurement System - Orientation-Independent Measurement")
        self.setGeometry(100, 100, 1400, 800)

        # ---- configuration: Using Camera Script Defaults --------------------
        self.pixels_per_mm = DEFAULT_PIXELS_PER_MM
        self.target_diameter = DEFAULT_TARGET_DIAMETER
        self.target_length = DEFAULT_TARGET_LENGTH
        self.tolerance = DEFAULT_TOLERANCE
        self.diameter_min, self.diameter_max = 0, 0
        self.length_min, self.length_max = 0, 0
        self.update_ranges()

        # ---- data -----------------------------------------------------------
        self.current_image = None
        self.current_image_path = None
        self.detected_pellets = []

        # ---- detector -------------------------------------------------------
        self.detector = PelletDetector()
        self.is_trained = True  # Always True for heuristic-based detector

        self.init_ui()

    def update_ranges(self):
        """Recalculate tolerance ranges based on current settings."""
        self.diameter_min = self.target_diameter - self.tolerance
        self.diameter_max = self.target_diameter + self.tolerance
        self.length_min = self.target_length - self.tolerance
        self.length_max = self.target_length + self.tolerance

    # Rotated and measure methods (Orientation-Independent Measurement)
    def rotated_rect_dimensions(self, polygon):
        """Return (width_px, height_px) of the rotated min-area rectangle."""
        # Note: cv2.minAreaRect is the key to orientation-independent measurement.
        # It finds the smallest bounding box, giving accurate L/W.
        rect = cv2.minAreaRect(polygon.astype(np.float32))  # ((cx,cy),(w,h),angle)
        return rect[1]  # (w, h)

    def measure_pellet(self, polygon, pixels_per_mm):
        """Return diameter/length in mm using the exact mask."""
        w_px, h_px = self.rotated_rect_dimensions(polygon)

        width_mm = w_px / pixels_per_mm
        height_mm = h_px / pixels_per_mm

        # For a cylindrical pellet:
        # Diameter is the smaller dimension (min)
        # Length is the larger dimension (max)
        diameter = min(width_mm, height_mm)
        length = max(width_mm, height_mm)

        # Check tolerance for both dimensions
        within = (
                (self.diameter_min <= diameter <= self.diameter_max) and
                (self.length_min <= length <= self.length_max)
        )

        # Determine status string
        if within:
            status = "IN RANGE"
        elif diameter < self.diameter_min or length < self.length_min:
            status = "UNDERSIZED"
        elif diameter > self.diameter_max or length > self.length_max:
            status = "OVERSIZED"
        else:
            status = "OUT OF SPEC"

        return {"diameter": diameter, "length": length, "within": within, "status": status}

    # --------------------------------------------------------------------- #

    # --------------------------------------------------------------------- #
    # UI (Modified to reflect non-training status and use current ranges)
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

        # ---- status label ----------------------------------------
        self.progress_label = QLabel("Detector initialized with color-based heuristics.")
        self.progress_label.setStyleSheet("padding: 5px; background-color: #e0e0e0;")
        layout.addWidget(self.progress_label)

        # ---- load image ------------------------------------------------
        self.load_btn = QPushButton("Load Pellet Image for Detection")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setStyleSheet("QPushButton { padding: 10px; font-size: 14px; }")
        self.load_btn.setEnabled(True)
        layout.addWidget(self.load_btn)

        # ---- calibration ----------------------------------------------------
        calib = QGroupBox("Calibration (Pixels per MM)")
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

        # ---- target specs (dynamic labels) ----------------------------------
        self.spec_group = QGroupBox("Target Specifications")
        self.spec_l = QVBoxLayout()

        self.target_d_label = QLabel()
        self.target_l_label = QLabel()
        self.tolerance_label = QLabel()
        self.acceptable_range_label = QLabel()

        for w in (self.target_d_label, self.target_l_label, self.tolerance_label, self.acceptable_range_label):
            self.spec_l.addWidget(w)

        self.spec_group.setLayout(self.spec_l)
        layout.addWidget(self.spec_group)
        self.update_spec_labels()

        # ---- statistics, details, etc. ---------------------------------
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
        layout.addWidget(details, stretch=1)

        layout.addStretch()
        return panel

    def update_spec_labels(self):
        """Updates the labels in the Target Specifications group box."""
        self.target_d_label.setText(f"Target Diameter: {self.target_diameter} mm")
        self.target_l_label.setText(f"Target Length: {self.target_length} mm")
        self.tolerance_label.setText(f"Tolerance: ±{self.tolerance} mm")
        self.acceptable_range_label.setText(
            f"Acceptable D/L: [{self.diameter_min:.2f} - "
            f"{self.diameter_max:.2f}] mm")

    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        self.image_label = QLabel("Load an image to start detection")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.image_label.setMinimumSize(400, 400)  # Smaller minimum size
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )

        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        return panel

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Pellet Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.current_image_path = path
            # Use the robust image loading function
            self.current_image = cv2_safe_imread(path)

            if self.current_image is not None:
                self.progress_label.setText("Detecting pellets using color-based heuristics...")
                QApplication.processEvents()
                self.process_image()
                self.progress_label.setText("Detection complete")
            else:
                QMessageBox.critical(self, "Image Load Error",
                                     f"Failed to load image at {os.path.basename(path)}.")
                self.image_label.setText("Error loading image")
                self.current_image = None
                self.update_statistics()  # Clear stats

    def update_calibration(self, value):
        self.pixels_per_mm = value
        self.update_ranges()
        self.update_spec_labels()
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

            # ---- exact measurement from mask (Orientation-Independent) ------
            meas = self.measure_pellet(polygon, self.pixels_per_mm)

            pellet = {
                'polygon': polygon,
                'bbox': det['bbox'],
                'diameter': meas['diameter'],
                'length': meas['length'],
                'within_tolerance': meas['within'],
                'status': meas['status'],
                'confidence': det['confidence'],
                'id': idx
            }
            self.detected_pellets.append(pellet)
            self.draw_pellet(display_img, pellet)

        self.update_statistics()
        self.display_image(display_img)

    def draw_pellet(self, image, pellet):
        """Draw only the outline + centered ID number."""
        poly = pellet['polygon']
        pid = pellet['id']
        ok = pellet['within_tolerance']

        # Get rotated bounding box for accurate drawing, especially on rotated pellets
        rect = cv2.minAreaRect(poly.astype(np.float32))
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        colour = (0, 255, 0) if ok else (0, 0, 255)  # Green or Red

        # semi-transparent fill
        overlay = image.copy()
        cv2.fillPoly(overlay, [poly.reshape(-1, 1, 2)], colour)
        cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)

        # thick outline (using the minimal bounding box)
        cv2.drawContours(image, [box], 0, colour, 2)

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
            txt = (f"Pellet {i} - Status: {p['status']}\n"
                   f"  Diameter (W): {p['diameter']:.3f} mm\n"
                   f"  Length (L):   {p['length']:.3f} mm\n"
                   f"  Confidence: {p['confidence']:.0f}%")

            style = "green" if p['within_tolerance'] else "red"
            if p['status'] == "OUT OF SPEC": style = "orange"  # Optional: for generic out of spec

            lbl = QLabel(txt)
            lbl.setStyleSheet(
                f"padding:5px;margin:2px;border:1px solid {style};")
            self.details_widget_layout.addWidget(lbl)

        self.details_widget_layout.addStretch()

    def display_image(self, cv_image):
        # 1. Convert BGR (OpenCV default) to RGB (Qt default for this format)
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        # 2. Robust QImage creation (Ensures memory stability)
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        # Scaling for display: Use the label's current size, with a minimum fallback size.
        width = max(self.image_label.width(), 400)
        height = max(self.image_label.height(), 400)

        scaled = pix.scaled(width, height,
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