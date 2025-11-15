import cv2
import numpy as np
import pandas as pd
import json
import os
from PyQt6.QtCore import (Qt, QAbstractTableModel, QModelIndex, QVariant)
from PyQt6.QtGui import (QPixmap, QImage, QColor)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QTableView, QHeaderView, QFileDialog, QPushButton,
                             QSplitter, QGridLayout)


# --- 1. COMPUTER VISION AND MEASUREMENT LOGIC ---

class PelletAnalyzer:
    """
    Handles all image processing, calibration, detection, and measurement.
    """

    # Target and Tolerance Specifications
    TARGET_SIZE_MM = 3.00
    TOLERANCE_MM = 0.05
    MIN_VALID_SIZE_MM = 2.95
    MAX_VALID_SIZE_MM = 3.05

    # Exclusion Thresholds (to ignore large debris/calibration strip and tiny noise)
    # These are initial pixel-based estimates based on 300 DPI (3.00mm ~ 35.4px)
    # They will be converted to mm-based checks after calibration.
    MIN_PIXEL_AREA_THRESHOLD = 500  # Minimum contour area (px^2) to be considered a pellet
    MAX_PIXEL_AREA_THRESHOLD = 15000  # Maximum contour area (px^2) to exclude large debris

    # HSV Color Range for the Blue Pellets (tuned for the sample image's bright blue)
    # Hue range for blue in OpenCV is around 100-140 (out of 180)
    LOWER_BLUE = np.array([100, 150, 50])
    UPPER_BLUE = np.array([140, 255, 255])

    # Calibration Strip Definition (for demonstration - assumes a black strip on the left edge)
    CALIBRATION_MM = 50.0  # Assumed real-world size (e.g., width of a 50mm strip)
    CALIBRATION_HUE_RANGE = np.array([0, 0, 0]), np.array([180, 255, 100])  # Dark/Black object
    CALIBRATION_CROP = (
    0, 0, 300, 3000)  # (x_min, y_min, x_max, y_max) to look in the far left of the image (300px wide strip)

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.px_per_mm = 0.0
        self.results_df = pd.DataFrame()
        self.annotated_image = None

    def _calibrate_system(self):
        """
        Detects a large, non-blue calibration object (simulated dark strip) and calculates px/mm.
        Returns: px_per_mm (float)
        """
        if self.image is None:
            return 0.0

        # 1. Crop to the calibration area (e.g., far left side)
        x_min, y_min, x_max, y_max = self.CALIBRATION_CROP
        cal_area = self.image[y_min:y_max, x_min:x_max]

        if cal_area.size == 0:
            print("Calibration area is out of bounds or zero size.")
            # Fallback to standard 300 DPI if live calibration fails
            return 300.0 / 25.4

        # 2. Convert to HSV for robust color thresholding (looking for a dark object)
        hsv_cal = cv2.cvtColor(cal_area, cv2.COLOR_BGR2HSV)
        mask_cal = cv2.inRange(hsv_cal, self.CALIBRATION_HUE_RANGE[0], self.CALIBRATION_HUE_RANGE[1])

        # 3. Apply morphology to connect the strip (Erosion followed by Dilation - Open operation)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        mask_cal = cv2.morphologyEx(mask_cal, cv2.MORPH_OPEN, kernel, iterations=2)

        # 4. Find the largest contour (the calibration strip)
        contours, _ = cv2.findContours(mask_cal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"Calibration strip not detected. Assuming {300 / 25.4:.2f} px/mm (300 DPI).")
            # Fallback to standard 300 DPI if live calibration fails
            return 300.0 / 25.4

        # Find the contour with the maximum area
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding rectangle of the largest contour
        # x, y, w, h = cv2.boundingRect(largest_contour)

        # Use a rotated bounding box for more accurate measurement
        rect = cv2.minAreaRect(largest_contour)
        width_px = max(rect[1])  # Width is the larger side

        # The calibration strip's true dimension is a known value (e.g., 50.0 mm)
        self.px_per_mm = width_px / self.CALIBRATION_MM
        print(
            f"Live Calibration successful: {self.px_per_mm:.2f} px/mm based on a {self.CALIBRATION_MM}mm strip ({width_px:.2f} pixels)")
        return self.px_per_mm

    def _process_pellet(self, contour, px_per_mm):
        """
        Measures a single pellet contour and determines its status.
        """
        # Get the minimum area bounding box (rotated rectangle) for accurate dimensions
        rect = cv2.minAreaRect(contour)
        (x_c, y_c), (w_px, h_px), angle = rect

        # The sides of the MinAreaRect are not guaranteed to be ordered by size.
        # For a cylindrical pellet, the smaller dimension is the 'diameter' (Width)
        # and the larger dimension is the 'height'/'length' (Height), or vice versa.
        # We'll treat the min as Width and max as Height.
        w_px, h_px = min(w_px, h_px), max(w_px, h_px)

        # Convert to mm
        w_mm = w_px / px_per_mm
        h_mm = h_px / px_per_mm

        # Use the standard bounding box for X/Y position
        x_pos, y_pos, _, _ = cv2.boundingRect(contour)

        # 1. Exclusion Rule (Ignore anything significantly too large or too small)
        # Check against a tighter window around 3.00mm (e.g., 1.0mm to 5.0mm)
        # Assuming no valid pellet will be < 1.0mm or > 5.0mm
        MIN_EXCLUSION_MM = 1.0
        MAX_EXCLUSION_MM = 5.0

        if not (MIN_EXCLUSION_MM <= w_mm <= MAX_EXCLUSION_MM and MIN_EXCLUSION_MM <= h_mm <= MAX_EXCLUSION_MM):
            status = "EXCLUDED"
        else:
            # 2. Tolerance Rules (Valid Pellet)
            w_ok = self.MIN_VALID_SIZE_MM <= w_mm <= self.MAX_VALID_SIZE_MM
            h_ok = self.MIN_VALID_SIZE_MM <= h_mm <= self.MAX_VALID_SIZE_MM

            if w_ok and h_ok:
                status = "IN RANGE"
            elif w_mm < self.MIN_VALID_SIZE_MM or h_mm < self.MIN_VALID_SIZE_MM:
                status = "UNDERSIZED"
            elif w_mm > self.MAX_VALID_SIZE_MM or h_mm > self.MAX_VALID_SIZE_MM:
                status = "OVERSIZED"
            else:
                status = "OUT OF SPEC"  # Should be covered by UNDERSIZED/OVERSIZED, but kept for completeness

        # Determine color for bounding box (Green, Yellow, Red)
        box_color = (0, 0, 255)  # Default Red (BGR format for OpenCV)

        if status == "IN RANGE":
            box_color = (0, 255, 0)  # Green
        elif status == "EXCLUDED":
            pass  # Skip drawing excluded items
        elif status in ["UNDERSIZED", "OVERSIZED", "OUT OF SPEC"]:
            # Check for "Near Tolerance" (Yellow) which is not in spec but not wildly out
            # We will use Red for all out-of-spec, as requested.
            box_color = (0, 0, 255)  # Red

        # Convert the rotated rectangle to an array of points for drawing
        if status != "EXCLUDED":
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(self.annotated_image, [box], 0, box_color, 4)

        return {
            'Pellet #': 0,  # Will be set during final aggregation
            'X-position (px)': int(x_pos),
            'Y-position (px)': int(y_pos),
            'Measured Width (mm)': round(w_mm, 3),
            'Measured Height (mm)': round(h_mm, 3),
            'Status': status,
        }

    def run_analysis(self):
        """
        Main function to run the entire analysis pipeline.
        """
        if self.image is None:
            return "Error: Could not load image."

        # Initialize annotated image copy
        self.annotated_image = self.image.copy()

        # 1. Calibration
        self.px_per_mm = self._calibrate_system()
        if self.px_per_mm == 0.0:
            return "Error: Calibration failed."

        # 2. Object Detection

        # Convert image to HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Create a mask for the blue color range
        mask = cv2.inRange(hsv, self.LOWER_BLUE, self.UPPER_BLUE)

        # Clean up the mask using morphological operations (Dilate then Erode - Close operation)
        # This closes small gaps and smooths the boundaries, useful for noisy scans.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        measurements = []
        valid_pellet_count = 0
        in_range_count = 0
        oversize_count = 0
        undersize_count = 0
        excluded_count = 0

        # 3. Measurement and Tolerance Check
        for i, contour in enumerate(contours):
            # Filtering step 1: Ignore very small/large contours based on pixel area.
            # This filters out tiny noise and the (simulated) large calibration strip.
            area = cv2.contourArea(contour)
            if area < self.MIN_PIXEL_AREA_THRESHOLD or area > self.MAX_PIXEL_AREA_THRESHOLD:
                excluded_count += 1
                continue

            pellet_data = self._process_pellet(contour, self.px_per_mm)

            # Count the pellet's status
            if pellet_data['Status'] != "EXCLUDED":
                valid_pellet_count += 1
                pellet_data['Pellet #'] = valid_pellet_count  # Assign number to valid pellets
                measurements.append(pellet_data)

                if pellet_data['Status'] == "IN RANGE":
                    in_range_count += 1
                elif pellet_data['Status'] == "OVERSIZED":
                    oversize_count += 1
                elif pellet_data['Status'] == "UNDERSIZED":
                    undersize_count += 1
            else:
                # Add excluded items to the full list for CSV/JSON output, but without a pellet number
                pellet_data['Pellet #'] = 'N/A'
                measurements.append(pellet_data)

        # Final Results Data
        self.results_df = pd.DataFrame(measurements)

        summary = {
            "Total objects detected": len(contours),
            "Total excluded (noise/too large)": excluded_count,
            "Total valid pellets": valid_pellet_count,
            "Total in-range": in_range_count,
            "Total oversize": oversize_count,
            "Total undersize": undersize_count,
            "Pixels per mm (Calibrated)": round(self.px_per_mm, 4)
        }

        return summary

    def save_outputs(self, base_filename):
        """Saves the annotated image and the raw measurement data."""
        # 1. Save Annotated Image
        annotated_path = f"{base_filename}_annotated.png"
        cv2.imwrite(annotated_path, self.annotated_image)

        # 2. Save CSV and JSON
        csv_path = f"{base_filename}_measurements.csv"
        json_path = f"{base_filename}_measurements.json"

        self.results_df.to_csv(csv_path, index=False)
        self.results_df.to_json(json_path, orient='records', indent=4)

        return annotated_path, csv_path, json_path


# --- 2. PYQT6 TABLE MODEL ---

class ResultsTableModel(QAbstractTableModel):
    """
    A custom model to handle pandas DataFrame data for QTableView.
    """

    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=QModelIndex()):
        return self._data.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QVariant()

        value = self._data.iloc[index.row(), index.column()]

        if role == Qt.ItemDataRole.DisplayRole:
            return str(value)

        # Apply background color based on 'Status' column
        if role == Qt.ItemDataRole.BackgroundRole and self.headerData(index.column(),
                                                                      Qt.Orientation.Horizontal) == 'Status':
            status = self._data.iloc[index.row()]['Status']
            if status == "IN RANGE":
                return QColor(144, 238, 144)  # Light Green
            elif status == "OVERSIZED" or status == "UNDERSIZED":
                return QColor(255, 99, 71)  # Tomato Red
            elif status == "EXCLUDED":
                return QColor(200, 200, 200)  # Light Gray (Skipped)

        return QVariant()

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            elif orientation == Qt.Orientation.Vertical:
                return str(section + 1)
        return QVariant()


# --- 3. PYQT6 MAIN WINDOW ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Size Measurement System")
        self.setGeometry(100, 100, 1400, 800)

        self.pellet_analyzer = None

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Left Panel (Image Display)
        self.image_label = QLabel("Load an image to start...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(800, 800)  # Fixed size for a visually clear annotated image
        self.main_layout.addWidget(self.image_label)

        # Right Panel (Results)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)

        # 1. Control Buttons
        self.btn_load = QPushButton("1. Load Image")
        self.btn_load.clicked.connect(self.load_image)
        self.btn_run = QPushButton("2. Run Analysis")
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_run.setEnabled(False)
        self.btn_save = QPushButton("3. Save Results (PNG, CSV, JSON)")
        self.btn_save.clicked.connect(self.save_results)
        self.btn_save.setEnabled(False)

        control_layout = QGridLayout()
        control_layout.addWidget(self.btn_load, 0, 0)
        control_layout.addWidget(self.btn_run, 0, 1)
        control_layout.addWidget(self.btn_save, 0, 2)
        self.right_layout.addLayout(control_layout)

        # 2. Summary Display
        self.summary_group = QWidget()
        self.summary_layout = QVBoxLayout(self.summary_group)
        self.summary_title = QLabel("--- Analysis Summary ---")
        self.summary_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.summary_layout.addWidget(self.summary_title)

        self.summary_labels = {
            'Total objects detected': QLabel("Total objects detected: N/A"),
            'Total excluded (noise/too large)': QLabel("Total excluded (noise/too large): N/A"),
            'Total valid pellets': QLabel("Total valid pellets: N/A"),
            'Total in-range': QLabel("Total in-range: N/A"),
            'Total oversize': QLabel("Total oversize: N/A"),
            'Total undersize': QLabel("Total undersize: N/A"),
            'Pixels per mm (Calibrated)': QLabel("Pixels per mm (Calibrated): N/A")
        }
        for label in self.summary_labels.values():
            self.summary_layout.addWidget(label)

        self.right_layout.addWidget(self.summary_group)

        # 3. Measurements Table
        self.table_label = QLabel("--- Raw Measurements Table ---")
        self.table_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_layout.addWidget(self.table_label)

        self.table_view = QTableView()
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.right_layout.addWidget(self.table_view)

        self.main_layout.addWidget(self.right_panel)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Scan Image", "", "Images (*.png *.jpg *.jpeg)")

        if file_path:
            self.image_path = file_path
            self.pellet_analyzer = PelletAnalyzer(file_path)

            # Display original image
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation))
            self.image_label.setText(os.path.basename(file_path))

            self.btn_run.setEnabled(True)
            self.btn_save.setEnabled(False)
            self.clear_results()

    def clear_results(self):
        # Clear summary labels
        for key, label in self.summary_labels.items():
            if key == 'Pixels per mm (Calibrated)':
                label.setText(f"Pixels per mm (Calibrated): N/A")
            else:
                label.setText(f"{key}: N/A")

        # Clear table
        self.table_view.setModel(QAbstractTableModel())

    def run_analysis(self):
        if self.pellet_analyzer is None:
            return

        self.clear_results()

        # Run the CV analysis
        summary = self.pellet_analyzer.run_analysis()

        if isinstance(summary, str) and summary.startswith("Error"):
            self.image_label.setText(summary)
            return

        # 1. Update Summary Display
        for key, value in summary.items():
            self.summary_labels[key].setText(f"{key}: {value}")

        # 2. Update Table View
        df = self.pellet_analyzer.results_df
        table_model = ResultsTableModel(df)
        self.table_view.setModel(table_model)

        # 3. Display Annotated Image
        q_image = self._convert_cv_to_qimage(self.pellet_analyzer.annotated_image)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation))

        self.btn_save.setEnabled(True)

    def _convert_cv_to_qimage(self, cv_img):
        """Converts an OpenCV BGR image (numpy array) to a PyQt6 QImage."""
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        return QImage(cv_img.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)

    def save_results(self):
        if self.pellet_analyzer is None or self.pellet_analyzer.results_df.empty:
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "pellet_analysis_output", "All Files (*)")

        if save_path:
            base_filename = os.path.splitext(save_path)[0]
            annotated_path, csv_path, json_path = self.pellet_analyzer.save_outputs(base_filename)
            print(f"Results saved:\n- {annotated_path}\n- {csv_path}\n- {json_path}")


# --- MAIN EXECUTION ---

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()