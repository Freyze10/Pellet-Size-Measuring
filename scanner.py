import sys
import cv2
import numpy as np
import csv
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
    QTabWidget, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QScrollArea, QFrame, QProgressBar, QComboBox, QCheckBox,
    QFileDialog, QSpacerItem, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImage, QIcon, QFont, QPalette, QColor
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QObject


# === Helper: Convert OpenCV to QPixmap ===
def cv_to_qpixmap(cv_img, target_size=None):
    if cv_img is None:
        return QPixmap()
    height, width = cv_img.shape[:2]
    if len(cv_img.shape) == 3:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888)
    else:
        qimg = QImage(cv_img.data, width, height, width, QImage.Format.Format_Grayscale8)
    pixmap = QPixmap.fromImage(qimg)
    if target_size and not target_size.isNull():
        return pixmap.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
    return pixmap


# === Calibration Worker (for clean separation) ===
class Calibrator(QObject):
    finished = pyqtSignal(float)  # emits PPM

    def __init__(self, img, pt1, pt2, real_mm):
        super().__init__()
        self.img = img
        self.pt1 = pt1
        self.pt2 = pt2
        self.real_mm = real_mm

    def run(self):
        if None in (self.pt1, self.pt2) or self.real_mm <= 0:
            self.finished.emit(0.0)
            return
        dist_px = np.linalg.norm(np.array(self.pt1) - np.array(self.pt2))
        if dist_px < 1:
            self.finished.emit(0.0)
            return
        ppm = dist_px / self.real_mm
        self.finished.emit(ppm)


# === Main Window ===
class PelletAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Size Analyzer Pro")
        self.setGeometry(100, 100, 1400, 900)
        self.raw_image = None
        self.processed_image = None
        self.PPM = 0.0
        self.calib_pt1 = None
        self.calib_pt2 = None
        self.calib_line_img = None

        self.init_ui()
        self.apply_dark_theme()

    def apply_dark_theme(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 50))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(100, 180, 255))
        QApplication.setPalette(dark_palette)
        QApplication.setStyle("Fusion")

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # === Left Panel: Tabs ===
        self.tabs = QTabWidget()
        self.tabs.setMaximumWidth(380)
        self.tabs.setFont(QFont("Segoe UI", 10))
        main_layout.addWidget(self.tabs)

        # --- Tab 1: Load & Calibrate ---
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)

        # Load Image
        load_group = QGroupBox("1. Load Scanned Image")
        load_layout = QVBoxLayout()
        self.load_btn = QPushButton("Choose Image File")
        self.load_btn.setIcon(QIcon.fromTheme("document-open"))
        self.load_btn.clicked.connect(self.load_image)
        load_layout.addWidget(self.load_btn)
        self.img_path_label = QLabel("No image selected")
        self.img_path_label.setWordWrap(True)
        self.img_path_label.setStyleSheet("color: #aaa; font-size: 11px;")
        load_layout.addWidget(self.img_path_label)
        load_group.setLayout(load_layout)
        tab1_layout.addWidget(load_group)

        # Calibration Group
        calib_group = QGroupBox("2. Real-Time Calibration (Click Two Points)")
        calib_layout = QFormLayout()

        self.calib_length_input = QDoubleSpinBox()
        self.calib_length_input.setRange(0.1, 1000.0)
        self.calib_length_input.setDecimals(2)
        self.calib_length_input.setValue(10.0)
        self.calib_length_input.setSuffix(" mm")
        calib_layout.addRow("Known Length:", self.calib_length_input)

        self.calib_status = QLabel("Click first point on image...")
        self.calib_status.setStyleSheet("color: #ffcc00;")
        calib_layout.addRow("Status:", self.calib_status)

        self.ppm_label = QLabel("PPM: Not calibrated")
        self.ppm_label.setStyleSheet("font-weight: bold; color: #00ff99;")
        calib_layout.addRow("Calibration:", self.ppm_label)

        self.reset_calib_btn = QPushButton("Reset Calibration")
        self.reset_calib_btn.clicked.connect(self.reset_calibration)
        calib_layout.addRow(self.reset_calib_btn)

        calib_group.setLayout(calib_layout)
        tab1_layout.addWidget(calib_group)

        # DPI Fallback
        dpi_group = QGroupBox("Fallback: Use Scanner DPI")
        dpi_layout = QFormLayout()
        self.dpi_input = QSpinBox()
        self.dpi_input.setRange(72, 4800)
        self.dpi_input.setValue(600)
        dpi_layout.addRow("DPI:", self.dpi_input)
        self.dpi_ppm_btn = QPushButton("Apply DPI → PPM")
        self.dpi_ppm_btn.clicked.connect(self.apply_dpi_ppm)
        dpi_layout.addRow(self.dpi_ppm_btn)
        dpi_group.setLayout(dpi_layout)
        tab1_layout.addWidget(dpi_group)

        tab1_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        self.tabs.addTab(tab1, "Load & Calibrate")

        # --- Tab 2: Analysis Settings ---
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)

        settings_group = QGroupBox("Analysis Parameters")
        settings_layout = QFormLayout()

        self.min_area_input = QSpinBox()
        self.min_area_input.setRange(100, 50000)
        self.min_area_input.setValue(1000)
        settings_layout.addRow("Min Area (px²):", self.min_area_input)

        self.blur_kernel_input = QSpinBox()
        self.blur_kernel_input.setRange(1, 21)
        self.blur_kernel_input.setSingleStep(2)
        self.blur_kernel_input.setValue(5)
        settings_layout.addRow("Blur Kernel:", self.blur_kernel_input)

        self.threshold_input = QSpinBox()
        self.threshold_input.setRange(0, 255)
        self.threshold_input.setValue(100)
        settings_layout.addRow("Threshold (INV):", self.threshold_input)

        self.auto_threshold_cb = QCheckBox("Auto Threshold (Otsu)")
        self.auto_threshold_cb.setChecked(False)
        settings_layout.addRow(self.auto_threshold_cb)

        settings_group.setLayout(settings_layout)
        tab2_layout.addWidget(settings_group)

        self.analyze_btn = QPushButton("Start Analysis")
        self.analyze_btn.setStyleSheet("QPushButton { background-color: #0066cc; color: white; font-weight: bold; padding: 10px; }")
        self.analyze_btn.clicked.connect(self.analyze_image)
        tab2_layout.addWidget(self.analyze_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        tab2_layout.addWidget(self.progress_bar)

        tab2_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        self.tabs.addTab(tab2, "Analyze")

        # --- Tab 3: Results ---
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)

        self.results_text = QLabel("Results will appear here after analysis.")
        self.results_text.setWordWrap(True)
        self.results_text.setStyleSheet("background: #222; padding: 15px; border-radius: 8px;")
        tab3_layout.addWidget(self.results_text)

        export_btn = QPushButton("Export Results to CSV")
        export_btn.clicked.connect(self.export_results)
        tab3_layout.addWidget(export_btn)

        tab3_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        self.tabs.addTab(tab3, "Results")

        # === Right Panel: Image Viewer ===
        self.image_label = QLabel("Image will appear here")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet("background: #111; border: 2px dashed #444; border-radius: 10px;")
        self.image_label.mousePressEvent = self.on_image_click

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.image_label)
        main_layout.addWidget(scroll, 1)

    # === Image Loading ===
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Scanned Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        if not file_name:
            return
        img = cv2.imread(file_name)
        if img is None:
            QMessageBox.critical(self, "Error", "Failed to load image.")
            return

        self.raw_image = img
        self.processed_image = img.copy()
        self.img_path_label.setText(file_name.split("/")[-1])
        self.display_image(self.raw_image)
        self.calib_pt1 = self.calib_pt2 = None
        self.update_calibration_status()

    def display_image(self, cv_img):
        pixmap = cv_to_qpixmap(cv_img, QSize(1000, 800))
        self.image_label.setPixmap(pixmap)
        self.image_label.resize(pixmap.size())

    # === Calibration via Mouse Click ===
    def on_image_click(self, event):
        if self.raw_image is None:
            return
        if self.PPM > 0:
            QMessageBox.information(self, "Calibration Locked", "Calibration already done. Reset to recalibrate.")
            return

        pos = event.pos()
        scale_x = self.raw_image.shape[1] / self.image_label.pixmap().width()
        scale_y = self.raw_image.shape[0] / self.image_label.pixmap().height()
        x = int(pos.x() * scale_x)
        y = int(pos.y() * scale_y)

        if self.calib_pt1 is None:
            self.calib_pt1 = (x, y)
            self.calib_status.setText(f"First point: ({x}, {y}) → Click second point")
        else:
            self.calib_pt2 = (x, y)
            self.calibrate_from_points()

        self.draw_calibration_line()

    def draw_calibration_line(self):
        if self.raw_image is None:
            return
        img = self.raw_image.copy()
        color = (0, 255, 255)  # Yellow
        thickness = 3
        if self.calib_pt1:
            cv2.circle(img, self.calib_pt1, 8, color, -1)
            cv2.circle(img, self.calib_pt1, 12, color, 3)
        if self.calib_pt1 and self.calib_pt2:
            cv2.line(img, self.calib_pt1, self.calib_pt2, color, thickness)
            cv2.circle(img, self.calib_pt2, 8, color, -1)
            cv2.circle(img, self.calib_pt2, 12, color, 3)
            mid = ((self.calib_pt1[0] + self.calib_pt2[0]) // 2, (self.calib_pt1[1] + self.calib_pt2[1]) // 2)
            cv2.putText(img, f"{self.calib_length_input.value():.2f} mm",
                        (mid[0] - 50, mid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.display_image(img)

    def calibrate_from_points(self):
        if not all((self.calib_pt1, self.calib_pt2)):
            return
        dist_px = np.linalg.norm(np.array(self.calib_pt1) - np.array(self.calib_pt2))
        real_mm = self.calib_length_input.value()
        if dist_px < 10:
            QMessageBox.warning(self, "Too Short", "Selected distance is too small. Try a longer reference.")
            return
        self.PPM = dist_px / real_mm
        self.update_calibration_status()
        QMessageBox.information(self, "Calibration Complete",
                                f"PPM = {self.PPM:.2f} px/mm\n"
                                f"Distance: {dist_px:.1f}px → {real_mm}mm")

    def update_calibration_status(self):
        if self.PPM > 0:
            self.ppm_label.setText(f"PPM: {self.PPM:.2f} px/mm")
            self.ppm_label.setStyleSheet("font-weight: bold; color: #00ff99;")
            self.calib_status.setText("Calibration complete!")
        else:
            self.ppm_label.setText("PPM: Not calibrated")
            self.ppm_label.setStyleSheet("font-weight: bold; color: #ff6666;")

    def reset_calibration(self):
        self.calib_pt1 = self.calib_pt2 = None
        self.PPM = 0.0
        self.update_calibration_status()
        if self.raw_image is not None:
            self.display_image(self.raw_image)

    def apply_dpi_ppm(self):
        dpi = self.dpi_input.value()
        self.PPM = dpi / 25.4
        self.update_calibration_status()
        QMessageBox.information(self, "DPI Applied", f"PPM set to {self.PPM:.2f} using {dpi} DPI")

    # === Analysis ===
    def analyze_image(self):
        if self.raw_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        if self.PPM <= 0:
            reply = QMessageBox.question(self, "No Calibration",
                                         "No pixel-to-mm calibration. Use DPI fallback?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.apply_dpi_ppm()
            else:
                return
            if self.PPM <= 0:
                return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.analyze_btn.setEnabled(False)

        # Copy for processing
        img = self.raw_image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ksize = self.blur_kernel_input.value() | 1  # ensure odd
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        if self.auto_threshold_cb.isChecked():
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(blurred, self.threshold_input.value(), 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = self.min_area_input.value()
        diameters_mm = []
        overlay = img.copy()

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            diameter_px = radius * 2
            diameter_mm = diameter_px / self.PPM
            diameters_mm.append(diameter_mm)

            center = (int(x), int(y))
            cv2.circle(overlay, center, int(radius), (0, 255, 255), 3)
            label = f"{diameter_mm:.2f}mm"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(overlay, label, (center[0] - tw//2, center[1] + int(radius) + th + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        self.processed_image = overlay
        self.display_image(overlay)

        # Results
        if not diameters_mm:
            result = "No pellets detected. Try lowering min area or adjusting threshold."
        else:
            avg = np.mean(diameters_mm)
            std = np.std(diameters_mm)
            result = (
                f"<b>Analysis Complete</b><br>"
                f"<b>Total Pellets:</b> {len(diameters_mm)}<br>"
                f"<b>Average Diameter:</b> {avg:.3f} mm<br>"
                f"<b>Std Deviation:</b> {std:.3f} mm<br>"
                f"<b>PPM Used:</b> {self.PPM:.2f} px/mm<br>"
                f"<b>Min Area Filter:</b> {min_area} px²"
            )
            self.last_results = {
                "diameters_mm": diameters_mm,
                "avg": avg,
                "std": std,
                "total": len(diameters_mm),
                "ppm": self.PPM
            }

        self.results_text.setText(result)
        self.tabs.setCurrentIndex(2)

        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)

    def export_results(self):
        if not hasattr(self, 'last_results'):
            QMessageBox.warning(self, "No Data", "Run analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "pellet_analysis.csv", "CSV Files (*.csv)")
        if not path:
            return
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Pellet #", "Diameter (mm)"])
            for i, d in enumerate(self.last_results["diameters_mm"], 1):
                writer.writerow([i, f"{d:.3f}"])
            writer.writerow([])
            writer.writerow(["Summary", ""])
            writer.writerow(["Total Pellets", self.last_results["total"]])
            writer.writerow(["Average (mm)", f"{self.last_results['avg']:.3f}"])
            writer.writerow(["Std Dev (mm)", f"{self.last_results['std']:.3f}"])
            writer.writerow(["PPM", f"{self.last_results['ppm']:.2f}"])
        QMessageBox.information(self, "Exported", f"Results saved to:\n{path}")


# === Run App ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PelletAnalyzer()
    window.show()
    sys.exit(app.exec())