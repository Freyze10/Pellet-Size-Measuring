import sys
import cv2
import numpy as np
import csv
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QScrollArea,
    QSpacerItem, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt6.QtCore import Qt, QSize


# === Convert OpenCV → QPixmap ===
def cv_to_qpixmap(cv_img, target_size=None):
    if cv_img is None:
        return QPixmap()
    h, w = cv_img.shape[:2]
    if len(cv_img.shape) == 3:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    else:
        qimg = QImage(cv_img.data, w, h, w, QImage.Format.Format_Grayscale8)
    pixmap = QPixmap.fromImage(qimg)
    if target_size and not target_size.isNull():
        return pixmap.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
    return pixmap


# === Main App ===
class PelletAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Size Analyzer")
        self.setGeometry(100, 100, 1300, 800)
        self.raw_image = None
        self.processed_image = None
        self.pixels_per_mm = 0.0
        self.calib_pt1 = None
        self.calib_pt2 = None

        self.init_ui()
        self.apply_light_theme()

    def apply_light_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f8f9fc; }
            QLabel { color: #2c3e50; font-family: 'Segoe UI'; }
            QGroupBox { 
                font-weight: bold; 
                border: 2px solid #e0e6ed; 
                border-radius: 10px; 
                margin-top: 10px; 
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 15px; 
                padding: 0 10px;
                color: #3498db;
            }
            QPushButton {
                background-color: #3498db; 
                color: white; 
                border: none; 
                padding: 12px; 
                border-radius: 8px; 
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton#analyzeBtn { background-color: #27ae60; }
            QPushButton#analyzeBtn:hover { background-color: #219653; }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                padding: 8px; 
                border: 1.5px solid #bdc3c7; 
                border-radius: 6px;
                font-size: 14px;
            }
            QScrollArea { border: none; background: white; }
        """)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # === LEFT: Controls ===
        left_panel = QWidget()
        left_panel.setMaximumWidth(360)
        left_layout = QVBoxLayout(left_panel)

        # 1. Load Image
        load_group = QGroupBox("1. Load Scanned Image")
        load_layout = QVBoxLayout()
        self.load_btn = QPushButton("Choose Image")
        self.load_btn.setIcon(QIcon.fromTheme("document-open"))
        self.load_btn.clicked.connect(self.load_image)
        load_layout.addWidget(self.load_btn)

        self.img_label = QLabel("No image selected")
        self.img_label.setStyleSheet("color: #7f8c8d; font-size: 12px;")
        self.img_label.setWordWrap(True)
        load_layout.addWidget(self.img_label)
        load_group.setLayout(load_layout)
        left_layout.addWidget(load_group)

        # 2. Calibration
        calib_group = QGroupBox("2. Calibrate Scale (Click 2 Points)")
        calib_layout = QFormLayout()

        self.length_input = QDoubleSpinBox()
        self.length_input.setRange(0.1, 1000.0)
        self.length_input.setDecimals(2)
        self.length_input.setValue(10.0)
        self.length_input.setSuffix(" mm")
        calib_layout.addRow("Known Length:", self.length_input)

        self.calib_status = QLabel("Click first point on image...")
        self.calib_status.setStyleSheet("color: #e67e22; font-weight: bold;")
        calib_layout.addRow("Status:", self.calib_status)

        self.scale_label = QLabel("Scale: Not set")
        self.scale_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        calib_layout.addRow("Scale:", self.scale_label)

        self.reset_btn = QPushButton("Reset Calibration")
        self.reset_btn.clicked.connect(self.reset_calibration)
        calib_layout.addRow(self.reset_btn)

        calib_group.setLayout(calib_layout)
        left_layout.addWidget(calib_group)

        # 3. Analysis Settings
        settings_group = QGroupBox("3. Analysis Settings")
        settings_layout = QFormLayout()

        self.min_area_input = QSpinBox()
        self.min_area_input.setRange(100, 50000)
        self.min_area_input.setValue(1000)
        settings_layout.addRow("Min Area (px²):", self.min_area_input)

        settings_group.setLayout(settings_layout)
        left_layout.addWidget(settings_group)

        # 4. Analyze Button
        self.analyze_btn = QPushButton("Analyze Pellets")
        self.analyze_btn.setObjectName("analyzeBtn")
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.analyze_btn.setEnabled(False)
        left_layout.addWidget(self.analyze_btn)

        # 5. Results Area
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()
        self.results_display = QLabel(
            "<div style='color:#7f8c8d; text-align:center; padding:20px;'>"
            "Load image and calibrate scale to begin."
            "</div>"
        )
        self.results_display.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.results_display.setWordWrap(True)
        self.results_display.setStyleSheet("background: #f0f3f7; border-radius: 8px; padding: 15px;")
        results_layout.addWidget(self.results_display)

        self.export_btn = QPushButton("Export to CSV")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        results_layout.addWidget(self.export_btn)

        results_group.setLayout(results_layout)
        left_layout.addWidget(results_group)

        left_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        main_layout.addWidget(left_panel)

        # === RIGHT: Image Viewer ===
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(600, 500)
        self.image_label.setStyleSheet("""
            background: white; 
            border: 3px dashed #bdc3c7; 
            border-radius: 12px;
            font-size: 18px; 
            color: #95a5a6;
        """)
        self.image_label.setText("Image will appear here\nClick to calibrate")
        self.image_label.mousePressEvent = self.on_image_click

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.image_label)
        main_layout.addWidget(scroll, 1)

    # === Load Image ===
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        if not file_name:
            return
        img = cv2.imread(file_name)
        if img is None:
            QMessageBox.critical(self, "Error", "Cannot load image.")
            return

        self.raw_image = img
        self.processed_image = img.copy()
        self.img_label.setText(file_name.split("/")[-1])
        self.display_image(self.raw_image)
        self.reset_calibration()
        self.analyze_btn.setEnabled(False)
        self.export_btn.setEnabled(False)

    def display_image(self, cv_img):
        pixmap = cv_to_qpixmap(cv_img, QSize(1000, 700))
        self.image_label.setPixmap(pixmap)
        self.image_label.setStyleSheet("background: white; border: 2px solid #3498db; border-radius: 12px;")

    # === Calibration Click ===
    def on_image_click(self, event):
        if self.raw_image is None:
            return
        if self.pixels_per_mm > 0:
            QMessageBox.information(self, "Already Calibrated", "Reset calibration to try again.")
            return

        pos = event.pos()
        pw, ph = self.image_label.pixmap().width(), self.image_label.pixmap().height()
        scale_x = self.raw_image.shape[1] / pw
        scale_y = self.raw_image.shape[0] / ph
        x = int(pos.x() * scale_x)
        y = int(pos.y() * scale_y)

        if self.calib_pt1 is None:
            self.calib_pt1 = (x, y)
            self.calib_status.setText(f"Point 1: ({x}, {y}) → Click Point 2")
        else:
            self.calib_pt2 = (x, y)
            self.finalize_calibration()

        self.draw_calibration_line()

    def draw_calibration_line(self):
        if not self.raw_image:
            return
        img = self.raw_image.copy()
        color = (0, 200, 255)  # Orange
        thick = 4
        if self.calib_pt1:
            cv2.circle(img, self.calib_pt1, 10, color, -1)
            cv2.circle(img, self.calib_pt1, 14, color, 4)
        if self.calib_pt1 and self.calib_pt2:
            cv2.line(img, self.calib_pt1, self.calib_pt2, color, thick)
            cv2.circle(img, self.calib_pt2, 10, color, -1)
            cv2.circle(img, self.calib_pt2, 14, color, 4)
            mid = ((self.calib_pt1[0] + self.calib_pt2[0]) // 2, (self.calib_pt1[1] + self.calib_pt2[1]) // 2)
            cv2.putText(img, f"{self.length_input.value():.2f} mm",
                        (mid[0] - 60, mid[1] - 15), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        self.display_image(img)

    def finalize_calibration(self):
        dist_px = np.linalg.norm(np.array(self.calib_pt1) - np.array(self.calib_pt2))
        real_mm = self.length_input.value()
        if dist_px < 20:
            QMessageBox.warning(self, "Too Short", "Select a longer reference distance.")
            return
        self.pixels_per_mm = dist_px / real_mm
        self.scale_label.setText(f"Scale: {self.pixels_per_mm:.2f} px/mm")
        self.calib_status.setText("Calibration complete!")
        self.analyze_btn.setEnabled(True)
        QMessageBox.information(
            self, "Calibration Done",
            f"Scale set: <b>{self.pixels_per_mm:.2f} pixels = 1 mm</b><br>"
            f"Distance: {dist_px:.1f} px → {real_mm} mm"
        )

    def reset_calibration(self):
        self.calib_pt1 = self.calib_pt2 = None
        self.pixels_per_mm = 0.0
        self.scale_label.setText("Scale: Not set")
        self.calib_status.setText("Click first point on image...")
        self.analyze_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        if self.raw_image is not None:
            self.display_image(self.raw_image)

    # === Analyze ===
    def analyze_image(self):
        if not self.raw_image or self.pixels_per_mm <= 0:
            return

        img = self.raw_image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
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
            (x, y), r = cv2.minEnclosingCircle(c)
            d_px = r * 2
            d_mm = d_px / self.pixels_per_mm
            diameters_mm.append(d_mm)

            center = (int(x), int(y))
            cv2.circle(overlay, center, int(r), (76, 175, 80), 3)  # Green
            label = f"{d_mm:.2f}mm"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(overlay, label, (center[0] - tw//2, center[1] + int(r) + th + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        self.processed_image = overlay
        self.display_image(overlay)

        if not diameters_mm:
            result = "<p style='color:#e74c3c;'>No pellets found.</p><p>Try lowering the minimum area.</p>"
        else:
            avg = np.mean(diameters_mm)
            std = np.std(diameters_mm)
            result = f"""
            <div style='line-height: 1.6;'>
                <b style='color:#27ae60; font-size:16px;'>Analysis Complete!</b><br><br>
                <b>Total Pellets:</b> {len(diameters_mm)}<br>
                <b>Average Diameter:</b> {avg:.3f} mm<br>
                <b>Std Deviation:</b> {std:.3f} mm<br>
                <b>Scale Used:</b> {self.pixels_per_mm:.2f} px/mm<br>
                <b>Min Area:</b> {min_area} px²
            </div>
            """
            self.last_results = {
                "diameters": diameters_mm,
                "avg": avg, "std": std, "count": len(diameters_mm),
                "scale": self.pixels_per_mm
            }
            self.export_btn.setEnabled(True)

        self.results_display.setText(result)

    def export_results(self):
        if not hasattr(self, 'last_results'):
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "pellet_sizes.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["Pellet", "Diameter (mm)"])
            for i, d in enumerate(self.last_results["diameters"], 1):
                w.writerow([i, f"{d:.3f}"])
            w.writerow([])
            w.writerow(["Summary", "Value"])
            w.writerow(["Total", self.last_results["count"]])
            w.writerow(["Average (mm)", f"{self.last_results['avg']:.3f}"])
            w.writerow(["Std Dev (mm)", f"{self.last_results['std']:.3f}"])
            w.writerow(["Scale (px/mm)", f"{self.last_results['scale']:.2f}"])
        QMessageBox.information(self, "Saved", f"Results exported to:\n{path}")


# === Run ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PelletAnalyzer()
    window.show()
    sys.exit(app.exec())