import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
    QGroupBox, QFormLayout, QDoubleSpinBox, QScrollArea,
    QSpacerItem, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImage, QIcon
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
            QLineEdit, QDoubleSpinBox {
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
        self.img_label.setText(file_name.split("/")[-1])
        self.display_image(self.raw_image)
        self.reset_calibration()

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
        if self.raw_image would not be None:
            self.display_image(self.raw_image)


# === Run ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PelletAnalyzer()
    window.show()
    sys.exit(app.exec())