from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
import glob
import random
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QGroupBox, QScrollArea, QTextEdit,
                             QDoubleSpinBox, QMessageBox, QSizePolicy)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage

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
    except: return None

# --- Color fallback detector ---
class ColorPelletDetector:
    LOWER_BLUE = np.array([90, 100, 30])
    UPPER_BLUE = np.array([150, 255, 255])
    MIN_PIXEL_AREA = 300
    MAX_PIXEL_AREA = 20000

    def detect_pellets(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWER_BLUE, self.UPPER_BLUE)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        for c in contours:
            a = cv2.contourArea(c)
            if self.MIN_PIXEL_AREA <= a <= self.MAX_PIXEL_AREA:
                w, h = cv2.minAreaRect(c)[1]
                if min(w,h)>0 and 1.0 <= max(w,h)/min(w,h) <= 5.0:
                    valid.append(c)
        return valid

# --- Main app ---
class PelletMeasurementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Inspector - YOLO Scanner")
        self.setGeometry(50,50,1400,900)

        self.pixels_per_mm = 25.4
        self.target_diameter = 3.0
        self.target_length = 3.0
        self.tolerance = 0.5
        self.update_ranges()

        self.current_image = None
        self.detected_pellets = []
        self.yolo_detector = None
        self.cv_detector = ColorPelletDetector()
        self.dataset_folder = "pellet_label_yolo"

        self.init_ui()
        self.load_model("trained_model/best.pt")  # <- your trained model

    def update_ranges(self):
        self.d_min = self.target_diameter - self.tolerance
        self.d_max = self.target_diameter + self.tolerance
        self.l_min = self.target_length - self.tolerance
        self.l_max = self.target_length + self.tolerance

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout()
        central.setLayout(layout)

        layout.addWidget(self.left_panel(),1)
        layout.addWidget(self.right_panel(),3)

    def left_panel(self):
        w = QWidget()
        l = QVBoxLayout()
        w.setLayout(l)

        # Load Image button
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        l.addWidget(self.load_btn)

        # Calibration spinbox
        g = QGroupBox("Calibration")
        gl = QVBoxLayout()
        h = QHBoxLayout()
        h.addWidget(QLabel("px/mm:"))
        self.px_spin = QDoubleSpinBox()
        self.px_spin.setRange(0.1,200)
        self.px_spin.setValue(self.pixels_per_mm)
        self.px_spin.valueChanged.connect(lambda v: setattr(self,'pixels_per_mm',v))
        h.addWidget(self.px_spin)
        gl.addLayout(h)
        g.setLayout(gl)
        l.addWidget(g)

        # Stats
        g = QGroupBox("Stats")
        gl = QVBoxLayout()
        self.total_lbl = QLabel("Total: 0")
        self.ok_lbl = QLabel("OK: 0")
        self.bad_lbl = QLabel("BAD: 0")
        gl.addWidget(self.total_lbl)
        gl.addWidget(self.ok_lbl)
        gl.addWidget(self.bad_lbl)
        g.setLayout(gl)
        l.addWidget(g)

        # Details
        g = QGroupBox("Pellet Details")
        gl = QVBoxLayout()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.det_w = QWidget()
        self.det_l = QVBoxLayout()
        self.det_w.setLayout(self.det_l)
        self.scroll.setWidget(self.det_w)
        gl.addWidget(self.scroll)
        g.setLayout(gl)
        l.addWidget(g)

        l.addStretch()
        return w

    def right_panel(self):
        w = QWidget()
        l = QVBoxLayout()
        w.setLayout(l)
        self.img_lbl = QLabel("Load image...")
        self.img_lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.img_lbl.setStyleSheet("border:2px solid #ccc;background:#f0f0f0;")
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setWidget(self.img_lbl)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        l.addWidget(scroll)
        return w

    def load_model(self, path):
        if os.path.exists(path):
            try:
                self.yolo_detector = YOLO(path)
                QMessageBox.information(self, "Model Loaded", f"YOLO model loaded from:\n{path}")
            except Exception as e:
                print("YOLO load error:", e)

    def load_image(self):
        path,_ = QFileDialog.getOpenFileName(self,"Load Image","","Images (*.png *.jpg *.jpeg *.bmp)")
        if not path: return
        self.current_image = cv2_safe_imread(path)
        if self.current_image is None:
            QMessageBox.critical(self,"Error","Failed to load image")
            return
        self.process_image()

    def process_image(self):
        if self.current_image is None: return
        img_disp = self.current_image.copy()
        self.detected_pellets = []

        # --- YOLO detection ---
        polygons, confidences = [], []
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
                    roi = self.current_image[y1:y2, x1:x2]
                    if roi.size == 0: continue
                    cnts = self.cv_detector.detect_pellets(roi)
                    if not cnts: continue
                    c = max(cnts, key=cv2.contourArea)
                    c += np.array([x1, y1])
                    M = cv2.moments(c)
                    cx = int(M["m10"] / M["m00"]) if M["m00"] else int(c[:, 0].mean())
                    cy = int(M["m01"] / M["m00"]) if M["m00"] else int(c[:, 1].mean())
                    temp_pellets.append({'polygon': c, 'confidence': conf, 'center': (cx, cy)})

                # --- FILTER DUPLICATES BY CENTER + HIGHEST CONF ---
                filtered = []
                threshold = 8  # px distance to consider same pellet
                for p in sorted(temp_pellets, key=lambda x: -x['confidence']):
                    duplicate = False
                    for f in filtered:
                        dist = np.hypot(p['center'][0] - f['center'][0], p['center'][1] - f['center'][1])
                        if dist < threshold:
                            duplicate = True
                            break
                    if not duplicate:
                        filtered.append(p)

                polygons = [p['polygon'] for p in filtered]
                confidences = [p['confidence'] for p in filtered]

        except:
            polygons = self.cv_detector.detect_pellets(self.current_image)
            confidences = [100] * len(polygons)

        # --- Measurement & Drawing ---
        for i, (poly, conf) in enumerate(zip(polygons, confidences), 1):
            rect = cv2.minAreaRect(poly.astype(np.float32))
            w, h = rect[1]
            d = min(w, h) / self.pixels_per_mm
            l = max(w, h) / self.pixels_per_mm
            ok = self.d_min <= d <= self.d_max and self.l_min <= l <= self.l_max
            pellet = {'polygon': poly, 'diameter': d, 'length': l, 'within': ok, 'confidence': conf, 'id': i}
            self.detected_pellets.append(pellet)
            self.draw_pellet(img_disp, pellet)

        self.update_stats()
        self.show_image(img_disp)

    def draw_pellet(self,img,p):
        color = (0,255,0) if p['within'] else (0,0,255)
        overlay = img.copy()
        cv2.fillPoly(overlay,[p['polygon'].reshape(-1,1,2)], color)
        cv2.addWeighted(overlay,0.25,img,0.75,0,img)
        box = np.intp(cv2.boxPoints(cv2.minAreaRect(p['polygon'].astype(np.float32))))
        cv2.drawContours(img,[box],0,color,3)
        M = cv2.moments(p['polygon'])
        cx = int(M["m10"]/M["m00"]) if M["m00"] else int(p['polygon'][:,0].mean())
        cy = int(M["m01"]/M["m00"]) if M["m00"] else int(p['polygon'][:,1].mean())
        cv2.putText(img,str(p['id']),(cx-20,cy+12),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2)
        if p['confidence']<100:
            cv2.putText(img,f"{p['confidence']:.0f}%",(cx-30,cy-20),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)

    def update_stats(self):
        total = len(self.detected_pellets)
        ok = sum(1 for p in self.detected_pellets if p['within'])
        bad = total-ok
        self.total_lbl.setText(f"Total: {total}")
        self.ok_lbl.setText(f"OK: {ok}")
        self.bad_lbl.setText(f"BAD: {bad}")
        for i in reversed(range(self.det_l.count())):
            w = self.det_l.itemAt(i).widget()
            if w: w.deleteLater()
        for i,p in enumerate(self.detected_pellets,1):
            txt = f"Pellet {i} - {'OK' if p['within'] else 'BAD'}\n  D:{p['diameter']:.3f}mm  L:{p['length']:.3f}mm  Conf:{p['confidence']:.1f}%"
            lbl = QLabel(txt)
            lbl.setStyleSheet(f"padding:5px;margin:2px;border:1px solid {'green' if p['within'] else 'red'};")
            self.det_l.addWidget(lbl)
        self.det_l.addStretch()

    def show_image(self,cv_img):
        avail_w = self.img_lbl.parent().width()-30
        h,w = cv_img.shape[:2]
        scale = avail_w / w
        pix = cv_to_qpixmap(cv_img,QSize(int(w*scale),int(h*scale)))
        self.img_lbl.setFixedWidth(int(w*scale))
        self.img_lbl.setMinimumHeight(int(h*scale))
        self.img_lbl.setPixmap(pix)

def main():
    app = QApplication(sys.argv)
    win = PelletMeasurementApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
