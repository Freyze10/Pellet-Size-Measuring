from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor

class PelletMeasurementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Inspector Pro")
        self.setGeometry(100, 100, 1600, 1000)
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f7fa; }
            QPushButton {
                background-color: #4361ee; color: white; border: none; padding: 12px;
                border-radius: 8px; font-weight: bold; font-size: 14px;
            }
            QPushButton:hover { background-color: #3a56d4; }
            QPushButton#loadBtn { background-color: #7209b7; padding: 15px; font-size: 16px; }
            QPushButton#loadBtn:hover { background-color: #5a0896; }
            QGroupBox {
                font-weight: bold; border: 2px solid #e0e0e0; border-radius: 10px;
                margin-top: 10px; padding-top: 10px; background-color: white;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 10px; }
            QLabel#statLabel { font-size: 18px; font-weight: bold; padding: 10px; border-radius: 8px; color: white; }
            QScrollArea { border: none; background: transparent; }
            QLabel#detailCard {
                background-color: white; border: 1px solid #ddd; border-left: 6px solid;
                border-radius: 8px; padding: 12px; margin: 6px; font-family: 'Segoe UI';
            }
            QDoubleSpinBox { padding: 8px; border: 1px solid #ccc; border-radius: 6px; font-size: 14px; }
        """)

        self.pixels_per_mm = 25.4
        self.target_diameter = 3.0
        self.target_length = 3.0
        self.tolerance = 0.5
        self.update_ranges()

        self.current_image = None
        self.detected_pellets = []
        self.yolo_detector = None
        self.cv_detector = ColorPelletDetector()

        self.init_ui()
        self.load_model("trained_model/best.pt")

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # === Left Panel ===
        left = self.create_left_panel()
        left.setMaximumWidth(420)
        left.setStyleSheet("background-color: white; border-radius: 12px; padding: 10px;")

        # === Right Panel (Image) ===
        right = self.create_right_panel()

        main_layout.addWidget(left, 1)
        main_layout.addWidget(right, 3)

    def create_left_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title = QLabel("Pellet Inspector Pro")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #2b2d42; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Load Image Button (Prominent)
        self.load_btn = QPushButton("Load Image")
        self.load_btn.setObjectName("loadBtn")
        self.load_btn.setIcon(QIcon.fromTheme("document-open"))
        self.load_btn.setFixedHeight(56)
        self.load_btn.clicked.connect(self.load_image)
        layout.addWidget(self.load_btn)

        # Calibration
        calib_box = QGroupBox("Calibration Settings")
        calib_layout = QFormLayout()
        calib_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.px_spin = QDoubleSpinBox()
        self.px_spin.setRange(0.1, 300)
        self.px_spin.setDecimals(2)
        self.px_spin.setSuffix(" px/mm")
        self.px_spin.setValue(self.pixels_per_mm)
        self.px_spin.setSingleStep(0.5)
        self.px_spin.valueChanged.connect(lambda v: setattr(self, 'pixels_per_mm', v))
        calib_layout.addRow("Pixels per mm:", self.px_spin)
        calib_box.setLayout(calib_layout)
        layout.addWidget(calib_box)

        # Target Dimensions
        target_box = QGroupBox("Target Dimensions")
        tlayout = QGridLayout()
        labels = ["Diameter (mm):", "Length (mm):", "Tolerance (±mm):"]
        values = [self.target_diameter, self.target_length, self.tolerance]
        self.spins = []
        for i, (label, val) in enumerate(zip(labels, values)):
            lbl = QLabel(label)
            spin = QDoubleSpinBox()
            spin.setRange(0.1, 50)
            spin.setValue(val)
            spin.setSingleStep(0.1)
            if i < 2:
                spin.valueChanged.connect(self.update_ranges)
            else:
                spin.valueChanged.connect(lambda v: setattr(self, 'tolerance', v) or self.update_ranges())
            self.spins.append(spin)
            tlayout.addWidget(lbl, i, 0)
            tlayout.addWidget(spin, i, 1)
        target_box.setLayout(tlayout)
        layout.addWidget(target_box)

        # Stats Cards
        stats_grid = QGridLayout()
        stats_grid.setSpacing(10)

        self.total_lbl = QLabel("0")
        self.total_lbl.setObjectName("statLabel")
        self.total_lbl.setStyleSheet("background-color: #4361ee;")
        self.total_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.ok_lbl = QLabel("0")
        self.ok_lbl.setObjectName("statLabel")
        self.ok_lbl.setStyleSheet("background-color: #06d6a0;")
        self.ok_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.bad_lbl = QLabel("0")
        self.bad_lbl.setObjectName("statLabel")
        self.bad_lbl.setStyleSheet("background-color: #ef476f;")
        self.bad_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        stats_grid.addWidget(QLabel("Total Pellets"), 0, 0)
        stats_grid.addWidget(QLabel("Good (OK)"), 0, 1)
        stats_grid.addWidget(QLabel("Defective"), 0, 2)
        stats_grid.addWidget(self.total_lbl, 1, 0)
        stats_grid.addWidget(self.ok_lbl, 1, 1)
        stats_grid.addWidget(self.bad_lbl, 1, 2)

        stats_group = QGroupBox("Detection Summary")
        stats_group.setLayout(stats_grid)
        layout.addWidget(stats_group)

        # Pellet Details
        details_box = QGroupBox("Detected Pellets")
        details_layout = QVBoxLayout()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(300)
        self.det_container = QWidget()
        self.det_layout = QVBoxLayout(self.det_container)
        self.det_layout.addStretch()
        self.scroll.setWidget(self.det_container)
        details_layout.addWidget(self.scroll)
        details_box.setLayout(details_layout)
        layout.addWidget(details_box)

        layout.addStretch()
        return widget

    def create_right_panel(self):
        widget = QWidget()
        widget.setStyleSheet("background-color: white; border-radius: 12px; overflow: hidden;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Inspection View")
        header.setStyleSheet("""
            background-color: #4361ee; color: white; padding: 16px; 
            font-size: 18px; font-weight: bold;
        """)
        layout.addWidget(header)

        self.img_lbl = QLabel("Click 'Load Image' to begin inspection")
        self.img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_lbl.setStyleSheet("""
            font-size: 18px; color: #888; background-color: #f8f9fa;
            border: 3px dashed #ccc; border-radius: 12px; margin: 20px;
        """)
        self.img_lbl.setMinimumSize(600, 600)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.img_lbl)
        scroll.setStyleSheet("border: none;")
        layout.addWidget(scroll, 1)
        return widget

    def update_stats(self):
        total = len(self.detected_pellets)
        ok = sum(1 for p in self.detected_pellets if p['within'])
        bad = total - ok

        self.total_lbl.setText(str(total))
        self.ok_lbl.setText(str(ok))
        self.bad_lbl.setText(str(bad))

        # Clear previous details
        for i in reversed(range(self.det_layout.count())):
            child = self.det_layout.itemAt(i).widget()
            if child: child.deleteLater()

        for i, p in enumerate(self.detected_pellets, 1):
            status = "OK" if p['within'] else "BAD"
            color = "#06d6a0" if p['within'] else "#ef476f"
            card = QLabel(
                f"<b>Pellet {i}</b> → <span style='color:{color}; font-size:16px'>{status}</span><br>"
                f"• Diameter: <b>{p['diameter']:.3f} mm</b> "
                f"(Target: {self.target_diameter} ± {self.tolerance})<br>"
                f"• Length: <b>{p['length']:.3f} mm</b> "
                f"(Target: {self.target_length} ± {self.tolerance})<br>"
                f"• Confidence: <b>{p['confidence']:.1f}%</b>"
            )
            card.setObjectName("detailCard")
            card.setStyleSheet(f"""
                border-left-color: {color}; 
                background-color: #fdfdfd; 
                font-size: 14px;
            """)
            card.setWordWrap(True)
            self.det_layout.insertWidget(self.det_layout.count() - 1, card)

    def show_image(self, cv_img):
        if cv_img is None:
            return
        h, w = cv_img.shape[:2]
        avail_w = self.img_lbl.width() - 40
        avail_h = self.img_lbl.height() - 40
        scale = min(avail_w / w, avail_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)

        pixmap = cv_to_qpixmap(cv_img, QSize(new_w, new_h))
        self.img_lbl.setPixmap(pixmap)
        self.img_lbl.setStyleSheet("border: none; background-color: #f8f9fa;")

    # Keep your existing methods: load_image, process_image, draw_pellet, etc.
    # Just make sure update_ranges() also updates spin values if needed
    def update_ranges(self):
        self.d_min = self.target_diameter - self.tolerance
        self.d_max = self.target_diameter + self.tolerance
        self.l_min = self.target_length - self.tolerance
        self.l_max = self.target_length + self.tolerance

        # Sync spinboxes
        self.target_diameter = self.spins[0].value()
        self.target_length = self.spins[1].value()
        self.tolerance = self.spins[2].value()