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
    """Machine learning-like detector using labeled samples"""

    def __init__(self):
        self.trained_samples = []
        self.feature_detector = cv2.SIFT_create()

    def train_from_coco(self, coco_data, images_folder=""):
        """Extract pellet samples from COCO labeled images"""
        self.trained_samples = []

        # Group annotations by image
        images_dict = {img['id']: img for img in coco_data.get('images', [])}

        for ann in coco_data.get('annotations', []):
            image_id = ann['image_id']
            if image_id not in images_dict:
                continue

            # Load the training image
            img_info = images_dict[image_id]
            img_path = os.path.join(images_folder, img_info['file_name'])

            if not os.path.exists(img_path):
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue

            # Extract pellet region using polygon
            if 'segmentation' in ann and ann['segmentation']:
                for seg in ann['segmentation']:
                    polygon = np.array(seg).reshape(-1, 2).astype(np.int32)

                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(polygon)

                    # Extract pellet sample with padding
                    padding = 10
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(image.shape[1], x + w + padding)
                    y2 = min(image.shape[0], y + h + padding)

                    pellet_sample = image[y1:y2, x1:x2].copy()

                    if pellet_sample.size == 0:
                        continue

                    # Create mask for the pellet
                    mask = np.zeros(pellet_sample.shape[:2], dtype=np.uint8)
                    polygon_shifted = polygon - [x1, y1]
                    cv2.fillPoly(mask, [polygon_shifted], 255)

                    # Store sample with features
                    gray_sample = cv2.cvtColor(pellet_sample, cv2.COLOR_BGR2GRAY)
                    keypoints, descriptors = self.feature_detector.detectAndCompute(gray_sample, mask)

                    if descriptors is not None and len(keypoints) > 5:
                        self.trained_samples.append({
                            'image': pellet_sample,
                            'mask': mask,
                            'gray': gray_sample,
                            'keypoints': keypoints,
                            'descriptors': descriptors,
                            'size': (w, h),
                            'polygon': polygon_shifted
                        })

        print(f"✓ Trained with {len(self.trained_samples)} pellet samples")
        return len(self.trained_samples) > 0

    def detect_pellets(self, image):
        """Detect pellets in a new image using learned features"""
        if not self.trained_samples:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Multi-scale detection
        detections = []

        # Method 1: Feature matching
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)

        if descriptors is not None:
            bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

            for sample in self.trained_samples:
                matches = bf_matcher.knnMatch(sample['descriptors'], descriptors, k=2)

                # Apply ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                # If enough good matches, find pellet location
                if len(good_matches) > 8:
                    src_pts = np.float32([sample['keypoints'][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    # Find homography
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if M is not None:
                        h, w = sample['gray'].shape
                        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)

                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(dst.astype(np.int32))

                        # Validate detection
                        if w > 10 and h > 10 and x >= 0 and y >= 0:
                            detections.append({
                                'bbox': (x, y, w, h),
                                'polygon': dst.reshape(-1, 2).astype(np.int32),
                                'confidence': len(good_matches),
                                'method': 'feature_matching'
                            })

        # Method 2: Contour-based detection with shape matching
        contour_detections = self.detect_by_contours(image, gray)
        detections.extend(contour_detections)

        # Remove duplicate detections
        detections = self.non_max_suppression(detections)

        return detections

    def detect_by_contours(self, image, gray):
        """Detect pellets using contour matching"""
        detections = []

        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get average sample size
        if self.trained_samples:
            avg_w = np.mean([s['size'][0] for s in self.trained_samples])
            avg_h = np.mean([s['size'][1] for s in self.trained_samples])
            min_area = (avg_w * avg_h) * 0.3  # 30% of average size
            max_area = (avg_w * avg_h) * 3.0  # 300% of average size
        else:
            min_area = 100
            max_area = 10000

        for contour in contours:
            area = cv2.contourArea(contour)

            if min_area <= area <= max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Check aspect ratio (pellets are roughly cylindrical)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

                if 1.0 <= aspect_ratio <= 4.0:  # Reasonable aspect ratio for pellets
                    # Compare shape with trained samples
                    match_score = self.compare_with_samples(contour)

                    if match_score > 0.5:  # Similarity threshold
                        detections.append({
                            'bbox': (x, y, w, h),
                            'polygon': contour.reshape(-1, 2),
                            'confidence': match_score * 100,
                            'method': 'contour_matching'
                        })

        return detections

    def compare_with_samples(self, contour):
        """Compare contour shape with trained samples"""
        if not self.trained_samples:
            return 0.5

        scores = []
        for sample in self.trained_samples[:10]:  # Compare with first 10 samples
            # Create a simple contour from sample polygon
            sample_contour = sample['polygon'].reshape(-1, 1, 2)

            # Shape matching using Hu moments
            try:
                score = cv2.matchShapes(contour, sample_contour, cv2.CONTOURS_MATCH_I2, 0)
                # Convert to similarity (lower score is better match)
                similarity = 1.0 / (1.0 + score)
                scores.append(similarity)
            except:
                continue

        return max(scores) if scores else 0.5

    def non_max_suppression(self, detections, overlap_threshold=0.5):
        """Remove overlapping detections"""
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        keep = []
        for i, det in enumerate(detections):
            x1, y1, w1, h1 = det['bbox']

            is_duplicate = False
            for kept_det in keep:
                x2, y2, w2, h2 = kept_det['bbox']

                # Calculate IoU (Intersection over Union)
                xi1 = max(x1, x2)
                yi1 = max(y1, y2)
                xi2 = min(x1 + w1, x2 + w2)
                yi2 = min(y1 + h1, y2 + h2)

                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                box1_area = w1 * h1
                box2_area = w2 * h2
                union_area = box1_area + box2_area - inter_area

                iou = inter_area / union_area if union_area > 0 else 0

                if iou > overlap_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep.append(det)

        return keep


class PelletMeasurementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pellet Size Measurement System - ML Detection")
        self.setGeometry(100, 100, 1400, 800)

        # Configuration
        self.pixels_per_mm = 6.0
        self.target_diameter = 3.0
        self.target_length = 3.0
        self.tolerance = 0.5

        # Data storage
        self.current_image = None
        self.current_image_path = None
        self.coco_data = None
        self.detected_pellets = []
        self.images_folder = ""

        # ML Detector
        self.detector = PelletDetector()
        self.is_trained = False

        # Setup UI
        self.init_ui()

        # Load and train
        self.load_and_train()

    def load_and_train(self):
        """Load COCO annotations and train the detector"""
        json_path = "pellets_label.json"
        if not os.path.exists(json_path):
            QMessageBox.warning(self, "Warning", f"{json_path} not found!")
            return

        try:
            with open(json_path, 'r') as f:
                self.coco_data = json.load(f)

            # Ask for images folder
            self.images_folder = QFileDialog.getExistingDirectory(
                self, "Select folder with labeled training images"
            )

            if self.images_folder:
                self.progress_label.setText("Training detector with labeled samples...")
                QApplication.processEvents()

                self.is_trained = self.detector.train_from_coco(self.coco_data, self.images_folder)

                if self.is_trained:
                    self.progress_label.setText(f"✓ Trained with {len(self.detector.trained_samples)} samples")
                    self.load_btn.setEnabled(True)
                else:
                    self.progress_label.setText("✗ Training failed - no valid samples found")
            else:
                self.progress_label.setText("Training folder not selected")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading annotations: {e}")

    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left panel - Controls and Stats
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)

        # Right panel - Image display
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 3)

    def create_left_panel(self):
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Training status
        self.progress_label = QLabel("Initializing...")
        self.progress_label.setStyleSheet("padding: 5px; background-color: #e0e0e0;")
        layout.addWidget(self.progress_label)

        # Load Image Button
        self.load_btn = QPushButton("Load Pellet Image for Detection")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setStyleSheet("QPushButton { padding: 10px; font-size: 14px; }")
        self.load_btn.setEnabled(False)
        layout.addWidget(self.load_btn)

        # Calibration Group
        calib_group = QGroupBox("Calibration")
        calib_layout = QVBoxLayout()

        # Pixels per mm
        px_layout = QHBoxLayout()
        px_label = QLabel("Pixels per mm:")
        self.px_spinbox = QDoubleSpinBox()
        self.px_spinbox.setRange(0.1, 100.0)
        self.px_spinbox.setValue(self.pixels_per_mm)
        self.px_spinbox.setSingleStep(0.1)
        self.px_spinbox.setDecimals(2)
        self.px_spinbox.valueChanged.connect(self.update_calibration)
        px_layout.addWidget(px_label)
        px_layout.addWidget(self.px_spinbox)
        calib_layout.addLayout(px_layout)

        calib_group.setLayout(calib_layout)
        layout.addWidget(calib_group)

        # Target Specifications Group
        spec_group = QGroupBox("Target Specifications")
        spec_layout = QVBoxLayout()

        spec_layout.addWidget(QLabel(f"Target Diameter: {self.target_diameter} mm"))
        spec_layout.addWidget(QLabel(f"Target Length: {self.target_length} mm"))
        spec_layout.addWidget(QLabel(f"Tolerance: ±{self.tolerance} mm"))
        spec_layout.addWidget(QLabel(
            f"Acceptable: {self.target_diameter - self.tolerance:.1f}-{self.target_diameter + self.tolerance:.1f} mm"))

        spec_group.setLayout(spec_layout)
        layout.addWidget(spec_group)

        # Statistics Group
        self.stats_group = QGroupBox("Detection Statistics")
        self.stats_layout = QVBoxLayout()

        self.total_label = QLabel("Total Pellets: 0")
        self.within_label = QLabel("Within Tolerance: 0")
        self.out_label = QLabel("Out of Tolerance: 0")
        self.status_label = QLabel("Status: No image loaded")
        self.status_label.setStyleSheet("font-weight: bold; padding: 5px;")

        self.stats_layout.addWidget(self.total_label)
        self.stats_layout.addWidget(self.within_label)
        self.stats_layout.addWidget(self.out_label)
        self.stats_layout.addWidget(self.status_label)

        self.stats_group.setLayout(self.stats_layout)
        layout.addWidget(self.stats_group)

        # Pellet Details Group (scrollable)
        details_group = QGroupBox("Pellet Details")
        details_layout = QVBoxLayout()

        self.details_scroll = QScrollArea()
        self.details_scroll.setWidgetResizable(True)
        self.details_widget = QWidget()
        self.details_widget_layout = QVBoxLayout()
        self.details_widget.setLayout(self.details_widget_layout)
        self.details_scroll.setWidget(self.details_widget)

        details_layout.addWidget(self.details_scroll)
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        layout.addStretch()

        return panel

    def create_right_panel(self):
        """Create right image display panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Image label
        self.image_label = QLabel("Train the model first, then load an image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.image_label.setMinimumSize(800, 600)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)

        layout.addWidget(scroll_area)

        return panel

    def load_image(self):
        """Load an image file for detection"""
        if not self.is_trained:
            QMessageBox.warning(self, "Not Trained", "Please train the detector first!")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Pellet Image for Detection",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            self.current_image_path = file_path
            self.current_image = cv2.imread(file_path)

            if self.current_image is not None:
                self.progress_label.setText("Detecting pellets...")
                QApplication.processEvents()
                self.process_image()
                self.progress_label.setText("✓ Detection complete")
            else:
                self.image_label.setText("Error loading image")

    def update_calibration(self, value):
        """Update calibration and reprocess image"""
        self.pixels_per_mm = value
        if self.current_image is not None:
            self.process_image()

    def process_image(self):
        """Process the image and detect pellets using ML"""
        if self.current_image is None or not self.is_trained:
            return

        # Detect pellets using trained model
        detections = self.detector.detect_pellets(self.current_image)

        # Create a copy for drawing
        display_image = self.current_image.copy()
        self.detected_pellets = []

        # Process each detection
        for detection in detections:
            x, y, w, h = detection['bbox']
            polygon = detection['polygon']

            # Calculate dimensions in mm
            width_mm = w / self.pixels_per_mm
            height_mm = h / self.pixels_per_mm

            # Determine diameter and length
            diameter = min(width_mm, height_mm)
            length = max(width_mm, height_mm)

            # Check tolerance
            within_tolerance = (
                    (self.target_diameter - self.tolerance <= diameter <= self.target_diameter + self.tolerance) and
                    (self.target_length - self.tolerance <= length <= self.target_length + self.tolerance)
            )

            pellet_info = {
                'polygon': polygon,
                'bbox': (x, y, w, h),
                'diameter': diameter,
                'length': length,
                'within_tolerance': within_tolerance,
                'confidence': detection['confidence']
            }

            self.detected_pellets.append(pellet_info)

            # Draw on image
            self.draw_pellet(display_image, pellet_info)

        # Update statistics
        self.update_statistics()

        # Display image
        self.display_image(display_image)

    def draw_pellet(self, image, pellet):
        """Draw pellet detection on image"""
        polygon = pellet['polygon']
        x, y, w, h = pellet['bbox']
        diameter = pellet['diameter']
        length = pellet['length']
        within_tolerance = pellet['within_tolerance']

        # Color based on tolerance
        if within_tolerance:
            color = (0, 255, 0)  # Green
            status_color = (255, 255, 255)
        else:
            color = (0, 0, 255)  # Red
            status_color = (255, 255, 255)

        # Draw polygon with semi-transparent fill
        overlay = image.copy()
        cv2.fillPoly(overlay, [polygon.reshape(-1, 1, 2)], (255, 255, 0))
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

        # Draw polygon outline
        cv2.polylines(image, [polygon.reshape(-1, 1, 2)], True, (0, 255, 255), 2)

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # Draw measurements
        bg_y = max(y - 50, 0)
        cv2.rectangle(image, (x, bg_y), (x + 130, y - 5), (0, 0, 0), -1)
        cv2.putText(image, f"D: {diameter:.2f}mm", (x + 5, bg_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        cv2.putText(image, f"L: {length:.2f}mm", (x + 5, bg_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        cv2.putText(image, f"C: {pellet['confidence']:.0f}", (x + 5, bg_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        # Warning indicator if out of tolerance
        if not within_tolerance:
            cv2.circle(image, (x + w - 10, y + 10), 8, (0, 0, 255), -1)
            cv2.putText(image, "!", (x + w - 14, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def update_statistics(self):
        """Update statistics display"""
        total = len(self.detected_pellets)
        within = sum(1 for p in self.detected_pellets if p['within_tolerance'])
        out_of = total - within

        self.total_label.setText(f"Total Pellets: {total}")
        self.within_label.setText(f"Within Tolerance: {within}")
        self.out_label.setText(f"Out of Tolerance: {out_of}")

        if total == 0:
            self.status_label.setText("Status: No pellets detected")
            self.status_label.setStyleSheet("font-weight: bold; padding: 5px; color: gray;")
        elif out_of == 0:
            self.status_label.setText("Status: ✓ All Within Tolerance")
            self.status_label.setStyleSheet("font-weight: bold; padding: 5px; color: green;")
        else:
            self.status_label.setText(f"Status: ✗ {out_of} Out of Tolerance")
            self.status_label.setStyleSheet("font-weight: bold; padding: 5px; color: red;")

        # Update pellet details
        self.update_pellet_details()

    def update_pellet_details(self):
        """Update detailed pellet information"""
        # Clear existing details
        while self.details_widget_layout.count():
            child = self.details_widget_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add pellet details
        for i, pellet in enumerate(self.detected_pellets, 1):
            detail_text = f"Pellet {i}:\n"
            detail_text += f"  Diameter: {pellet['diameter']:.2f} mm\n"
            detail_text += f"  Length: {pellet['length']:.2f} mm\n"
            detail_text += f"  Confidence: {pellet['confidence']:.0f}\n"
            detail_text += f"  Status: {'✓ OK' if pellet['within_tolerance'] else '✗ Out'}"

            label = QLabel(detail_text)
            label.setStyleSheet(
                f"padding: 5px; margin: 2px; border: 1px solid "
                f"{'green' if pellet['within_tolerance'] else 'red'};"
            )
            self.details_widget_layout.addWidget(label)

        self.details_widget_layout.addStretch()

    def display_image(self, cv_image):
        """Display OpenCV image in QLabel"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        # Convert to QImage
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Convert to QPixmap and display
        pixmap = QPixmap.fromImage(qt_image)

        # Scale to fit while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setMinimumSize(1, 1)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = PelletMeasurementApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()