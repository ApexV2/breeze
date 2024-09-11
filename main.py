import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, 
                             QVBoxLayout, QWidget, QFrame, QHBoxLayout, QSlider, QStyle,
                             QComboBox, QSpinBox, QCheckBox, QGroupBox, QGridLayout, QScrollArea,
                             QSizePolicy, QSplitter, QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QMutex
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from ultralytics import YOLO
import logging

class FrameProcessor(QThread):
    finished = pyqtSignal(np.ndarray, list, tuple)

    def __init__(self, model, conf):
        super().__init__()
        self.model = model
        self.conf = conf
        self.frame = None
        self.mutex = QMutex()
        self.running = True

    def run(self):
        while self.running:
            self.mutex.lock()
            try:
                if self.frame is not None:
                    frame = self.frame.copy()
                    self.frame = None
                    self.mutex.unlock()
                    
                    original_dims = frame.shape[1::-1]
                    resized_frame = cv2.resize(frame, (640, 480))
                    
                    results = self.model(resized_frame, conf=self.conf)
                    
                    detections = [
                        (self.model.names[int(box.cls[0])], float(box.conf[0]), box.xyxy[0].tolist())
                        for r in results
                        for box in r.boxes
                    ]
                    
                    self.finished.emit(frame, detections, original_dims)
                else:
                    self.mutex.unlock()
                    self.msleep(10)
            except Exception as e:
                logging.error(f"Error in FrameProcessor: {str(e)}")
                self.mutex.unlock()

    def process_frame(self, frame):
        self.mutex.lock()
        self.frame = frame
        self.mutex.unlock()

    def stop(self):
        self.running = False

    def change_model(self, model):
        self.model = model

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, list)
    progress_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)

    def __init__(self, video_path, model_name):
        super().__init__()
        self.video_path = video_path
        self.model_name = model_name
        self.model = YOLO(f'{model_name}.pt')
        self.running = True
        self.paused = True
        self.frame_skip = 2
        self.confidence_threshold = 0.5
        self.processor = FrameProcessor(self.model, self.confidence_threshold)
        self.processor.finished.connect(self.process_result)
        self.processor.start()
        self.current_frame = 0
        self.total_frames = 0
        self.cap = None
        self.fps = 30

    def run(self):
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise IOError("Error opening video file")
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.current_frame = 0

            while self.running:
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        self.current_frame = 0
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

                    self.current_frame += 1
                    if self.current_frame % (self.frame_skip + 1) == 0:
                        self.processor.process_frame(frame)
                        progress = int((self.current_frame / self.total_frames) * 100)
                        self.progress_signal.emit(progress)
                else:
                    self.msleep(50)

                self.msleep(int(1000 / self.fps))

        except Exception as e:
            logging.error(f"Error in VideoThread: {str(e)}")
            self.error_signal.emit(str(e))
        finally:
            if self.cap:
                self.cap.release()
            self.processor.stop()
            self.processor.wait()

    def process_result(self, frame, detections, original_dims):
        orig_w, orig_h = original_dims
        scale_x = orig_w / 640
        scale_y = orig_h / 480

        scaled_detections = []
        for obj, conf, (x1, y1, x2, y2) in detections:
            x1_scaled = int(x1 * scale_x)
            y1_scaled = int(y1 * scale_y)
            x2_scaled = int(x2 * scale_x)
            y2_scaled = int(y2 * scale_y)
            
            cv2.rectangle(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)
            label = f"{obj} {conf:.2f}"
            cv2.putText(frame, label, (x1_scaled, y1_scaled - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            scaled_detections.append((obj, conf, (x1_scaled, y1_scaled, x2_scaled, y2_scaled)))
        
        self.change_pixmap_signal.emit(frame, scaled_detections)

    def stop(self):
        self.running = False
        self.processor.stop()
        self.processor.wait()

    def pause(self):
        self.paused = not self.paused

    def set_frame_skip(self, skip):
        self.frame_skip = skip

    def set_confidence(self, conf):
        self.confidence_threshold = conf
        self.processor.conf = conf

    def seek(self, position):
        if self.cap:
            self.current_frame = int(position * self.total_frames / 100)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

    def change_model(self, model_name):
        self.model_name = model_name
        self.model = YOLO(f'{model_name}.pt')
        self.processor.change_model(self.model)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("breeze")
        self.setMinimumSize(1200, 800)
        self.init_ui()
        self.center_on_screen()
        self.current_detections = []
        self.total_detections = {}
        self.set_style()

    def center_on_screen(self):
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        center_point = QApplication.desktop().screenGeometry(screen).center()
        frame_geometry = self.frameGeometry()
        frame_geometry.moveCenter(center_point)
        self.move(frame_geometry.topLeft())

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Video display
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(800, 600)

        # Controls
        controls_layout = QHBoxLayout()
        self.upload_button = QPushButton("Upload Video")
        self.upload_button.clicked.connect(self.upload_video)
        self.play_pause_button = QPushButton()
        self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_pause_button.clicked.connect(self.play_pause_video)
        self.play_pause_button.setEnabled(False)
        self.stop_button = QPushButton()
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(False)

        controls_layout.addWidget(self.upload_button)
        controls_layout.addWidget(self.play_pause_button)
        controls_layout.addWidget(self.stop_button)

        # Add Finish Testing button
        self.finish_testing_button = QPushButton("Finish Testing")
        self.finish_testing_button.clicked.connect(self.show_final_statistics)
        self.finish_testing_button.setEnabled(False)
        controls_layout.addWidget(self.finish_testing_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        # Settings
        settings_layout = QHBoxLayout()
        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(0, 10)
        self.frame_skip_spin.setValue(2)
        self.frame_skip_spin.valueChanged.connect(self.set_frame_skip)
        settings_layout.addWidget(QLabel("Frame Skip:"))
        settings_layout.addWidget(self.frame_skip_spin)

        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(1, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.set_confidence)
        settings_layout.addWidget(QLabel("Confidence:"))
        settings_layout.addWidget(self.confidence_slider)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
        self.model_combo.currentTextChanged.connect(self.change_model)
        settings_layout.addWidget(QLabel("Model:"))
        settings_layout.addWidget(self.model_combo)

        # Detection results
        results_layout = QHBoxLayout()
        self.detection_list = QLabel("Detected Objects:")
        self.detection_list.setAlignment(Qt.AlignTop)
        self.detection_stats = QLabel("Detection Statistics:")
        self.detection_stats.setAlignment(Qt.AlignTop)
        results_layout.addWidget(self.detection_list)
        results_layout.addWidget(self.detection_stats)

        # Status
        self.status_label = QLabel("Ready")

        # Add widgets to main layout
        main_layout.addWidget(self.label)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addLayout(settings_layout)
        main_layout.addLayout(results_layout)
        main_layout.addWidget(self.status_label)

    def set_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                font-size: 14px;
            }
            QPushButton {
                font-size: 14px;
                padding: 5px 10px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
            QProgressBar {
                border: 1px solid #999999;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
            }
        """)

    @pyqtSlot(np.ndarray, list)
    def update_image(self, cv_img, detections):
        try:
            qt_img = self.convert_cv_qt(cv_img)
            scaled_pixmap = QPixmap.fromImage(qt_img).scaled(
                self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)
            self.current_detections = detections
            self.update_detections(detections)
            self.update_total_detections(detections)
        except Exception as e:
            logging.error(f"Error updating image: {str(e)}")

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return convert_to_Qt_format

    def upload_video(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov)")
            if file_name:
                self.total_detections.clear()  # Reset total detections
                model_name = self.model_combo.currentText()
                self.thread = VideoThread(file_name, model_name)
                self.thread.change_pixmap_signal.connect(self.update_image)
                self.thread.progress_signal.connect(self.update_progress)
                self.thread.error_signal.connect(self.handle_error)
                self.thread.start()
                self.upload_button.setEnabled(False)
                self.play_pause_button.setEnabled(True)
                self.stop_button.setEnabled(True)
                self.finish_testing_button.setEnabled(True)
                self.status_label.setText(f"Processing: {file_name.split('/')[-1]}")
        except Exception as e:
            self.handle_error(str(e))

    def handle_error(self, error_message):
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")
        self.stop_video()

    def play_pause_video(self):
        if hasattr(self, 'thread'):
            self.thread.pause()
            if self.thread.paused:
                self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            else:
                self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def stop_video(self):
        if hasattr(self, 'thread'):
            self.thread.stop()
            self.thread.wait()
        self.upload_button.setEnabled(True)
        self.play_pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.finish_testing_button.setEnabled(True)  # Enable the button when video stops
        self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.status_label.setText("Video processing stopped")
        self.progress_bar.setValue(0)
        self.detection_list.setText("Detected Objects:")
        self.detection_stats.setText("Detection Statistics:")

    def seek_video(self):
        if hasattr(self, 'thread'):
            position = self.progress_bar.value()
            self.thread.seek(position)

    @pyqtSlot(int)
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def set_frame_skip(self, value):
        if hasattr(self, 'thread'):
            self.thread.set_frame_skip(value)

    def set_confidence(self, value):
        confidence = value / 100.0
        if hasattr(self, 'thread'):
            self.thread.set_confidence(confidence)

    def change_model(self, model_name):
        if hasattr(self, 'thread'):
            self.thread.change_model(model_name)

    @pyqtSlot(list)
    def update_detections(self, detections):
        detection_text = "Detected Objects:\n"
        stats = {}
        for obj, conf, _ in detections:
            detection_text += f"{obj}: {conf:.2f}\n"
            stats[obj] = stats.get(obj, 0) + 1
        self.detection_list.setText(detection_text)

        stats_text = "Detection Statistics:\n"
        for obj, count in stats.items():
            stats_text += f"{obj}: {count}\n"
        self.detection_stats.setText(stats_text)

    def update_total_detections(self, detections):
        for obj, _, _ in detections:
            self.total_detections[obj] = self.total_detections.get(obj, 0) + 1

    def show_final_statistics(self):
        stats_text = "Final Detection Statistics:\n\n"
        total_objects = sum(self.total_detections.values())
        
        for obj, count in sorted(self.total_detections.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_objects) * 100
            stats_text += f"{obj}: {count} ({percentage:.2f}%)\n"
        
        stats_text += f"\nTotal objects detected: {total_objects}"
        
        QMessageBox.information(self, "Final Statistics", stats_text)

    def closeEvent(self, event):
        if hasattr(self, 'thread'):
            self.thread.stop()
            self.thread.wait()
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'label') and self.label.pixmap():
            scaled_pixmap = self.label.pixmap().scaled(
                self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())