import sys
import time  # For throttling
from collections import deque
import cv2
import numpy as np
import matplotlib.pyplot as plt  # this was used to see the intermediate results
import threading  # we removed the threading for now

from OCR_video import LiscencePlateExtractor  # OCR_video is the model engine for this demonstration
                                              # OCR_Image is the model engine for another demonstration

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QFileDialog, QSplitter, QLabel
)
from PyQt5.QtMultimedia import (
    QMediaPlayer, QMediaContent, QAbstractVideoSurface, QVideoFrame,
    QVideoSurfaceFormat, QAbstractVideoBuffer
)
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QImage, QPixmap


# -----------------------------------------------
# Custom Video Surface: Captures frames from the media stream.
# -----------------------------------------------
class VideoFrameSurface(QAbstractVideoSurface):
    def __init__(self, callback, parent=None):
        super(VideoFrameSurface, self).__init__(parent)
        self.callback = callback  # Function to call with each captured QImage

    def supportedPixelFormats(self, handleType=0):
        # We support only the RGB32 format.
        return [QVideoFrame.PixelFormat.Format_RGB32]

    def start(self, format: QVideoSurfaceFormat):
        return super(VideoFrameSurface, self).start(format)

    def present(self, frame: QVideoFrame) -> bool:
        if frame.isValid():
            if not frame.map(QAbstractVideoBuffer.ReadOnly):
                return False
            # Create a QImage from the frame data.
            image = QImage(
                frame.bits(),
                frame.width(),
                frame.height(),
                frame.bytesPerLine(),
                QImage.Format_RGB32
            ).copy()  # Deep copy the image
            frame.unmap()
            # Call the callback with the captured QImage.
            self.callback(image)
            return True
        return False


# -----------------------------------------------
# Custom Widget to Display Video Frames
# -----------------------------------------------
class VideoDisplayWidget(QLabel):
    def __init__(self, parent=None):
        super(VideoDisplayWidget, self).__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Video not playing...")

    def update_image(self, image: QImage):
        pixmap = QPixmap.fromImage(image)
        self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


# -----------------------------------------------
# Main Application Window
# -----------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Video Player with Number Plate Console")

        # Create the media player.
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        # Use our custom video display widget instead of QVideoWidget.
        self.videoDisplay = VideoDisplayWidget()

        # Create a text console.
        self.console = QTextEdit()
        self.console.setReadOnly(True)

        # Create control buttons.
        self.playButton = QPushButton("Play")
        self.pauseButton = QPushButton("Pause")
        self.loadButton = QPushButton("Load Video")

        self.playButton.clicked.connect(self.play_video)
        self.pauseButton.clicked.connect(self.pause_video)
        self.loadButton.clicked.connect(self.load_video)

        # Layout for video display and controls.
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.videoDisplay)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.loadButton)
        btn_layout.addWidget(self.playButton)
        btn_layout.addWidget(self.pauseButton)
        video_layout.addLayout(btn_layout)
        video_container = QWidget()
        video_container.setLayout(video_layout)

        # Splitter for video and console.
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(video_container)
        splitter.addWidget(self.console)
        splitter.setSizes([600, 200])
        self.setCentralWidget(splitter)

        # Set up the custom video surface.
        self.videoSurface = VideoFrameSurface(self.process_captured_frame)
        self.mediaPlayer.setVideoOutput(self.videoSurface)

        # Queue to store up to 10 recent number plate results
        self.plate_queue = deque(maxlen=10)

        # Variable to throttle frame processing (1 frame per second).
        self.lastFrameTime = 0

    def load_video(self):
        """Open a file dialog to load a video file."""
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video File")
        if filename:
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
            self.console.append(f"Loaded video: {filename}")

    def play_video(self):
        """Start video playback."""
        self.mediaPlayer.play()
        self.console.append("Playing video...")

    def pause_video(self):
        """Pause video playback."""
        self.mediaPlayer.pause()
        self.console.append("Paused video.")

    def qimage_to_cv_gray(self, qimage: QImage) -> np.ndarray:
        if qimage.isNull():
            return None

        # Convert QImage to an RGB888 format.
        qimage = qimage.convertToFormat(QImage.Format_RGB888)
        width = qimage.width()
        height = qimage.height()
        bpl = qimage.bytesPerLine()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())

        # Create a numpy array from the raw data.
        arr = np.array(ptr).reshape(height, bpl)
        # Extract only the pixel data (width * 3 bytes).
        arr = arr[:, :width * 3]

        try:
            rgb_image = arr.reshape((height, width, 3))
        except Exception as e:
            # If reshape fails, try to interpret as a 2D array (already grayscale)
            rgb_image = arr.reshape((height, width))

        # Check the number of channels:
        if rgb_image.ndim == 2:
            # Image is already grayscale.
            gray = rgb_image
        elif rgb_image.ndim == 3:
            if rgb_image.shape[2] == 3:
                # Normal 3-channel image: convert from RGB to grayscale.
                gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            elif rgb_image.shape[2] == 1:
                # Already single channel.
                gray = rgb_image.squeeze()
            else:
                # Unexpected number of channels; fallback to taking the first channel.
                gray = rgb_image[:, :, 0]
        else:
            gray = None

        return gray

    def process_captured_frame(self, q_image: QImage):
        """
        Callback function invoked by our custom video surface when a new frame is available.
        Processes one frame per second and plots it using matplotlib.
        """
        if q_image.isNull():
            self.console.append("Captured QImage is null.")
            return

        # Throttle: process a frame only if at least one second has elapsed.
        current_time = time.time()
        if current_time - self.lastFrameTime < 1:
            return
        self.lastFrameTime = current_time

        # Update the custom video display widget.
        self.videoDisplay.update_image(q_image)

        # Convert the QImage to grayscale.
        gray = self.qimage_to_cv_gray(q_image)
        if gray is not None:
            #     print(f"Captured frame: {gray.shape}, mean: {np.mean(gray)}")
            #     # Plot the grayscale frame using matplotlib.
            #     plt.ion()  # Enable interactive mode.
            #     fig = plt.figure("Captured Grayscale Frame", figsize=(8, 6))
            #
            #     plt.imshow(gray, cmap='gray')
            #     plt.title("Captured Grayscale Frame")
            #     plt.axis('off')
            #     plt.draw()
            #     plt.pause(0.001)
            # else:
            #     self.console.append("Gray conversion failed.")

            detector = LiscencePlateExtractor()
            result = detector.extract(gray)

            if len(result) == 0 or any(np.array_equal(result, item) for item in self.plate_queue):
                # If the result already exists, do not display it.
                return
            else:
                # If the result is new, append it to the queue.
                if len(self.plate_queue) < 10:
                    self.plate_queue.append(result)
                else:
                    self.plate_queue.popleft()
                    self.plate_queue.append(result)

                # Get the current video position in seconds (timestamp)
                timestamp = self.mediaPlayer.position() // 1000
                minute = timestamp // 60
                hour = minute // 60
                second = timestamp % 60

                # Display the number plate result and timestamp in the console.
                self.console.append(f"[{hour}:{minute}:{second} sec] recognised: {result}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())
