"""
camera.py - RTSP / Webcam / Video file stream handler.

Manages OpenCV VideoCapture with robust connection handling,
automatic reconnection on failure, clean resource cleanup, and
asynchronous background thread reading for RTSP streams.

Supports three source types (auto-detected from RTSP_URL config):
  - Integer (e.g. "0")    -> local webcam
  - File path             -> pre-recorded video file
  - RTSP URL              -> network CCTV camera
"""

import time
import os
import threading
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

# Module-level logger
logger = logging.getLogger(__name__)

# Default retry delay in seconds when connection fails
DEFAULT_RETRY_DELAY = 5.0

# Maximum consecutive read failures before triggering reconnect
MAX_READ_FAILURES = 30


class CameraStream:
    """
    Robust camera/RTSP stream handler using OpenCV VideoCapture.
    Supports asynchronous thread reading for RTSP streams to eliminate latency.
    """

    def __init__(
        self,
        source: str,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ):
        self.source = str(source)
        self.retry_delay = retry_delay
        self._cap: Optional[cv2.VideoCapture] = None
        self._consecutive_failures = 0
        self._is_connected = False

        self._source_type = self._detect_source_type(self.source)
        self._capture_arg = self._parse_source(self.source)

        # Threading for zero-latency RTSP stream reading
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._thread_running = False
        self._latest_frame: Optional[np.ndarray] = None

        logger.info("CameraStream initialized.")
        logger.info("  Source: %s", self.source)
        logger.info("  Source type: %s", self._source_type)
        logger.info("  Retry delay: %.1fs", self.retry_delay)

    @staticmethod
    def _detect_source_type(source: str) -> str:
        stripped = str(source).strip().lower()
        if stripped.isdigit() or stripped in ["0", "1", "webcam", "laptop", "laptop webcam"]:
            return "webcam"
        elif stripped.startswith("rtsp://"):
            return "rtsp"
        else:
            return "file"

    @staticmethod
    def _parse_source(source: str):
        stripped = str(source).strip().lower()
        if stripped.isdigit():
            return int(stripped)
        if stripped in ["webcam", "laptop", "laptop webcam"]:
            return 0
        return str(source).strip()

    def connect(self) -> bool:
        self.release()
        logger.info("Connecting to camera source: %s ...", self.source)

        try:
            if self._source_type == "rtsp":
                # Fast RTSP FFMPEG options
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|max_delay;500000"
                self._cap = cv2.VideoCapture(self._capture_arg, cv2.CAP_FFMPEG)
                if self._cap is not None:
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            elif self._source_type == "webcam":
                # Try DirectShow first on Windows for instant webcam opening
                self._cap = cv2.VideoCapture(self._capture_arg, cv2.CAP_DSHOW)
                if self._cap is None or not self._cap.isOpened():
                    self._cap = cv2.VideoCapture(self._capture_arg)
            else:
                self._cap = cv2.VideoCapture(self._capture_arg)
        except Exception as e:
            logger.error("Failed to create VideoCapture: %s", e)
            self._is_connected = False
            return False

        if self._cap is None or not self._cap.isOpened():
            logger.error("Could not open camera source: %s", self.source)
            self._is_connected = False
            return False

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)

        self._is_connected = True
        self._consecutive_failures = 0

        logger.info("Camera connected successfully!")
        logger.info("  Resolution: %dx%d", width, height)
        logger.info("  FPS: %.1f", fps if fps > 0 else 0.0)

        if self._source_type == "rtsp":
            self._start_reader_thread()

        return True

    def _start_reader_thread(self):
        self._thread_running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        while self._thread_running and self._cap and self._cap.isOpened():
            try:
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    with self._lock:
                        self._latest_frame = frame
                        self._consecutive_failures = 0
                else:
                    time.sleep(0.005)
            except Exception:
                time.sleep(0.01)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            self._is_connected = False
            return False, None

        if self._source_type == "rtsp":
            with self._lock:
                if self._latest_frame is not None:
                    return True, self._latest_frame.copy()
                return False, None

        try:
            ret, frame = self._cap.read()
        except Exception as e:
            logger.warning("Exception during frame read: %s", e)
            self._consecutive_failures += 1
            if self._consecutive_failures >= MAX_READ_FAILURES:
                self._is_connected = False
            return False, None

        if not ret or frame is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= MAX_READ_FAILURES:
                self._is_connected = False
            return False, None

        self._consecutive_failures = 0
        return True, frame

    def reconnect(self) -> bool:
        logger.info("Attempting reconnection in %.1f seconds...", self.retry_delay)
        time.sleep(self.retry_delay)
        return self.connect()

    def release(self):
        self._thread_running = False
        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
            self._thread = None

        if self._cap is not None:
            try:
                self._cap.release()
                logger.info("Camera capture released.")
            except Exception as e:
                logger.warning("Error releasing capture: %s", e)
            finally:
                self._cap = None
                self._is_connected = False
                self._latest_frame = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._cap is not None and self._cap.isOpened()

    @property
    def source_type(self) -> str:
        return self._source_type

    def get_properties(self) -> dict:
        if self._cap is None or not self._cap.isOpened():
            return {}

        return {
            "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self._cap.get(cv2.CAP_PROP_FPS),
            "backend": self._cap.getBackendName(),
        }

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
