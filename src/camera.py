"""
camera.py - RTSP / Webcam / Video file stream handler.

Manages OpenCV VideoCapture with robust connection handling,
automatic reconnection on failure, and clean resource cleanup.

Supports three source types (auto-detected from RTSP_URL config):
  - Integer (e.g. "0")    -> local webcam
  - File path             -> pre-recorded video file
  - RTSP URL              -> network CCTV camera
"""

import time
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

    Handles connection, frame reading, automatic reconnection on
    failure, and clean resource release.
    """

    def __init__(
        self,
        source: str,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ):
        """
        Initialize the CameraStream.

        Args:
            source: Camera source string from config. Can be:
                - "0", "1", etc. for webcam index
                - A file path for a recorded video
                - An RTSP URL (e.g. rtsp://admin:pass@192.168.1.100:554/stream)
            retry_delay: Seconds to wait before retrying a failed connection.
        """
        self.source = source
        self.retry_delay = retry_delay
        self._cap: Optional[cv2.VideoCapture] = None
        self._consecutive_failures = 0
        self._is_connected = False

        # Parse source type for logging and special handling
        self._source_type = self._detect_source_type(source)
        self._capture_arg = self._parse_source(source)

        logger.info("CameraStream initialized.")
        logger.info("  Source: %s", self.source)
        logger.info("  Source type: %s", self._source_type)
        logger.info("  Retry delay: %.1fs", self.retry_delay)

    @staticmethod
    def _detect_source_type(source: str) -> str:
        """Detect whether the source is a webcam index, file, or RTSP URL."""
        stripped = source.strip()
        if stripped.isdigit():
            return "webcam"
        elif stripped.lower().startswith("rtsp://"):
            return "rtsp"
        else:
            return "file"

    @staticmethod
    def _parse_source(source: str):
        """
        Convert the source string to the appropriate type for
        cv2.VideoCapture (int for webcam, str for RTSP/file).
        """
        stripped = source.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped

    def connect(self) -> bool:
        """
        Open the video capture connection.

        Returns:
            True if connection was successful, False otherwise.
        """
        # Release any existing capture first
        self.release()

        logger.info("Connecting to camera source: %s ...", self.source)

        try:
            if self._source_type == "rtsp":
                # Use FFMPEG backend for RTSP streams for better compatibility
                self._cap = cv2.VideoCapture(self._capture_arg, cv2.CAP_FFMPEG)
            else:
                self._cap = cv2.VideoCapture(self._capture_arg)
        except Exception as e:
            logger.error("Failed to create VideoCapture: %s", e)
            self._is_connected = False
            return False

        if self._cap is None or not self._cap.isOpened():
            logger.error(
                "Could not open camera source: %s. "
                "Check that the source is valid and accessible.",
                self.source,
            )
            self._is_connected = False
            return False

        # Read camera properties for logging
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)

        self._is_connected = True
        self._consecutive_failures = 0

        logger.info("Camera connected successfully!")
        logger.info("  Resolution: %dx%d", width, height)
        logger.info("  FPS: %.1f", fps if fps > 0 else 0.0)

        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the video source.

        Returns:
            Tuple of (success, frame). On failure, frame is None.
            After MAX_READ_FAILURES consecutive failures, the stream
            is considered disconnected.
        """
        if self._cap is None or not self._cap.isOpened():
            self._is_connected = False
            return False, None

        try:
            ret, frame = self._cap.read()
        except Exception as e:
            logger.warning("Exception during frame read: %s", e)
            self._consecutive_failures += 1
            if self._consecutive_failures >= MAX_READ_FAILURES:
                logger.error(
                    "Exceeded %d consecutive read failures. Stream disconnected.",
                    MAX_READ_FAILURES,
                )
                self._is_connected = False
            return False, None

        if not ret or frame is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= MAX_READ_FAILURES:
                logger.error(
                    "Exceeded %d consecutive read failures. Stream disconnected.",
                    MAX_READ_FAILURES,
                )
                self._is_connected = False
            return False, None

        # Successful read — reset failure counter
        self._consecutive_failures = 0
        return True, frame

    def reconnect(self) -> bool:
        """
        Attempt to reconnect to the camera source after a failure.

        Waits for retry_delay seconds before attempting, to avoid
        busy-loop CPU waste.

        Returns:
            True if reconnection was successful, False otherwise.
        """
        logger.info(
            "Attempting reconnection in %.1f seconds...", self.retry_delay
        )
        time.sleep(self.retry_delay)
        return self.connect()

    def release(self):
        """Release the video capture resources."""
        if self._cap is not None:
            try:
                self._cap.release()
                logger.info("Camera capture released.")
            except Exception as e:
                logger.warning("Error releasing capture: %s", e)
            finally:
                self._cap = None
                self._is_connected = False

    @property
    def is_connected(self) -> bool:
        """Check whether the stream is currently connected and readable."""
        return self._is_connected and self._cap is not None and self._cap.isOpened()

    @property
    def source_type(self) -> str:
        """Return the detected source type: 'webcam', 'rtsp', or 'file'."""
        return self._source_type

    def get_properties(self) -> dict:
        """
        Return current capture properties as a dictionary.
        Returns empty dict if not connected.
        """
        if self._cap is None or not self._cap.isOpened():
            return {}

        return {
            "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self._cap.get(cv2.CAP_PROP_FPS),
            "backend": self._cap.getBackendName(),
        }

    def __enter__(self):
        """Context manager entry — connect on enter."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit — release on exit."""
        self.release()
        return False
