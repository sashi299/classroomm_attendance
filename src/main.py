"""
main.py - Entry point for the Classroom Attendance System.

Initializes all components and runs the main pipeline:
  Camera Source -> Face Detection/Recognition -> Attendance Marking -> Live Display

Usage:
    python src/main.py

Press Q in the video window to safely exit.
"""

import sys
import time
import logging

import cv2
import numpy as np

from config import Config
from camera import CameraStream
from face_engine import FaceEngine
from database import DatabaseManager
from attendance import AttendanceManager, AttendanceStatus

# ── Logging Configuration ─────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

# ── UI Drawing Constants ──────────────────────────────────────
# Colors in BGR format
COLOR_RECOGNIZED = (0, 200, 0)       # Green for recognized students
COLOR_UNKNOWN = (0, 0, 220)          # Red for unknown faces
COLOR_NEWLY_MARKED = (0, 255, 255)   # Yellow/Cyan highlight for new attendance
COLOR_TEXT = (255, 255, 255)         # White text
COLOR_FPS = (0, 255, 255)           # Yellow for FPS counter
COLOR_STATUS_BAR = (40, 40, 40)     # Dark gray status bar

# Font settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_NAME = 0.6
FONT_SCALE_FPS = 0.5
FONT_SCALE_INFO = 0.45
FONT_SCALE_ATTENDANCE = 0.5
FONT_THICKNESS = 1
BOX_THICKNESS = 2


def draw_recognition_overlay(
    frame: np.ndarray,
    results_with_status: list,
    fps: float,
    registered_count: int,
    today_attendance_count: int,
) -> np.ndarray:
    """
    Draw bounding boxes, names, distance scores, attendance status, and status bar.

    Args:
        frame: The BGR frame to draw on (modified in-place).
        results_with_status: List of tuples (RecognitionResult, AttendanceStatus).
        fps: Current frames-per-second measurement.
        registered_count: Number of registered students.
        today_attendance_count: Total students marked present today.

    Returns:
        The annotated frame.
    """
    h, w = frame.shape[:2]

    for result, att_status in results_with_status:
        top, right, bottom, left = result.face_location

        if result.is_recognized:
            color = COLOR_RECOGNIZED
            label = f"{result.student_id} - {result.student_name}"
            conf_text = f"Dist: {result.distance:.3f}"
        else:
            color = COLOR_UNKNOWN
            label = "Unknown"
            conf_text = f"Dist: {result.distance:.3f}"

        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, BOX_THICKNESS)

        # Draw name label background & text above bounding box
        label_size, _ = cv2.getTextSize(label, FONT, FONT_SCALE_NAME, FONT_THICKNESS)
        label_w, label_h = label_size
        label_y = max(top - 10, label_h + 6)
        cv2.rectangle(
            frame,
            (left, label_y - label_h - 6),
            (left + label_w + 8, label_y + 2),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame, label,
            (left + 4, label_y - 4),
            FONT, FONT_SCALE_NAME, COLOR_TEXT, FONT_THICKNESS, cv2.LINE_AA,
        )

        # Draw distance/confidence score below bounding box
        cv2.putText(
            frame, conf_text,
            (left, bottom + 16),
            FONT, FONT_SCALE_INFO, color, FONT_THICKNESS, cv2.LINE_AA,
        )

        # Draw Attendance Status string (PRESENT / ALREADY PRESENT)
        if result.is_recognized and att_status is not None:
            if att_status == AttendanceStatus.NEWLY_MARKED:
                status_str = f"PRESENT - {result.student_name}"
                status_color = COLOR_NEWLY_MARKED
            elif att_status == AttendanceStatus.ALREADY_PRESENT:
                status_str = f"ALREADY PRESENT - {result.student_name}"
                status_color = COLOR_RECOGNIZED
            else:
                status_str = ""
                status_color = color

            if status_str:
                cv2.putText(
                    frame, status_str,
                    (left, bottom + 32),
                    FONT, FONT_SCALE_ATTENDANCE, status_color, FONT_THICKNESS + 1, cv2.LINE_AA,
                )

    # ── Status Bar (top of frame) ─────────────────────────────
    bar_height = 30
    cv2.rectangle(frame, (0, 0), (w, bar_height), COLOR_STATUS_BAR, cv2.FILLED)

    # FPS counter (left)
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame, fps_text,
        (10, 20),
        FONT, FONT_SCALE_FPS, COLOR_FPS, FONT_THICKNESS, cv2.LINE_AA,
    )

    # Attendance & face counts (center)
    status_text = f"Faces: {len(results_with_status)} | Registered: {registered_count} | Present Today: {today_attendance_count}"
    cv2.putText(
        frame, status_text,
        (w // 2 - 140, 20),
        FONT, FONT_SCALE_FPS, COLOR_TEXT, FONT_THICKNESS, cv2.LINE_AA,
    )

    # Exit instruction (right)
    exit_text = "Press Q to exit"
    exit_size, _ = cv2.getTextSize(exit_text, FONT, FONT_SCALE_FPS, FONT_THICKNESS)
    cv2.putText(
        frame, exit_text,
        (w - exit_size[0] - 10, 20),
        FONT, FONT_SCALE_FPS, (150, 150, 150), FONT_THICKNESS, cv2.LINE_AA,
    )

    return frame


def main():
    """Main application entry point."""
    logger.info("=" * 60)
    logger.info("Classroom Attendance System - Starting")
    logger.info("=" * 60)

    # ── Load Configuration ────────────────────────────────────
    config = Config()
    logger.info("Configuration loaded.")
    logger.info("  Camera source : %s", config.RTSP_URL)
    logger.info("  Threshold     : %.2f", config.RECOGNITION_THRESHOLD)
    logger.info("  Students dir  : %s", config.KNOWN_STUDENTS_DIR)
    logger.info("  MySQL Host    : %s:%d", config.MYSQL_HOST, config.MYSQL_PORT)
    logger.info("  MySQL DB      : %s", config.MYSQL_DATABASE)

    # ── Initialize Database & Attendance Manager ──────────────
    db_manager = DatabaseManager(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        database=config.MYSQL_DATABASE,
    )

    db_connected = db_manager.connect()
    if not db_connected:
        logger.warning(
            "Database connection failed! Attendance recording will be disabled until MySQL is available."
        )

    attendance_manager = AttendanceManager(db_manager=db_manager)

    # ── Initialize Face Engine ────────────────────────────────
    students_dir = config.KNOWN_STUDENTS_DIR if os.path.exists(config.KNOWN_STUDENTS_DIR) else config.get_department_students_dir("CSE")
    face_engine = FaceEngine(
        known_students_dir=students_dir,
        recognition_threshold=config.RECOGNITION_THRESHOLD,
    )
    num_registered = face_engine.load_registered_students()
    logger.info("Face engine ready. %d student(s) registered.", num_registered)

    if num_registered == 0:
        logger.warning(
            "No students registered! Place photos in '%s/' with format {ID}_{NAME}.jpg.",
            config.KNOWN_STUDENTS_DIR,
        )

    # ── Initialize Camera ─────────────────────────────────────
    camera = CameraStream(source=config.RTSP_URL)

    if not camera.connect():
        logger.error("Failed to connect to camera source: %s", config.RTSP_URL)
        return 1

    # ── Main Loop ─────────────────────────────────────────────
    logger.info("Starting live pipeline. Press Q in the video window to exit.")

    window_name = "Classroom Attendance System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()
    fps_update_interval = 1.0

    recognition_interval = max(1, config.FRAME_SKIP)
    last_results_with_status = []

    try:
        while True:
            success, frame = camera.read_frame()

            if not success:
                if not camera.is_connected:
                    logger.warning("Camera disconnected. Attempting reconnection...")
                    if camera.reconnect():
                        logger.info("Reconnected successfully!")
                        continue
                    else:
                        logger.error("Reconnection failed. Retrying...")
                        continue
                continue

            frame_count += 1

            # ── Face Recognition (every Nth frame based on FRAME_SKIP) ──
            if frame_count % recognition_interval == 0 or not last_results_with_status:
                recognition_results = face_engine.recognize_faces(
                    frame,
                    resize_factor=config.FACE_RESIZE_FACTOR,
                )
                current_results_with_status = []

                for result in recognition_results:
                    if result.is_recognized and result.student_id != "Unknown":
                        # Process attendance only for recognized students
                        status = attendance_manager.mark_present(
                            student_id=result.student_id,
                            student_name=result.student_name,
                        )
                    else:
                        # Unknown faces NEVER trigger attendance
                        status = AttendanceStatus.SKIPPED_UNKNOWN

                    current_results_with_status.append((result, status))

                last_results_with_status = current_results_with_status

            # ── FPS Calculation ───────────────────────────────
            elapsed = time.time() - fps_start_time
            if elapsed >= fps_update_interval:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            # ── Draw Overlay ──────────────────────────────────
            display_frame = draw_recognition_overlay(
                frame,
                last_results_with_status,
                fps,
                num_registered,
                attendance_manager.get_today_count(),
            )

            # ── Display ──────────────────────────────────────
            cv2.imshow(window_name, display_frame)

            # ── Key Handling ──────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                logger.info("Exit requested by user (Q pressed).")
                break

    except KeyboardInterrupt:
        logger.info("Exit requested by user (Ctrl+C).")

    finally:
        logger.info("Shutting down...")
        camera.release()
        db_manager.disconnect()
        cv2.destroyAllWindows()
        logger.info("Cleanup complete. Goodbye!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
