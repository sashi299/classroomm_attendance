"""
system_state.py - Centralized system settings and Exam Mode state manager.

Provides:
  - Thread-safe state access for Exam Mode (ON/OFF).
  - Defense-in-depth toggle preventing attendance insertion while Exam Mode is active.
  - Live system health status aggregator.
"""

import logging
import threading
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class SystemStateManager:
    """
    Thread-safe system state manager.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._exam_mode_enabled: bool = False

        logger.info("SystemStateManager initialized.")
        logger.info("  Exam Mode: DISABLED (normal attendance operating mode)")

    def is_exam_mode_enabled(self) -> bool:
        """Return True if Exam Mode is active."""
        with self._lock:
            return self._exam_mode_enabled

    def is_attendance_enabled(self) -> bool:
        """Return True if attendance marking is enabled (Exam Mode OFF)."""
        return not self.is_exam_mode_enabled()

    def enable_exam_mode(self) -> Tuple[bool, str]:
        """Enable Exam Mode."""
        with self._lock:
            self._exam_mode_enabled = True
            logger.warning("EXAM MODE ENABLED: Live face recognition & attendance marking are now PAUSED.")
            return True, "Exam Mode enabled successfully. Face recognition & attendance marking are paused."

    def disable_exam_mode(self) -> Tuple[bool, str]:
        """Disable Exam Mode."""
        with self._lock:
            self._exam_mode_enabled = False
            logger.info("EXAM MODE DISABLED: Live face recognition & attendance marking have RESUMED.")
            return True, "Exam Mode disabled successfully. Face recognition & attendance marking have resumed."

    def get_system_status(
        self,
        db_manager=None,
        camera_manager=None,
        face_engine_manager=None,
    ) -> Dict[str, Any]:
        """Return aggregated system health, Exam Mode status, and department stats."""
        with self._lock:
            exam_mode = self._exam_mode_enabled

        db_connected = db_manager.is_connected if db_manager else False

        camera_status = {}
        if camera_manager:
            camera_status = camera_manager.get_camera_status()

        registered_counts = {}
        if face_engine_manager:
            try:
                ids_map = face_engine_manager.get_all_registered_student_ids(db_manager=db_manager)
                registered_counts = {dept: len(ids) for dept, ids in ids_map.items()}
            except Exception as e:
                logger.warning("Error getting registered counts: %s", e)

        current_period = ""
        if db_manager:
            try:
                current_period = db_manager.get_current_hourly_period()
            except Exception:
                pass

        return {
            "status": "EXAM_PAUSED" if exam_mode else ("ONLINE" if db_connected else "DEGRADED"),
            "exam_mode": exam_mode,
            "attendance_enabled": not exam_mode,
            "db_connected": db_connected,
            "current_hourly_period": current_period,
            "camera_status": camera_status,
            "registered_students": registered_counts,
        }


# Singleton system state manager instance
system_state_manager = SystemStateManager()
