"""
camera_manager.py - Dynamic Department-to-camera mapping and lifecycle management.

Provides:
  - Per-department camera source configuration.
  - Dynamic camera creation, updates, and caching.
  - Graceful handling of unconfigured/offline cameras.
  - Clean shutdown of all camera resources.
"""

import logging
from typing import Optional, Dict, List, Tuple, Any

import cv2
import numpy as np

from camera import CameraStream

logger = logging.getLogger(__name__)

OFFLINE_FRAME_WIDTH = 640
OFFLINE_FRAME_HEIGHT = 480


def generate_offline_frame(
    dept_code: str = "",
    width: int = OFFLINE_FRAME_WIDTH,
    height: int = OFFLINE_FRAME_HEIGHT,
) -> np.ndarray:
    """Generate a 'Camera Offline' placeholder frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (30, 30, 40)

    font = cv2.FONT_HERSHEY_SIMPLEX

    main_text = "Camera Offline"
    main_scale = 1.2
    main_thickness = 2
    main_size, _ = cv2.getTextSize(main_text, font, main_scale, main_thickness)
    main_x = (width - main_size[0]) // 2
    main_y = (height // 2) - 10
    cv2.putText(
        frame, main_text,
        (main_x, main_y),
        font, main_scale, (100, 100, 200), main_thickness, cv2.LINE_AA,
    )

    if dept_code:
        sub_text = f"Department: {dept_code} — No camera connected"
    else:
        sub_text = "No camera configured"
    sub_scale = 0.5
    sub_size, _ = cv2.getTextSize(sub_text, font, sub_scale, 1)
    sub_x = (width - sub_size[0]) // 2
    sub_y = main_y + 40
    cv2.putText(
        frame, sub_text,
        (sub_x, sub_y),
        font, sub_scale, (120, 120, 140), 1, cv2.LINE_AA,
    )

    return frame


class CameraManager:
    """
    Manages per-department camera streams with dynamic updating and caching.
    Supports multi-camera per section classrooms.
    """

    def __init__(self, db_manager=None, department_cameras: Optional[Dict[str, List[str]]] = None):
        if isinstance(db_manager, dict):
            self.db = None
            self._legacy_config = db_manager
        else:
            self.db = db_manager
            self._legacy_config = dict(department_cameras) if department_cameras else {}

        # Cache: (dept_code, section, classroom, cam_name) -> CameraStream instance
        self._cameras: Dict[Tuple[str, str, str, str], CameraStream] = {}

        logger.info("CameraManager initialized.")

    def _get_camera_config(self, dept_code: str, section: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch camera configurations from database or legacy fallback."""
        if self.db:
            try:
                return self.db.get_cameras(department=dept_code, section=section, active_only=True)
            except: pass

        # Legacy fallback logic
        dept = dept_code.upper()
        if dept in self._legacy_config:
            return [{
                "id": 0, "name": "Default", "department": dept, "section": "B",
                "classroom": "Main", "source": s, "is_active": True
            } for s in self._legacy_config[dept]]
        return []

    def set_department_camera(self, dept_code: str, camera_source: str):
        """Update or set camera source for a department dynamically (Legacy)."""
        code = (dept_code or "").strip().upper()
        self._legacy_config[code] = [camera_source.strip()] if camera_source else []
        # Clear cache for this dept
        for key in list(self._cameras.keys()):
            if key[0] == code:
                try: self._cameras[key].release()
                except: pass
                del self._cameras[key]

    def get_camera(self, dept_code: str, section: str = "B", classroom: str = "Main", cam_name: str = "Default", **kwargs) -> Optional[CameraStream]:
        """Get (or lazily create and connect) camera. Supports legacy cam_index."""
        dept = dept_code.upper()
        sec = section.upper()

        # Legacy cam_index handling
        if "cam_index" in kwargs:
            idx = kwargs["cam_index"]
            if not self.db:
                sources = self._legacy_config.get(dept, [])
                if 0 <= idx < len(sources):
                    cam_name = str(idx)
                    sec = "B"
                    classroom = "Main"
                else:
                    return None # Out of bounds legacy index

        cache_key = (dept, sec, classroom, cam_name)

        if cache_key in self._cameras:
            cam = self._cameras[cache_key]
            if cam and cam._is_connected:
                return cam
            else:
                logger.info("Retrying connection for cached camera %s-%s [%s]...", dept, sec, cam_name)
                cam.connect()
                if cam._is_connected:
                    return cam
                try:
                    cam.release()
                except Exception:
                    pass
                del self._cameras[cache_key]

        # Handle Laptop Webcam selection for demo
        if cam_name in ["Laptop Webcam", "Webcam", "Laptop Camera", "0"]:
            cfg = {"source": "0"}
        else:
            configs = self._get_camera_config(dept, sec)
            cfg = None
            if configs:
                if cam_name == "Default":
                    cfg = configs[0]
                else:
                    cfg = next((c for c in configs if c["name"] == cam_name), None)

        if not cfg and not self.db:
            # If no DB, try legacy index-based access in legacy_config
            sources = self._legacy_config.get(dept, [])
            try:
                idx = int(cam_name) if cam_name.isdigit() else -1
                if 0 <= idx < len(sources):
                    cfg = {"source": sources[idx]}
            except Exception: pass

        if not cfg:
            return None

        source = cfg["source"]
        logger.info("Creating camera for %s-%s [%s]: source=%s", dept, sec, cam_name, source)
        cam = CameraStream(source=source)
        cam.connect()
        self._cameras[cache_key] = cam

        return self._cameras[cache_key]

    def is_camera_available(self, dept_code: str, section: str = "B", classroom: str = "Main", cam_name: str = "Default", **kwargs) -> bool:
        """Check if camera exists and is connected (cached)."""
        dept = dept_code.upper()
        sec = section.upper()
        if "cam_index" in kwargs:
            idx = kwargs["cam_index"]
            cam_name = str(idx)

        cache_key = (dept, sec, classroom, cam_name)
        cam = self._cameras.get(cache_key)
        if cam is None: return False
        return cam.is_connected

    def get_available_cameras(self, dept_code: str, section: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return list of active cameras for UI selection."""
        return self._get_camera_config(dept_code, section)

    def release_all(self):
        """Release all cached camera streams."""
        logger.info("Releasing all camera streams...")
        for key, cam in list(self._cameras.items()):
            try:
                cam.release()
                logger.info("  Released camera %s", key)
            except Exception as e:
                logger.warning("  Error releasing camera %s: %s", key, e)
        self._cameras.clear()
        logger.info("All camera streams released.")

    def get_camera_status(self) -> Dict[str, dict]:
        """Return camera status for all departments (Legacy support)."""
        status = {}
        depts = []
        if self.db:
            try:
                rows = self.db.get_departments(enabled_only=True)
                depts = [r["code"] for r in rows]
            except: depts = ["CSD"]
        else:
            depts = list(self._legacy_config.keys())

        for dept in depts:
            configs = self._get_camera_config(dept)
            sources = [c["source"] for c in configs]
            connected_list = []
            for c in configs:
                is_conn = False
                for k, cam in self._cameras.items():
                    if k[0] == dept and cam.is_connected:
                        is_conn = True
                        break
                connected_list.append(is_conn)

            status[dept] = {
                "configured": len(sources) > 0,
                "sources": sources,
                "connected": connected_list,
            }
        return status

    def get_all_departments(self) -> List[str]:
        """Return all configured department codes."""
        if self.db:
            try:
                return [d["code"] for d in self.db.get_departments(enabled_only=True)]
            except: return ["CSD"]
        return list(self._legacy_config.keys())

    def get_camera_sources(self, dept_code: str) -> List[str]:
        """Return configured camera sources for a department."""
        configs = self._get_camera_config(dept_code)
        return [c["source"] for c in configs]

    def __del__(self):
        try:
            self.release_all()
        except Exception:
            pass
