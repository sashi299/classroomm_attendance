"""
config.py - Application configuration loader and production validation.

Loads settings from environment variables and .env file using python-dotenv.
All sensitive credentials and tunable parameters are managed here with safe defaults
and production configuration validation routines.
"""

import os
import re
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Centralized configuration for the Classroom Attendance System."""

    # Environment Mode
    ENVIRONMENT = os.getenv("ENVIRONMENT", os.getenv("FLASK_ENV", "development")).lower()

    # RTSP Camera
    RTSP_URL = os.getenv("RTSP_URL", "0")

    # CORS Settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

    # Cloud / MySQL Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "")
    _db_host = os.getenv("DB_HOST", os.getenv("MYSQL_HOST", "localhost"))
    _db_port = int(os.getenv("DB_PORT", os.getenv("MYSQL_PORT", "3306")))
    _db_user = os.getenv("DB_USER", os.getenv("MYSQL_USER", "root"))
    _db_pass = os.getenv("DB_PASSWORD", os.getenv("MYSQL_PASSWORD", ""))
    _db_name = os.getenv("DB_NAME", os.getenv("MYSQL_DATABASE", "classroom_db"))

    if DATABASE_URL:
        from urllib.parse import urlparse
        try:
            p = urlparse(DATABASE_URL)
            if p.hostname: _db_host = p.hostname
            if p.port: _db_port = p.port
            if p.username: _db_user = p.username
            if p.password: _db_pass = p.password
            if p.path: _db_name = p.path.lstrip("/")
        except Exception:
            pass

    MYSQL_HOST = _db_host
    MYSQL_PORT = _db_port
    MYSQL_USER = _db_user
    MYSQL_PASSWORD = _db_pass
    MYSQL_DATABASE = _db_name

    # SMTP Configuration
    SMTP_HOST = os.getenv("SMTP_HOST", "")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USERNAME)

    # Face Recognition Parameters
    RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.55"))

    # Student Directory & Storage Configuration
    STUDENTS_BASE_DIR = os.getenv("STUDENTS_BASE_DIR", "students")
    KNOWN_STUDENTS_DIR = os.getenv("KNOWN_STUDENTS_DIR", "known_students")  # Backward compatibility
    STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower()

    # Server Port Configuration
    PORT = int(os.getenv("PORT", os.getenv("DASHBOARD_PORT", "5000")))
    DASHBOARD_PORT = PORT
    def get_department_students_dir(cls, dept_code: str) -> str:
        """Get the student photos directory path for a specific department."""
        code = (dept_code or "CSD").strip().upper()
        return os.path.join(cls.STUDENTS_BASE_DIR, code)

    # Default Department & Dashboard Settings
    DEPARTMENT_NAME = os.getenv("DEPARTMENT_NAME", "CSD")
    DEPARTMENT_CODE = os.getenv("DEPARTMENT_CODE", "CSD")
    DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "5000"))

    # Dynamic Department Helper Methods (CSD only for CCTV trial)
    DEFAULT_DEPARTMENT_CODES = ["CSD"]

    @classmethod
    def _parse_camera_sources(cls, env_val: str) -> list:
        """Parse comma-separated camera sources from environment string."""
        if not env_val or not env_val.strip():
            return []
        return [s.strip() for s in env_val.split(",") if s.strip()]

    @classmethod
    def get_active_department_codes(cls, db_manager=None) -> List[str]:
        """Dynamically fetch active department codes from database or config defaults."""
        if db_manager is not None:
            try:
                depts = db_manager.get_departments(enabled_only=True)
                if depts:
                    return [d["code"] for d in depts]
            except Exception:
                pass
        return list(cls.DEFAULT_DEPARTMENT_CODES)

    @classmethod
    def get_department_cameras_map(cls, db_manager=None) -> Dict[str, List[str]]:
        """Dynamically fetch department camera sources map from database or env overrides."""
        camera_map = {}
        if db_manager is not None:
            try:
                depts = db_manager.get_departments(enabled_only=True)
                for d in depts:
                    src = d.get("camera_source", "").strip()
                    camera_map[d["code"]] = [src] if src else ["0"]
                if camera_map:
                    return camera_map
            except Exception:
                pass

        return {
            "CSD": cls._parse_camera_sources(os.getenv("CAMERA_CSD", os.getenv("RTSP_URL", "0"))),
        }

    # Backward compatibility properties
    DEPARTMENT_CODES = DEFAULT_DEPARTMENT_CODES
    DEPARTMENT_CAMERAS = {
        "CSD": ["0"],
    }

    # Performance Optimization Settings
    FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))
    FACE_RESIZE_FACTOR = float(os.getenv("FACE_RESIZE_FACTOR", "1.0"))

    # Flask Secret Key
    DEFAULT_SECRET_KEY = "classroom-attendance-secret-key-change-in-production"
    SECRET_KEY = os.getenv("SECRET_KEY", DEFAULT_SECRET_KEY)

    @classmethod
    def mask_secret(cls, value: str) -> str:
        """
        Mask passwords or secret credentials safely.

        Examples:
          'mysecret' -> '***'
          'rtsp://admin:secret123@192.168.1.10:554/live' -> 'rtsp://admin:***@192.168.1.10:554/live'
          '' -> '[NOT SET]'
        """
        if not value:
            return "[NOT SET]"
        if "://" in value:
            return re.sub(r"://([^:@]+):([^@]+)@", r"://\1:***@", value)
        return "***"

    @classmethod
    def get_masked_camera_sources(cls, db_manager=None) -> Dict[str, List[str]]:
        """Return department camera sources with RTSP credentials masked."""
        cams = cls.get_department_cameras_map(db_manager=db_manager)
        masked_cams = {}
        for dept, sources in cams.items():
            masked_cams[dept] = [cls.mask_secret(src) for src in sources]
        return masked_cams

    @classmethod
    def validate_production_config(cls, db_manager=None) -> Tuple[bool, List[str], Dict]:
        """
        Validate production configuration and safety.
        """
        warnings = []
        is_valid = True

        if not cls.SECRET_KEY or cls.SECRET_KEY == cls.DEFAULT_SECRET_KEY:
            warnings.append("SECRET_KEY is using default development key. Change in production!")
            if os.getenv("FLASK_ENV") == "production" or os.getenv("ENVIRONMENT") == "production":
                is_valid = False

        if not cls.MYSQL_HOST:
            warnings.append("MYSQL_HOST is empty.")
            is_valid = False
        if not cls.MYSQL_DATABASE:
            warnings.append("MYSQL_DATABASE is empty.")
            is_valid = False
        if cls.MYSQL_PORT < 1 or cls.MYSQL_PORT > 65535:
            warnings.append(f"Invalid MYSQL_PORT: {cls.MYSQL_PORT}")
            is_valid = False

        cams = cls.get_department_cameras_map(db_manager=db_manager)
        configured_depts = [dept for dept, src in cams.items() if len(src) > 0]
        if len(configured_depts) == 0:
            warnings.append("No department cameras configured. Live feed will show offline frames.")

        if cls.FRAME_SKIP < 1:
            warnings.append(f"FRAME_SKIP ({cls.FRAME_SKIP}) should be >= 1.")
        if cls.FACE_RESIZE_FACTOR <= 0 or cls.FACE_RESIZE_FACTOR > 1.0:
            warnings.append(f"FACE_RESIZE_FACTOR ({cls.FACE_RESIZE_FACTOR}) should be between 0.1 and 1.0.")

        summary = {
            "secret_key_configured": cls.SECRET_KEY != cls.DEFAULT_SECRET_KEY,
            "database_host": cls.mask_secret(cls.MYSQL_HOST),
            "database_port": cls.MYSQL_PORT,
            "database_name": cls.MYSQL_DATABASE,
            "database_user": cls.mask_secret(cls.MYSQL_USER),
            "database_password": cls.mask_secret(cls.MYSQL_PASSWORD),
            "camera_sources": cls.get_masked_camera_sources(db_manager=db_manager),
            "frame_skip": cls.FRAME_SKIP,
            "face_resize_factor": cls.FACE_RESIZE_FACTOR,
            "storage_backend": cls.STORAGE_BACKEND,
            "department_code": cls.DEPARTMENT_CODE,
            "is_valid": is_valid,
            "warnings_count": len(warnings),
        }

        return is_valid, warnings, summary
