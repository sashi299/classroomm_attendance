"""
tests/__init__.py - Test package initializer.
Safely injects face_recognition_mock into sys.modules ONLY during unit test execution if face_recognition package is not installed.
"""

import sys
import os

try:
    import face_recognition
except ImportError:
    mock_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "face_recognition_mock"))
    if os.path.isdir(mock_dir) and mock_dir not in sys.path:
        sys.path.insert(0, mock_dir)
        try:
            import face_recognition_mock as mock_pkg
            sys.modules["face_recognition"] = mock_pkg
        except Exception:
            pass
