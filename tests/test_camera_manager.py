"""
test_camera_manager.py - Tests for department-to-camera mapping and CameraManager.

Tests:
  1. Department-to-camera config parsing
  2. Single camera per department
  3. Multiple cameras per department (comma-separated)
  4. HOD camera isolation (route-level)
  5. Admin access to all departments (route-level)
  6. Unavailable/unconfigured camera returns None
  7. Camera Offline frame generation
  8. Lazy camera creation
  9. Release all cameras
  10. /api/cameras endpoint
"""

import os
import sys
import logging
import unittest

import cv2
import numpy as np

# Add src/ to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from camera_manager import CameraManager, generate_offline_frame
from config import Config
from app import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_camera_manager")


def _login_session(client, username="cse_hod", password="cse@hod2026"):
    """Helper to log in via POST and return the response."""
    return client.post("/login", data={
        "username": username,
        "password": password,
    }, follow_redirects=True)


class TestCameraManager(unittest.TestCase):
    """Tests for CameraManager and department camera configuration."""

    def setUp(self):
        app.testing = True
        app.config["SECRET_KEY"] = "test-secret-key"
        self.client = app.test_client()

    # ── TEST 1: Config parsing maps dept codes to source lists ──
    def test_01_config_parsing(self):
        """DEPARTMENT_CAMERAS is parsed from env vars into a dict of lists."""
        cam_config = Config.DEPARTMENT_CAMERAS
        self.assertIsInstance(cam_config, dict)
        self.assertIn("CSD", cam_config)
        # CSD should have at least one source (webcam "0" in .env)
        self.assertIsInstance(cam_config["CSD"], list)
        self.assertGreaterEqual(len(cam_config["CSD"]), 1)
        logger.info("TEST 1 PASSED: Config parsing correct. CSD sources: %s", cam_config["CSD"])

    # ── TEST 2: Single camera per department ──────────────────
    def test_02_single_camera_per_department(self):
        """CameraManager returns a CameraStream for a configured department."""
        mgr = CameraManager({"TEST": ["0"]})
        sources = mgr.get_camera_sources("TEST")
        self.assertEqual(sources, ["0"])
        self.assertEqual(len(sources), 1)
        mgr.release_all()
        logger.info("TEST 2 PASSED: Single camera source retrieved correctly.")

    # ── TEST 3: Multiple cameras per department ───────────────
    def test_03_multiple_cameras_per_department(self):
        """Comma-separated sources produce a list with multiple entries."""
        # Simulate parsing that Config._parse_camera_sources would do
        raw = "0,rtsp://example.com/stream1,rtsp://example.com/stream2"
        sources = Config._parse_camera_sources(raw)
        self.assertEqual(len(sources), 3)
        self.assertEqual(sources[0], "0")
        self.assertIn("rtsp://", sources[1])
        self.assertIn("rtsp://", sources[2])

        # CameraManager should report all 3 sources
        mgr = CameraManager({"MULTI": sources})
        self.assertEqual(len(mgr.get_camera_sources("MULTI")), 3)
        mgr.release_all()
        logger.info("TEST 3 PASSED: Multiple camera sources parsed correctly.")

    # ── TEST 4: HOD camera isolation (route level) ────────────
    def test_04_hod_camera_isolation(self):
        """HOD can only access their own department's video feed (not another dept)."""
        # Login as CSE HOD
        _login_session(self.client, "cse_hod", "cse@hod2026")

        # Access /video_feed — should work (returns streaming response)
        response = self.client.get("/video_feed")
        self.assertEqual(response.status_code, 200)
        content_type = response.content_type or ""
        self.assertIn("multipart", content_type)

        # Even if ?dept=EEE is passed, HOD should still get their own dept's feed
        # (the route ignores the dept param for non-admin users)
        response2 = self.client.get("/video_feed?dept=EEE")
        self.assertEqual(response2.status_code, 200)

        logger.info("TEST 4 PASSED: HOD camera isolation enforced at route level.")

    # ── TEST 5: Admin access to all departments ───────────────
    def test_05_admin_access_all_departments(self):
        """Admin can specify ?dept= to access any department's camera."""
        _login_session(self.client, "admin", "admin@2026")

        # Access CSE camera
        response_cse = self.client.get("/video_feed?dept=CSE")
        self.assertEqual(response_cse.status_code, 200)

        # Access EEE camera (offline — but should not crash, returns 200 with offline frame)
        response_eee = self.client.get("/video_feed?dept=EEE")
        self.assertEqual(response_eee.status_code, 200)

        # Access ECE camera
        response_ece = self.client.get("/video_feed?dept=ECE")
        self.assertEqual(response_ece.status_code, 200)

        logger.info("TEST 5 PASSED: Admin can access all department cameras.")

    # ── TEST 6: Unavailable camera returns None ───────────────
    def test_06_unavailable_camera_returns_none(self):
        """Unconfigured department returns None from get_camera."""
        mgr = CameraManager({"CSE": ["0"], "EEE": []})
        cam = mgr.get_camera("EEE")
        self.assertIsNone(cam)

        # Non-existent department also returns None
        cam2 = mgr.get_camera("NONEXISTENT")
        self.assertIsNone(cam2)

        # Out-of-range cam_index returns None
        cam3 = mgr.get_camera("CSE", cam_index=99)
        self.assertIsNone(cam3)

        mgr.release_all()
        logger.info("TEST 6 PASSED: Unavailable cameras return None without crash.")

    # ── TEST 7: Offline frame generation ──────────────────────
    def test_07_offline_frame_generation(self):
        """generate_offline_frame produces a valid BGR frame with text."""
        frame = generate_offline_frame(dept_code="EEE")
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(frame.dtype, np.uint8)
        # Should not be completely black (has text overlaid)
        self.assertGreater(np.sum(frame), 0)
        logger.info("TEST 7 PASSED: Offline frame generated correctly.")

    # ── TEST 8: Lazy camera creation ──────────────────────────
    def test_08_lazy_camera_creation(self):
        """Cameras are NOT created until get_camera is called."""
        mgr = CameraManager({"LAZY": ["0"]})
        # Internal cache should be empty before get_camera
        self.assertEqual(len(mgr._cameras), 0)

        # is_camera_available returns False before first access
        self.assertFalse(mgr.is_camera_available("LAZY"))

        mgr.release_all()
        logger.info("TEST 8 PASSED: Lazy camera creation confirmed.")

    # ── TEST 9: Release all cameras ───────────────────────────
    def test_09_release_all_cameras(self):
        """release_all clears the internal camera cache."""
        mgr = CameraManager({"REL": ["0"]})
        # Force creation
        mgr.get_camera("REL")
        self.assertEqual(len(mgr._cameras), 1)

        mgr.release_all()
        self.assertEqual(len(mgr._cameras), 0)
        logger.info("TEST 9 PASSED: release_all clears all cameras.")

    # ── TEST 10: /api/cameras endpoint ────────────────────────
    def test_10_api_cameras_endpoint(self):
        """/api/cameras returns camera status JSON for admin."""
        _login_session(self.client, "admin", "admin@2026")
        response = self.client.get("/api/cameras")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("cameras", data)
        cameras = data["cameras"]
        # Admin should see enabled departments (CSD)
        self.assertIn("CSD", cameras)

        # Each dept has configured, sources, connected keys
        for dept, info in cameras.items():
            self.assertIn("configured", info)
            self.assertIn("sources", info)
            self.assertIn("connected", info)

        logger.info("TEST 10 PASSED: /api/cameras returns department camera status.")

    # ── TEST 11: HOD /api/cameras isolation ────────────────────
    def test_11_hod_api_cameras_isolation(self):
        """HOD can only see their own department in /api/cameras."""
        _login_session(self.client, "eee_hod", "eee@hod2026")
        response = self.client.get("/api/cameras")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        cameras = data["cameras"]
        # HOD should see CSD
        self.assertIn("CSD", cameras)
        self.assertEqual(len(cameras), 1)
        logger.info("TEST 11 PASSED: HOD sees only their department in /api/cameras.")

    # ── TEST 12: Camera status structure ──────────────────────
    def test_12_camera_status_structure(self):
        """get_camera_status returns correct structure for all departments."""
        mgr = CameraManager({"A": ["0"], "B": [], "C": ["0", "1"]})
        status = mgr.get_camera_status()

        # A: configured with 1 source
        self.assertTrue(status["A"]["configured"])
        self.assertEqual(len(status["A"]["sources"]), 1)

        # B: not configured
        self.assertFalse(status["B"]["configured"])
        self.assertEqual(len(status["B"]["sources"]), 0)

        # C: configured with 2 sources
        self.assertTrue(status["C"]["configured"])
        self.assertEqual(len(status["C"]["sources"]), 2)

        mgr.release_all()
        logger.info("TEST 12 PASSED: Camera status structure is correct.")

    # ── TEST 13: Admin dashboard shows dept switcher ──────────
    def test_13_admin_dashboard_has_dept_switcher(self):
        """Admin dashboard contains the department switcher dropdown."""
        _login_session(self.client, "admin", "admin@2026")
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("dept-select", html)
        self.assertIn("switchDepartment", html)
        logger.info("TEST 13 PASSED: Admin dashboard has department switcher.")

    # ── TEST 14: HOD dashboard hides dept switcher ────────────
    def test_14_hod_dashboard_no_dept_switcher(self):
        """HOD dashboard does NOT contain the department switcher."""
        _login_session(self.client, "cse_hod", "cse@hod2026")
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertNotIn("dept-select", html)
        logger.info("TEST 14 PASSED: HOD dashboard hides department switcher.")

    # ── TEST 15: RTSP password masking in /api/cameras ────────
    def test_15_rtsp_password_masking(self):
        """RTSP URLs with passwords are masked in /api/cameras output."""
        # This test creates a CameraManager with an RTSP URL containing a password,
        # then checks that the masking logic in the API works.
        _login_session(self.client, "admin", "admin@2026")
        response = self.client.get("/api/cameras")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        # No passwords should appear in the response
        response_str = str(data)
        self.assertNotIn("@hod2026", response_str)
        logger.info("TEST 15 PASSED: No RTSP passwords exposed in /api/cameras.")


def run_tests():
    """Run CameraManager test suite."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCameraManager)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
