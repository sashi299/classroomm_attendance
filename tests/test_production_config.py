"""
test_production_config.py - Unit and Integration tests for Production Configuration & Cloud Readiness.

Tests:
  1. Required environment variables loading.
  2. Missing configuration handling / defaults.
  3. Secret masking (passwords, RTSP credentials, secret key).
  4. Camera configuration parsing.
  5. Database configuration validation.
  6. Unauthorized API access blocked on /api/system/config-status.
  7. Admin vs HOD visibility on /api/system/config-status.
"""

import os
import sys
import unittest
import logging

# Add src/ to Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    import face_recognition
except ImportError:
    mock_dir = os.path.join(src_dir, "face_recognition_mock")
    if os.path.isdir(mock_dir) and mock_dir not in sys.path:
        sys.path.insert(0, mock_dir)
    import face_recognition_mock as mock_pkg
    sys.modules["face_recognition"] = mock_pkg

from config import Config
from app import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_production_config")


def _login(client, username="cse_hod", password="cse@hod2026"):
    """Helper to log in via POST."""
    return client.post("/login", data={
        "username": username,
        "password": password,
    }, follow_redirects=True)


class TestProductionConfig(unittest.TestCase):
    """Test suite for Config production validation and API endpoint."""

    def setUp(self):
        app.testing = True
        app.config["SECRET_KEY"] = "test-secret-key"
        self.client = app.test_client()

    # ── TEST 1: Required Environment Variables ───────────────
    def test_01_required_env_variables_loaded(self):
        """Configuration properties load properly with defaults or env values."""
        self.assertIsNotNone(Config.MYSQL_HOST)
        self.assertIsNotNone(Config.MYSQL_PORT)
        self.assertIsNotNone(Config.MYSQL_USER)
        self.assertIsNotNone(Config.MYSQL_DATABASE)
        self.assertIsNotNone(Config.SECRET_KEY)
        self.assertIsInstance(Config.FRAME_SKIP, int)
        self.assertIsInstance(Config.FACE_RESIZE_FACTOR, float)
        logger.info("TEST 1 PASSED: Environment variables loaded properly.")

    # ── TEST 2: Missing Configuration Handling ───────────────
    def test_02_missing_config_defaults(self):
        """Missing or empty camera sources parse into empty lists safely."""
        empty_sources = Config._parse_camera_sources("")
        self.assertEqual(empty_sources, [])

        whitespace_sources = Config._parse_camera_sources("   ,   ")
        self.assertEqual(whitespace_sources, [])
        logger.info("TEST 2 PASSED: Missing camera configuration handled safely.")

    # ── TEST 3: Secret Masking ───────────────────────────────
    def test_03_secret_masking(self):
        """Passswords and RTSP credentials are masked correctly."""
        masked_pass = Config.mask_secret("supersecret123")
        self.assertEqual(masked_pass, "***")
        self.assertNotIn("supersecret123", masked_pass)

        rtsp_url = "rtsp://admin:secretpass@192.168.1.100:554/live"
        masked_rtsp = Config.mask_secret(rtsp_url)
        self.assertEqual(masked_rtsp, "rtsp://admin:***@192.168.1.100:554/live")
        self.assertNotIn("secretpass", masked_rtsp)

        empty_mask = Config.mask_secret("")
        self.assertEqual(empty_mask, "[NOT SET]")
        logger.info("TEST 3 PASSED: Secret masking verified for raw passwords and RTSP URLs.")

    # ── TEST 4: Camera Configuration Parsing ──────────────────
    def test_04_camera_config_parsing(self):
        """Comma-separated camera sources are parsed correctly."""
        sources = Config._parse_camera_sources("0, 1, rtsp://cam1")
        self.assertEqual(len(sources), 3)
        self.assertEqual(sources[0], "0")
        self.assertEqual(sources[1], "1")
        self.assertEqual(sources[2], "rtsp://cam1")
        logger.info("TEST 4 PASSED: Camera sources parsed into lists.")

    # ── TEST 5: Database Configuration Validation ────────────
    def test_05_database_config_validation(self):
        """validate_production_config returns summary with masked DB credentials."""
        is_valid, warnings, summary = Config.validate_production_config()
        self.assertIn("database_host", summary)
        self.assertIn("database_password", summary)
        self.assertEqual(summary["database_password"], Config.mask_secret(Config.MYSQL_PASSWORD))
        self.assertNotIn("secretpass", str(summary))
        logger.info("TEST 5 PASSED: Database configuration validation verified.")

    # ── TEST 6: Unauthorized API Access ─────────────────────
    def test_06_unauthorized_config_status_access(self):
        """Unauthenticated GET /api/system/config-status is redirected to /login (302)."""
        response = self.client.get("/api/system/config-status")
        self.assertEqual(response.status_code, 302)
        logger.info("TEST 6 PASSED: Unauthenticated config-status access blocked.")

    # ── TEST 7: Admin vs HOD Visibility ──────────────────────
    def test_07_admin_vs_hod_visibility(self):
        """Admin receives full non-sensitive config summary; HOD receives safe operational data."""
        # 1. HOD User
        _login(self.client, "cse_hod", "cse@hod2026")
        res_hod = self.client.get("/api/system/config-status")
        self.assertEqual(res_hod.status_code, 200)
        data_hod = res_hod.get_json()
        self.assertEqual(data_hod["role"], "hod")
        self.assertNotIn("config", data_hod)
        self.assertIn("operational", data_hod)

        # 2. Admin User
        self.client.get("/logout")
        _login(self.client, "admin", "admin@2026")
        res_admin = self.client.get("/api/system/config-status")
        self.assertEqual(res_admin.status_code, 200)
        data_admin = res_admin.get_json()
        self.assertEqual(data_admin["role"], "admin")
        self.assertIn("config", data_admin)
        self.assertIn("production_ready", data_admin)

        # Confirm NO passwords or raw RTSP credentials anywhere in response
        raw_text = res_admin.get_data(as_text=True)
        if Config.MYSQL_PASSWORD and Config.MYSQL_PASSWORD != Config.MYSQL_USER:
            self.assertNotIn(Config.MYSQL_PASSWORD, raw_text)
        self.assertNotIn("secret123", raw_text)
        logger.info("TEST 7 PASSED: Admin vs HOD config-status visibility rules enforced.")


def run_tests():
    """Run Production Config test suite."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestProductionConfig)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
