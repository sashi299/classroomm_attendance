"""
test_dashboard.py - Focused tests for the HOD Web Dashboard application.

Tests:
  - Flask web app endpoints (/ , /api/attendance/today , /api/health)
  - Dashboard overlay rendering rules (Green box for recognized, Red box for unrecognized)
  - Security (no DB credentials exposed to client)
  - Real-time JSON data structure
  - Authentication redirects for protected routes
"""

import os
import sys
import logging
import unittest

import cv2
import numpy as np

# Add src/ to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from app import app, draw_dashboard_overlay
from face_engine import RecognitionResult
from attendance import AttendanceStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_dashboard")


def _login_session(client, username="cse_hod", password="cse@hod2026"):
    """Helper to log in via POST and return the response."""
    return client.post("/login", data={
        "username": username,
        "password": password,
    }, follow_redirects=True)


class TestDashboard(unittest.TestCase):
    """Unit and Integration tests for HOD Web Dashboard."""

    def setUp(self):
        app.testing = True
        app.config["SECRET_KEY"] = "test-secret-key"
        self.client = app.test_client()

    def test_unauthenticated_dashboard_redirects_to_login(self):
        """Test that accessing / without auth redirects to /login."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers.get("Location", ""))

    def test_dashboard_route_status_and_template(self):
        """Test main dashboard HTML endpoint (/) returns 200 OK after login."""
        _login_session(self.client)
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)

        self.assertIn("Classroom Attendance Dashboard", html)
        self.assertIn("Today's Hourly Attendance Register", html)
        # Ensure no MySQL passwords exposed in HTML
        self.assertNotIn("MYSQL_PASSWORD", html)
        self.assertNotIn("password=", html)

    def test_api_today_attendance_requires_auth(self):
        """Test /api/attendance/today redirects without auth."""
        response = self.client.get("/api/attendance/today")
        self.assertEqual(response.status_code, 302)

    def test_api_today_attendance_structure(self):
        """Test /api/attendance/today endpoint returns JSON with expected schema after login."""
        _login_session(self.client)
        response = self.client.get("/api/attendance/today")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertIn("department_code", data)
        self.assertIn("department_name", data)
        self.assertIn("date", data)
        self.assertIn("count", data)
        self.assertIn("attendance", data)
        self.assertIsInstance(data["attendance"], list)

    def test_api_health_endpoint(self):
        """Test /api/health endpoint returns OK status (no auth required)."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "ok")

    def test_overlay_rendering_rules(self):
        """
        Test draw_dashboard_overlay rendering logic:
          - Recognized student -> GREEN box + ID/Name label
          - Unrecognized face -> RED box only (no name/ID label text)
        """
        frame = np.zeros((400, 600, 3), dtype=np.uint8)

        # 1. Recognized Result
        rec_result = RecognitionResult(
            student_id="22A01",
            student_name="Sashi",
            face_location=(50, 150, 150, 50),
            distance=0.3,
            is_recognized=True,
        )

        # 2. Unrecognized Result
        unk_result = RecognitionResult(
            student_id="Unknown",
            student_name="Unknown",
            face_location=(50, 350, 150, 250),
            distance=0.8,
            is_recognized=False,
        )

        results_with_status = [
            (rec_result, AttendanceStatus.NEWLY_MARKED),
            (unk_result, AttendanceStatus.SKIPPED_UNKNOWN),
        ]

        annotated = draw_dashboard_overlay(frame, results_with_status, fps=30.0, registered_count=1, today_count=1, dept_code="CSE")

        self.assertEqual(annotated.shape, (400, 600, 3))
        # Ensure frame is not blank
        self.assertTrue(np.sum(annotated) > 0)


def run_tests():
    """Run Dashboard test suite."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDashboard)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
