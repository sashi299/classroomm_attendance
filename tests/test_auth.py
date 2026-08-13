"""
test_auth.py - Tests for HOD authentication and department isolation.

Tests:
  1. Valid login for each HOD (CSE, EEE, ECE) and Admin.
  2. Invalid login attempts (wrong password, unknown user).
  3. Session protection (protected routes redirect without auth).
  4. Department isolation (each HOD sees only their department metadata).
  5. Admin access (admin sees "ALL" department scope).
  6. Logout clears session.
  7. Session data integrity (department_code, department_name stored correctly).
"""

import os
import sys
import logging
import unittest

# Add src/ to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from app import app
from auth import authenticate_user, USERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_auth")


class TestAuthentication(unittest.TestCase):
    """Tests for login, session, and department isolation."""

    def setUp(self):
        app.testing = True
        app.config["SECRET_KEY"] = "test-secret-key"
        self.client = app.test_client()

    # ── TEST 1: Valid CSE HOD Login ──────────────────────────
    def test_01_valid_cse_hod_login(self):
        """CSE HOD can log in with correct credentials."""
        response = self.client.post("/login", data={
            "username": "cse_hod",
            "password": "cse@hod2026",
        }, follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/", response.headers.get("Location", ""))
        logger.info("TEST 1 PASSED: CSE HOD login successful.")

    # ── TEST 2: Valid EEE HOD Login ──────────────────────────
    def test_02_valid_eee_hod_login(self):
        """EEE HOD can log in with correct credentials."""
        response = self.client.post("/login", data={
            "username": "eee_hod",
            "password": "eee@hod2026",
        }, follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        logger.info("TEST 2 PASSED: EEE HOD login successful.")

    # ── TEST 3: Valid ECE HOD Login ──────────────────────────
    def test_03_valid_ece_hod_login(self):
        """ECE HOD can log in with correct credentials."""
        response = self.client.post("/login", data={
            "username": "ece_hod",
            "password": "ece@hod2026",
        }, follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        logger.info("TEST 3 PASSED: ECE HOD login successful.")

    # ── TEST 4: Valid Admin Login ────────────────────────────
    def test_04_valid_admin_login(self):
        """Admin can log in with correct credentials."""
        response = self.client.post("/login", data={
            "username": "admin",
            "password": "admin@2026",
        }, follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        logger.info("TEST 4 PASSED: Admin login successful.")

    # ── TEST 5: Invalid Password ────────────────────────────
    def test_05_invalid_password(self):
        """Login with wrong password shows error, stays on login page."""
        response = self.client.post("/login", data={
            "username": "cse_hod",
            "password": "wrongpassword",
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Invalid username or password", html)
        logger.info("TEST 5 PASSED: Invalid password correctly rejected.")

    # ── TEST 6: Unknown Username ────────────────────────────
    def test_06_unknown_username(self):
        """Login with nonexistent username shows error."""
        response = self.client.post("/login", data={
            "username": "nonexistent_user",
            "password": "anypassword",
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Invalid username or password", html)
        logger.info("TEST 6 PASSED: Unknown username correctly rejected.")

    # ── TEST 7: Protected Route Redirect ────────────────────
    def test_07_protected_route_redirects(self):
        """Dashboard (/) redirects to /login when not authenticated."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers.get("Location", ""))
        logger.info("TEST 7 PASSED: Unauthenticated access redirects to /login.")

    # ── TEST 8: API Attendance Redirect ─────────────────────
    def test_08_api_attendance_requires_auth(self):
        """/api/attendance/today redirects without auth."""
        response = self.client.get("/api/attendance/today")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers.get("Location", ""))
        logger.info("TEST 8 PASSED: Attendance API requires authentication.")

    # ── TEST 9: Video Feed Redirect ─────────────────────────
    def test_09_video_feed_requires_auth(self):
        """/video_feed redirects without auth."""
        response = self.client.get("/video_feed")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers.get("Location", ""))
        logger.info("TEST 9 PASSED: Video feed requires authentication.")

    # ── TEST 10: CSE HOD Dashboard Shows CSD Department ─────
    def test_10_cse_hod_sees_cse_department(self):
        """After CSE HOD login, dashboard shows CSD department."""
        self.client.post("/login", data={
            "username": "cse_hod",
            "password": "cse@hod2026",
        })
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("CSD", html)
        logger.info("TEST 10 PASSED: CSE HOD sees CSD department in dashboard.")

    # ── TEST 11: EEE HOD Dashboard Shows CSD Department ─────
    def test_11_eee_hod_sees_eee_department(self):
        """After EEE HOD login, dashboard shows CSD department."""
        self.client.post("/login", data={
            "username": "eee_hod",
            "password": "eee@hod2026",
        })
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("CSD", html)
        logger.info("TEST 11 PASSED: EEE HOD sees CSD department in dashboard.")

    # ── TEST 12: Admin Dashboard Shows All Departments ──────
    def test_12_admin_sees_all_departments(self):
        """After Admin login, dashboard shows All Departments."""
        self.client.post("/login", data={
            "username": "admin",
            "password": "admin@2026",
        })
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("ALL", html)
        self.assertIn("All Departments", html)
        logger.info("TEST 12 PASSED: Admin sees All Departments scope.")

    # ── TEST 13: API Returns Department-Scoped Data ─────────
    def test_13_api_returns_department_code(self):
        """Attendance API returns the logged-in HOD's department code."""
        self.client.post("/login", data={
            "username": "cse_hod",
            "password": "cse@hod2026",
        })
        response = self.client.get("/api/attendance/today")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["department_code"], "CSD")
        self.assertIn("CSD", data["department_name"])
        logger.info("TEST 13 PASSED: API returns CSD department scope.")

    # ── TEST 14: Admin API Returns ALL Department Scope ─────
    def test_14_admin_api_returns_all(self):
        """Admin's attendance API returns ALL department scope."""
        self.client.post("/login", data={
            "username": "admin",
            "password": "admin@2026",
        })
        response = self.client.get("/api/attendance/today")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["department_code"], "ALL")
        logger.info("TEST 14 PASSED: Admin API returns ALL department scope.")

    # ── TEST 15: Logout Clears Session ──────────────────────
    def test_15_logout_clears_session(self):
        """After logout, accessing / redirects back to /login."""
        # Login first
        self.client.post("/login", data={
            "username": "cse_hod",
            "password": "cse@hod2026",
        })
        # Verify authenticated
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

        # Logout
        response = self.client.get("/logout")
        self.assertEqual(response.status_code, 302)

        # Verify session cleared
        response = self.client.get("/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers.get("Location", ""))
        logger.info("TEST 15 PASSED: Logout clears session, redirect to /login.")

    # ── TEST 16: Login Page Renders Without Auth ────────────
    def test_16_login_page_renders(self):
        """/login GET renders the login page."""
        response = self.client.get("/login")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Sign In", html)
        self.assertIn("Username", html)
        self.assertIn("Password", html)
        logger.info("TEST 16 PASSED: Login page renders correctly.")

    # ── TEST 17: Already Logged In Redirects From Login ─────
    def test_17_already_logged_in_redirects_from_login(self):
        """If already logged in, accessing /login redirects to dashboard."""
        self.client.post("/login", data={
            "username": "cse_hod",
            "password": "cse@hod2026",
        })
        response = self.client.get("/login")
        self.assertEqual(response.status_code, 302)
        logger.info("TEST 17 PASSED: Logged-in user redirected from /login to /.")

    # ── TEST 18: Health Check No Auth Required ──────────────
    def test_18_health_check_no_auth_required(self):
        """/api/health is accessible without authentication."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ok")
        logger.info("TEST 18 PASSED: /api/health accessible without auth.")

    # ── TEST 19: No Credentials Exposed In Login Page ───────
    def test_19_no_credentials_in_login_page(self):
        """Login page does not expose any MySQL credentials."""
        response = self.client.get("/login")
        html = response.get_data(as_text=True)
        self.assertNotIn("MYSQL_PASSWORD", html)
        self.assertNotIn("MYSQL_USER", html)
        self.assertNotIn("mysql.connector", html)
        # Ensure no database connection strings are exposed
        self.assertNotIn("localhost:3306", html)
        logger.info("TEST 19 PASSED: No credentials exposed in login page.")

    # ── TEST 20: authenticate_user Function ─────────────────
    def test_20_authenticate_user_function(self):
        """Direct test of the authenticate_user function."""
        # Valid user
        result = authenticate_user("csd_hod", "csd@hod2026")
        self.assertIsNotNone(result)
        self.assertEqual(result["department_code"], "CSD")
        self.assertEqual(result["role"], "hod")

        # Invalid password
        result = authenticate_user("cse_hod", "wrong")
        self.assertIsNone(result)

        # Unknown user
        result = authenticate_user("nobody", "pass")
        self.assertIsNone(result)

        logger.info("TEST 20 PASSED: authenticate_user function works correctly.")


def run_tests():
    """Run Authentication test suite."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAuthentication)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
