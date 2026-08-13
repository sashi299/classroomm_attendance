"""
test_system_state.py - Unit and Integration tests for System State & Exam Mode.

Tests:
  1. Default state is attendance enabled (Exam Mode OFF).
  2. Admin can enable Exam Mode.
  3. Admin can disable Exam Mode.
  4. HOD cannot enable Exam Mode (403 Forbidden).
  5. HOD cannot disable Exam Mode (403 Forbidden).
  6. Unauthenticated access blocked.
  7. Exam Mode state is thread-safe.
  8. Camera remains available during Exam Mode.
  9. Recognition is skipped during Exam Mode.
 10. Attendance is not inserted during Exam Mode.
 11. Recognition resumes after disabling Exam Mode.
 12. Existing attendance records remain unchanged.
"""

import os
import sys
import logging
import threading
import unittest
from datetime import date, time, datetime, timedelta

# Add src/ to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from system_state import SystemStateManager, system_state_manager
from database import DatabaseManager
from attendance import AttendanceManager, AttendanceStatus
from app import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_system_state")


def _login(client, username="cse_hod", password="cse@hod2026"):
    """Helper to log in via POST."""
    return client.post("/login", data={
        "username": username,
        "password": password,
    }, follow_redirects=True)


class TestSystemState(unittest.TestCase):
    """Test suite for SystemStateManager and Exam Mode integration."""

    def setUp(self):
        app.testing = True
        app.config["SECRET_KEY"] = "test-secret-key"
        self.client = app.test_client()
        # Ensure Exam Mode is OFF before each test
        system_state_manager.disable_exam_mode()

    def tearDown(self):
        # Reset Exam Mode to OFF after each test
        system_state_manager.disable_exam_mode()

    # ── TEST 1: Default state is attendance enabled ────────────
    def test_01_default_state_attendance_enabled(self):
        """Default runtime state is Exam Mode OFF (Attendance Enabled)."""
        self.assertFalse(system_state_manager.is_exam_mode_enabled())
        self.assertTrue(system_state_manager.is_attendance_enabled())
        logger.info("TEST 1 PASSED: Default state is attendance enabled.")

    # ── TEST 2: Admin can enable Exam Mode ───────────────────
    def test_02_admin_can_enable_exam_mode(self):
        """Admin POST /api/system/exam-mode/enable succeeds."""
        _login(self.client, "admin", "admin@2026")
        response = self.client.post("/api/system/exam-mode/enable")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertTrue(data["exam_mode"])
        self.assertTrue(system_state_manager.is_exam_mode_enabled())
        logger.info("TEST 2 PASSED: Admin enabled Exam Mode.")

    # ── TEST 3: Admin can disable Exam Mode ──────────────────
    def test_03_admin_can_disable_exam_mode(self):
        """Admin POST /api/system/exam-mode/disable succeeds."""
        system_state_manager.enable_exam_mode()
        _login(self.client, "admin", "admin@2026")

        response = self.client.post("/api/system/exam-mode/disable")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertFalse(data["exam_mode"])
        self.assertFalse(system_state_manager.is_exam_mode_enabled())
        logger.info("TEST 3 PASSED: Admin disabled Exam Mode.")

    # ── TEST 4: HOD cannot enable Exam Mode ──────────────────
    def test_04_hod_cannot_enable_exam_mode(self):
        """HOD user POSTing to /api/system/exam-mode/enable gets 403 Forbidden."""
        _login(self.client, "cse_hod", "cse@hod2026")
        response = self.client.post("/api/system/exam-mode/enable")
        self.assertEqual(response.status_code, 403)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertFalse(system_state_manager.is_exam_mode_enabled())
        logger.info("TEST 4 PASSED: HOD blocked from enabling Exam Mode (403).")

    # ── TEST 5: HOD cannot disable Exam Mode ─────────────────
    def test_05_hod_cannot_disable_exam_mode(self):
        """HOD user POSTing to /api/system/exam-mode/disable gets 403 Forbidden."""
        system_state_manager.enable_exam_mode()
        _login(self.client, "eee_hod", "eee@hod2026")

        response = self.client.post("/api/system/exam-mode/disable")
        self.assertEqual(response.status_code, 403)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertTrue(system_state_manager.is_exam_mode_enabled())
        logger.info("TEST 5 PASSED: HOD blocked from disabling Exam Mode (403).")

    # ── TEST 6: Unauthenticated access blocked ───────────────
    def test_06_unauthenticated_access_blocked(self):
        """Unauthenticated requests to POST endpoints are redirected to /login (302)."""
        res_en = self.client.post("/api/system/exam-mode/enable")
        self.assertEqual(res_en.status_code, 302)

        res_dis = self.client.post("/api/system/exam-mode/disable")
        self.assertEqual(res_dis.status_code, 302)
        logger.info("TEST 6 PASSED: Unauthenticated access redirected to /login.")

    # ── TEST 7: Exam Mode state is thread-safe ───────────────
    def test_07_thread_safety(self):
        """Concurrent state updates across 20 threads operate safely."""
        errors = []

        def worker(toggle_id):
            try:
                if toggle_id % 2 == 0:
                    system_state_manager.enable_exam_mode()
                else:
                    system_state_manager.disable_exam_mode()
                _ = system_state_manager.is_exam_mode_enabled()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        logger.info("TEST 7 PASSED: Thread safety verified across 20 concurrent threads.")

    # ── TEST 8: Camera remains available during Exam Mode ────
    def test_08_camera_remains_available_during_exam_mode(self):
        """Video feed endpoint /video_feed is accessible when Exam Mode is active."""
        system_state_manager.enable_exam_mode()
        _login(self.client, "cse_hod", "cse@hod2026")

        response = self.client.get("/video_feed")
        self.assertEqual(response.status_code, 200)
        self.assertIn("multipart/x-mixed-replace", response.content_type)
        logger.info("TEST 8 PASSED: Camera stream remains available during Exam Mode.")

    # ── TEST 9: Recognition is skipped during Exam Mode ──────
    def test_09_recognition_skipped_during_exam_mode(self):
        """System status shows EXAM_PAUSED and attendance_enabled=False during Exam Mode."""
        system_state_manager.enable_exam_mode()
        _login(self.client, "admin", "admin@2026")

        response = self.client.get("/api/system/status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["exam_mode"])
        self.assertFalse(data["attendance_enabled"])
        self.assertEqual(data["status"], "EXAM_PAUSED")
        logger.info("TEST 9 PASSED: System status reflects face recognition pause.")

    # ── TEST 10: Attendance is not inserted during Exam Mode ──
    def test_10_attendance_not_inserted_during_exam_mode(self):
        """AttendanceManager.mark_present returns SKIPPED_EXAM_MODE when Exam Mode is active."""
        db = DatabaseManager(
            host=Config.MYSQL_HOST,
            port=Config.MYSQL_PORT,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DATABASE,
        )
        if not db.connect():
            self.skipTest("MySQL database not connected.")

        att_mgr = AttendanceManager(db_manager=db)

        # Enable Exam Mode
        system_state_manager.enable_exam_mode()

        status = att_mgr.mark_present(student_id="EXAM_STUDENT", student_name="Exam Student")
        self.assertEqual(status, AttendanceStatus.SKIPPED_EXAM_MODE)

        # Confirm not inserted into DB
        exists = db.check_attendance_exists("EXAM_STUDENT", date.today())
        self.assertFalse(exists)

        db.disconnect()
        logger.info("TEST 10 PASSED: Attendance marking skipped during Exam Mode.")

    # ── TEST 11: Recognition resumes after disabling ─────────
    def test_11_recognition_resumes_after_disabling(self):
        """After disabling Exam Mode, attendance marking resumes normal operation."""
        db = DatabaseManager(
            host=Config.MYSQL_HOST,
            port=Config.MYSQL_PORT,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DATABASE,
        )
        if not db.connect():
            self.skipTest("MySQL database not connected.")

        att_mgr = AttendanceManager(db_manager=db)

        # Enable then Disable Exam Mode
        system_state_manager.enable_exam_mode()
        system_state_manager.disable_exam_mode()

        # Use fixed time to ensure timetable allows attendance (Mon 09:30)
        test_now = datetime(2026, 8, 10, 9, 30)
        status = att_mgr.mark_present(student_id="RESUME_STUDENT", student_name="Resume Student", now=test_now)
        self.assertIn(status, [AttendanceStatus.NEWLY_MARKED, AttendanceStatus.ALREADY_PRESENT])

        db.disconnect()
        logger.info("TEST 11 PASSED: Attendance marking resumes automatically after disabling Exam Mode.")

    # ── TEST 12: Existing attendance records remain unchanged ──
    def test_12_existing_attendance_records_unchanged(self):
        """Enabling and disabling Exam Mode does not alter or delete existing attendance records."""
        db = DatabaseManager(
            host=Config.MYSQL_HOST,
            port=Config.MYSQL_PORT,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DATABASE,
        )
        if not db.connect():
            self.skipTest("MySQL database not connected.")

        test_date = date(2026, 8, 10)
        test_period = "09:15-10:20"

        # 1. Insert a test attendance record for CSD
        db.insert_attendance("PRESERVE_TEST", "Preserve Test", test_date, time(9, 30, 0), "Present", department="CSD", hourly_period=test_period)
        self.assertTrue(db.check_attendance_exists("PRESERVE_TEST", test_date, department="CSD", hourly_period=test_period))

        # 2. Toggle Exam Mode ON and OFF
        system_state_manager.enable_exam_mode()
        self.assertTrue(db.check_attendance_exists("PRESERVE_TEST", test_date, department="CSD", hourly_period=test_period))

        system_state_manager.disable_exam_mode()
        self.assertTrue(db.check_attendance_exists("PRESERVE_TEST", test_date, department="CSD", hourly_period=test_period))

        db.disconnect()
        logger.info("TEST 12 PASSED: Existing attendance records preserved across Exam Mode toggles.")


def run_tests():
    """Run System State test suite."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSystemState)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
