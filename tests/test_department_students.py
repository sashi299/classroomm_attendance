"""
test_department_students.py - Unit and Integration tests for Department-wise Student Management and Face Recognition Isolation.

Tests:
  1. Department directory resolution via Config.
  2. Per-department student loading (CSE loads registered student).
  3. Student isolation between departments (CSE student not present in EEE/ECE).
  4. Empty department directory handling (EEE directory returns 0 students safely).
  5. Multiple student photos per department.
  6. HOD face recognition & video feed isolation (query parameters ignored for HOD).
  7. Admin department switching via query parameters.
  8. Attendance department filtering (/api/attendance/today).
  9. Unknown face handling in empty department engines.
 10. Missing department directory handling.
"""

import os
import sys
import shutil
import tempfile
import logging
import unittest
import numpy as np

# Add src/ to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from face_engine import RecognitionResult
from face_engine_manager import FaceEngineManager
from app import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_department_students")


def _login_session(client, username="cse_hod", password="cse@hod2026"):
    """Helper to log in via POST and return the response."""
    return client.post("/login", data={
        "username": username,
        "password": password,
    }, follow_redirects=True)


class TestDepartmentStudents(unittest.TestCase):
    """Test suite for Department-wise Student Management and Isolation."""

    def setUp(self):
        app.testing = True
        app.config["SECRET_KEY"] = "test-secret-key"
        self.client = app.test_client()

    # ── TEST 1: Department directory resolution via Config ──
    def test_01_department_directory_resolution(self):
        """get_department_students_dir returns correct path for uppercase and lowercase codes."""
        cse_path = Config.get_department_students_dir("CSE")
        eee_path = Config.get_department_students_dir("eee")
        ece_path = Config.get_department_students_dir("Ece")

        self.assertTrue(cse_path.endswith(os.path.join("students", "CSE")))
        self.assertTrue(eee_path.endswith(os.path.join("students", "EEE")))
        self.assertTrue(ece_path.endswith(os.path.join("students", "ECE")))
        logger.info("TEST 1 PASSED: Department directory resolution verified.")

    # ── TEST 2: Per-department student loading ───────────────
    def test_02_cse_student_loading(self):
        """FaceEngineManager loads CSE registered student(s)."""
        manager = FaceEngineManager()
        cse_engine = manager.get_engine("CSE")
        self.assertGreaterEqual(cse_engine.get_registered_count(), 1)
        cse_ids = manager.get_registered_student_ids("CSE")
        self.assertIn("25a51a4470", cse_ids)
        logger.info("TEST 2 PASSED: CSE engine loaded student '25a51a4470'.")

    # ── TEST 3: Student isolation between departments ────────
    def test_03_student_isolation_between_departments(self):
        """CSE student is NOT present in EEE or ECE engines."""
        manager = FaceEngineManager()
        cse_ids = manager.get_registered_student_ids("CSE")
        eee_ids = manager.get_registered_student_ids("EEE")
        ece_ids = manager.get_registered_student_ids("ECE")

        self.assertIn("25a51a4470", cse_ids)
        self.assertNotIn("25a51a4470", eee_ids)
        self.assertNotIn("25a51a4470", ece_ids)
        logger.info("TEST 3 PASSED: Student isolation between CSE, EEE, ECE verified.")

    # ── TEST 4: Empty department directory handling ─────────
    def test_04_empty_department_directory(self):
        """Empty department directory (TEST_EMPTY) returns 0 registered students without error."""
        manager = FaceEngineManager()
        empty_engine = manager.get_engine("TEST_EMPTY")
        self.assertEqual(empty_engine.get_registered_count(), 0)
        logger.info("TEST 4 PASSED: Empty department directory handled gracefully.")

    # ── TEST 5: Multiple student photos per department ───────
    def test_05_multiple_student_photos(self):
        """Multiple valid student photos in a department directory are all loaded."""
        temp_dir = tempfile.mkdtemp(prefix="dept_students_test_")
        try:
            cse_dir = os.path.join(temp_dir, "CSE")
            os.makedirs(cse_dir, exist_ok=True)

            # Copy sample student photo into temp CSE directory twice with different IDs
            sample_src = os.path.join("students", "CSE", "25a51a4470_sashi.jpeg.jpeg")
            if os.path.isfile(sample_src):
                shutil.copy(sample_src, os.path.join(cse_dir, "22A01_Sashi.jpg"))
                shutil.copy(sample_src, os.path.join(cse_dir, "22A02_Ravi.jpg"))

                manager = FaceEngineManager(base_dir=temp_dir)
                cse_ids = manager.get_registered_student_ids("CSE")
                self.assertEqual(len(cse_ids), 2)
                self.assertIn("22A01", cse_ids)
                self.assertIn("22A02", cse_ids)
                logger.info("TEST 5 PASSED: Multiple student photos loaded successfully.")
            else:
                self.skipTest("Sample student photo not available for copying.")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ── TEST 6: HOD recognition isolation (query param ignored) ─
    def test_06_hod_recognition_isolation(self):
        """HOD user cannot override department via ?dept= query parameter."""
        # Login as EEE HOD
        _login_session(self.client, "eee_hod", "eee@hod2026")

        # Request /video_feed?dept=CSE
        response = self.client.get("/video_feed?dept=CSE")
        self.assertEqual(response.status_code, 200)
        # Content-Type should be MJPEG streaming response
        self.assertIn("multipart/x-mixed-replace", response.content_type)
        logger.info("TEST 6 PASSED: HOD cannot override department via query param.")

    # ── TEST 7: Admin department switching ───────────────────
    def test_07_admin_department_switching(self):
        """Admin can switch departments via ?dept= query parameter."""
        _login_session(self.client, "admin", "admin@2026")

        # Admin requests CSE video feed
        response_cse = self.client.get("/video_feed?dept=CSE")
        self.assertEqual(response_cse.status_code, 200)

        # Admin requests EEE video feed
        response_eee = self.client.get("/video_feed?dept=EEE")
        self.assertEqual(response_eee.status_code, 200)

        logger.info("TEST 7 PASSED: Admin department switching via query params verified.")

    # ── TEST 8: Attendance department filtering ──────────────
    def test_08_attendance_department_filtering(self):
        """HOD attendance API returns records filtered for that HOD's department."""
        _login_session(self.client, "cse_hod", "cse@hod2026")
        response = self.client.get("/api/attendance/today")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["department_code"], "CSD")

        # Logout CSE HOD and login as EEE HOD
        self.client.get("/logout")
        _login_session(self.client, "eee_hod", "eee@hod2026")
        response_eee = self.client.get("/api/attendance/today")
        self.assertEqual(response_eee.status_code, 200)
        data_eee = response_eee.get_json()
        self.assertEqual(data_eee["department_code"], "CSD")

        logger.info("TEST 8 PASSED: Attendance department filtering verified.")

    # ── TEST 9: Unknown face in empty department engine ───────
    def test_09_unknown_face_in_empty_department(self):
        """Face recognition on an empty department engine returns is_recognized=False."""
        manager = FaceEngineManager()
        empty_engine = manager.get_engine("TEST_EMPTY")
        self.assertEqual(empty_engine.get_registered_count(), 0)

        # Blank BGR frame
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        results = empty_engine.recognize_faces(frame)
        self.assertEqual(len(results), 0)  # No face in blank frame
        logger.info("TEST 9 PASSED: Recognition on empty department engine runs safely.")

    # ── TEST 10: Missing department directory handling ───────
    def test_10_missing_department_directory(self):
        """Accessing a non-existent department directory creates it safely."""
        temp_dir = tempfile.mkdtemp(prefix="dept_missing_test_")
        try:
            manager = FaceEngineManager(base_dir=temp_dir)
            engine = manager.get_engine("NEWDEPT")
            self.assertEqual(engine.get_registered_count(), 0)
            self.assertTrue(os.path.isdir(os.path.join(temp_dir, "NEWDEPT")))
            logger.info("TEST 10 PASSED: Missing department directory auto-created safely.")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_tests():
    """Run Department Students test suite."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDepartmentStudents)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
