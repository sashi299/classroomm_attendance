"""
test_student_management.py - Unit and Integration tests for Student Management.

Tests:
  1. Admin can access student directory API (/api/students).
  2. HOD cannot add students (403 Forbidden).
  3. HOD cannot delete students (403 Forbidden).
  4. Valid single-face image registration by Admin.
  5. Invalid image file rejection (.txt / corrupted file).
  6. Multiple-face image rejection (2 faces in single photo).
  7. Duplicate student ID rejection in the same department.
  8. Correct department directory placement (students/EEE).
  9. FaceEngine cache reload after registration.
 10. Student deletion by Admin.
 11. Attendance records preserved after student photo deletion.
"""

import os
import io
import sys
import shutil
import tempfile
import logging
import unittest

import cv2
import numpy as np

import glob
from datetime import date, datetime

# Add src/ to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from face_engine_manager import FaceEngineManager
from database import DatabaseManager
from attendance import AttendanceManager
from app import app


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_student_management")


def _login_session(client, username="cse_hod", password="cse@hod2026"):
    """Helper to log in via POST and return the response."""
    return client.post("/login", data={
        "username": username,
        "password": password,
    }, follow_redirects=True)


class TestStudentManagement(unittest.TestCase):
    """Test suite for Admin-only Student Management."""

    def setUp(self):
        app.testing = True
        app.config["SECRET_KEY"] = "test-secret-key"
        self.client = app.test_client()

        # Path to existing valid sample face image
        self.sample_face_path = os.path.join("students", "CSE", "25a51a4470_Sashi Duplicate", "25a51a4470_sashi.jpeg.jpeg")
        if not os.path.isfile(self.sample_face_path):
            self.sample_face_path = os.path.join("students", "CSE", "25a51a4470_sashi.jpeg.jpeg")

    def tearDown(self):
        """Clean up test photos created in EEE or ECE directories."""
        for d in ["students/EEE", "students/ECE"]:
            if os.path.isdir(d):
                for f in glob.glob(os.path.join(d, "*.*")):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
        from app import face_engine_manager
        if face_engine_manager:
            face_engine_manager.reload_engine("EEE")
            face_engine_manager.reload_engine("ECE")

    # ── TEST 1: Admin can access student list API ────────────
    def test_01_admin_can_access_student_list(self):
        """Admin can GET /api/students and see registered students."""
        _login_session(self.client, "admin", "admin@2026")
        response = self.client.get("/api/students?dept=CSD")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("students", data)
        self.assertGreaterEqual(data["count"], 0)
        logger.info("TEST 1 PASSED: Admin can access student directory API.")

    # ── TEST 2: HOD can add students for their own dept ───────
    def test_02_hod_can_add_students_own_dept(self):
        """HOD user can register students for their own department."""
        _login_session(self.client, "csd_hod", "csd@hod2026")
        if not os.path.isfile(self.sample_face_path): self.skipTest("No sample face")

        with open(self.sample_face_path, "rb") as img_file:
            response = self.client.post("/api/students/add", data={
                "department": "CSD",
                "student_id": "HOD_TEST_01",
                "student_name": "HOD Test Student",
                "photo": (img_file, "hod_test.jpg"),
            }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)

        # Cleanup
        self.client.post("/api/students/delete", json={"department": "CSD", "student_id": "HOD_TEST_01"})
        logger.info("TEST 2 PASSED: HOD can add students for their own department.")

    # ── TEST 3: HOD cannot add students for other dept ────────
    def test_03_hod_cannot_add_students_other_dept(self):
        """HOD user is blocked from registering students for another department."""
        _login_session(self.client, "csd_hod", "csd@hod2026")
        response = self.client.post("/api/students/add", data={
            "department": "EEE",
            "student_id": "HOD_FAIL",
            "student_name": "Should Fail",
        })
        self.assertEqual(response.status_code, 403)
        logger.info("TEST 3 PASSED: HOD blocked from other department enrollment.")

    # ── TEST 4: Valid single-face image registration ─────────
    def test_04_valid_single_face_registration(self):
        """Admin can register a valid single-face photo."""
        if not os.path.isfile(self.sample_face_path):
            self.skipTest("Sample face image file not found.")

        _login_session(self.client, "admin", "admin@2026")
        with open(self.sample_face_path, "rb") as img_file:
            response = self.client.post(
                "/api/students/add",
                data={
                    "department": "CSD",
                    "student_id": "99A01",
                    "student_name": "Test CSD Student",
                    "photo": (img_file, "valid_student.jpg"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])

        # Clean up created test student photo
        expected_path = os.path.join("students", "CSD", "99A01_Test CSD Student", "valid_student.jpg")
        alt_path = os.path.join("students", "CSD", "99A01_Test CSD Student.jpg")
        self.assertTrue(os.path.isfile(expected_path) or os.path.isfile(alt_path))

        # Cleanup
        if os.path.isfile(expected_path):
            os.remove(expected_path)
        if os.path.isfile(alt_path):
            os.remove(alt_path)
        from app import face_engine_manager
        if face_engine_manager:
            face_engine_manager.reload_engine("CSD")

        logger.info("TEST 4 PASSED: Valid single-face photo registered successfully.")

    # ── TEST 5: Invalid image file rejection ─────────────────
    def test_05_invalid_image_file_rejection(self):
        """Uploading a text or corrupt file returns 400 Bad Request."""
        _login_session(self.client, "admin", "admin@2026")
        response = self.client.post(
            "/api/students/add",
            data={
                "department": "CSD",
                "student_id": "99A02",
                "student_name": "Invalid File Test",
                "photo": (io.BytesIO(b"this is not an image file content"), "test.txt"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        logger.info("TEST 5 PASSED: Invalid image file correctly rejected.")

    # ── TEST 6: Multiple-face image rejection ────────────────
    def test_06_multiple_face_image_rejection(self):
        """Uploading an image containing multiple faces returns 400 Bad Request."""
        if not os.path.isfile(self.sample_face_path):
            self.skipTest("Sample face image file not found.")

        # Read single face image and stitch it side-by-side to create a 2-face image
        img = cv2.imread(self.sample_face_path)
        if img is None:
            self.skipTest("Could not read sample face image.")

        two_faces_img = np.hstack((img, img))
        is_success, buffer = cv2.imencode(".jpg", two_faces_img)
        self.assertTrue(is_success)

        _login_session(self.client, "admin", "admin@2026")
        response = self.client.post(
            "/api/students/add",
            data={
                "department": "CSD",
                "student_id": "99A03",
                "student_name": "Double Face Test",
                "photo": (io.BytesIO(buffer.tobytes()), "double_face.jpg"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Multiple faces", data["error"])
        logger.info("TEST 6 PASSED: Multiple-face image correctly rejected.")

    # ── TEST 7: Multi-photo registration for existing student ID ────────
    def test_07_multi_photo_registration_for_existing_student(self):
        """Registering an additional photo for an existing student ID returns 200 OK."""
        if not os.path.isfile(self.sample_face_path):
            self.skipTest("Sample face image file not found.")

        _login_session(self.client, "admin", "admin@2026")
        from app import face_engine_manager
        if face_engine_manager:
            face_engine_manager.delete_student("CSD", "MULTI_TEST_01")

        with open(self.sample_face_path, "rb") as img_file:
            res1 = self.client.post(
                "/api/students/add",
                data={
                    "department": "CSD",
                    "student_id": "MULTI_TEST_01",
                    "student_name": "Multi Student",
                    "photo": (img_file, "photo1.jpg"),
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(res1.status_code, 200)

        with open(self.sample_face_path, "rb") as img_file:
            res2 = self.client.post(
                "/api/students/add",
                data={
                    "department": "CSD",
                    "student_id": "MULTI_TEST_01",
                    "student_name": "Multi Student",
                    "photo": (img_file, "photo2.jpg"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(res2.status_code, 200)
        data = res2.get_json()
        self.assertTrue(data["success"])

        if face_engine_manager:
            face_engine_manager.delete_student("CSD", "MULTI_TEST_01")
        logger.info("TEST 7 PASSED: Multi-photo registration for existing student ID succeeded.")

    # ── TEST 8: Correct department directory placement ───────
    def test_08_correct_department_directory_placement(self):
        """Registering a CSD student places the file strictly in students/CSD/."""
        if not os.path.isfile(self.sample_face_path):
            self.skipTest("Sample face image file not found.")

        _login_session(self.client, "admin", "admin@2026")
        with open(self.sample_face_path, "rb") as img_file:
            response = self.client.post(
                "/api/students/add",
                data={
                    "department": "CSD",
                    "student_id": "88E01",
                    "student_name": "CSD Student",
                    "photo": (img_file, "csd_student.jpg"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)

        expected_csd_file = os.path.join("students", "CSD", "88E01_CSD Student", "csd_student.jpg")
        alt_csd_file = os.path.join("students", "CSD", "88E01_CSD Student.jpg")
        self.assertTrue(os.path.isfile(expected_csd_file) or os.path.isfile(alt_csd_file))

        # Cleanup
        if os.path.isfile(expected_csd_file):
            os.remove(expected_csd_file)
        if os.path.isfile(alt_csd_file):
            os.remove(alt_csd_file)
        from app import face_engine_manager
        if face_engine_manager:
            face_engine_manager.reload_engine("CSD")

        logger.info("TEST 8 PASSED: File placed strictly in students/CSD/.")

    # ── TEST 9: FaceEngine cache reload after registration ────
    def test_09_face_engine_cache_reload(self):
        """After registering a student, FaceEngineManager immediately updates registered count."""
        if not os.path.isfile(self.sample_face_path):
            self.skipTest("Sample face image file not found.")

        from app import face_engine_manager, initialize_components
        initialize_components()

        initial_csd_count = face_engine_manager.get_engine("CSD").get_registered_count()

        _login_session(self.client, "admin", "admin@2026")
        with open(self.sample_face_path, "rb") as img_file:
            response = self.client.post(
                "/api/students/add",
                data={
                    "department": "CSD",
                    "student_id": "77E01",
                    "student_name": "Reload Test",
                    "photo": (img_file, "reload_student.jpg"),
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)

        # Active CSD engine should now have count incremented
        new_csd_count = face_engine_manager.get_engine("CSD").get_registered_count()
        self.assertEqual(new_csd_count, initial_csd_count + 1)

        # Cleanup
        target_file = os.path.join("students", "CSD", "77E01_Reload Test", "reload_student.jpg")
        alt_file = os.path.join("students", "CSD", "77E01_Reload Test.jpg")
        if os.path.isfile(target_file):
            os.remove(target_file)
        if os.path.isfile(alt_file):
            os.remove(alt_file)
        face_engine_manager.reload_engine("CSD")

        logger.info("TEST 9 PASSED: FaceEngine cache updated immediately after registration.")

    # ── TEST 10: Student deletion ────────────────────────────
    def test_10_student_deletion(self):
        """Admin can delete a student registration via POST /api/students/delete."""
        if not os.path.isfile(self.sample_face_path):
            self.skipTest("Sample face image file not found.")

        _login_session(self.client, "admin", "admin@2026")

        # 1. First add a test student to CSD
        with open(self.sample_face_path, "rb") as img_file:
            res_add = self.client.post(
                "/api/students/add",
                data={
                    "department": "CSD",
                    "student_id": "66E01",
                    "student_name": "Delete Me",
                    "photo": (img_file, "del_student.jpg"),
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(res_add.status_code, 200)

        target_path = os.path.join("students", "CSD", "66E01_Delete Me", "del_student.jpg")
        alt_path = os.path.join("students", "CSD", "66E01_Delete Me.jpg")
        self.assertTrue(os.path.isfile(target_path) or os.path.isfile(alt_path))

        # 2. Delete the student
        res_del = self.client.post(
            "/api/students/delete",
            json={"department": "CSD", "student_id": "66E01"},
        )
        self.assertEqual(res_del.status_code, 200)
        data_del = res_del.get_json()
        self.assertTrue(data_del["success"])

        # Verify photo file is removed from disk
        self.assertFalse(os.path.isfile(target_path))
        self.assertFalse(os.path.isfile(alt_path))

        logger.info("TEST 10 PASSED: Student deleted and photo removed from disk.")

    # ── TEST 11: Attendance records preserved after deletion ──
    def test_11_attendance_records_preserved_after_deletion(self):
        """Deleting a student registration does NOT delete their MySQL attendance history."""
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
        # Ensure no prior attendance for dummy student exists
        try:
            if db._connection and db._connection.is_connected():
                cur = db._connection.cursor()
                cur.execute(
                    "DELETE FROM attendance WHERE student_id=%s AND department=%s",
                    ("55A01", "CSD"),
                )
                db._connection.commit()
                cur.close()
        except Exception as e:
            logger.error("Pre-test cleanup failed: %s", e)

        # Use try/finally to guarantee cleanup of the dummy attendance record
        try:
            # Use a fixed weekday time to ensure timetable allows attendance (Mon 09:30 is P1)
            test_now = datetime(2026, 8, 10, 9, 30)
            test_date = test_now.date()
            # Mark attendance for a dummy student ID
            att_mgr.mark_present(student_id="55A01", student_name="Hist Preserved Student", dept_code="CSD", now=test_now)

            # Period label for P1 is "09:15-10:20"
            current_period = "09:15-10:20"
            # Verify attendance exists for the captured period
            self.assertTrue(db.check_attendance_exists("55A01", test_date, hourly_period=current_period, department="CSD"))

            # Perform student deletion for 55A01
            _login_session(self.client, "admin", "admin@2026")
            self.client.post(
                "/api/students/delete",
                json={"department": "CSD", "student_id": "55A01"},
            )

            # Attendance record in MySQL MUST still exist for the same period
            still_exists = db.check_attendance_exists("55A01", test_date, hourly_period=current_period, department="CSD")
            self.assertTrue(still_exists)
        finally:
            # Cleanup: remove the dummy attendance entry
            try:
                if db._connection and db._connection.is_connected():
                    cur = db._connection.cursor()
                    cur.execute(
                        "DELETE FROM attendance WHERE student_id=%s AND department=%s AND attendance_date=%s",
                        ("55A01", "CSD", test_date if 'test_date' in locals() else date.today()),
                    )
                    db._connection.commit()
                    cur.close()
            except Exception as e:
                logger.error("Failed to cleanup dummy attendance record: %s", e)
            # Ensure the DB connection is closed
            db.disconnect()
        logger.info("TEST 11 PASSED: MySQL attendance history preserved after student deletion.")

    # ── TEST 12: No duplicate student IDs in /api/students ─────
    def test_12_no_duplicate_student_ids_in_api_students(self):
        """Ensure /api/students returns each student_id only ONCE even with multi-photo encodings."""
        _login_session(self.client, "admin", "admin@2026")
        response = self.client.get("/api/students?dept=CSD")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        students = data.get("students", [])
        student_ids = [s["student_id"] for s in students]

        # Assert no duplicate student IDs in API response
        self.assertEqual(len(student_ids), len(set(student_ids)),
                         f"Duplicate student IDs detected in /api/students response: {student_ids}")

        # Specifically check '25a51a4470' if present
        sashi_records = [s for s in students if s["student_id"].lower() == "25a51a4470"]
        if sashi_records:
            self.assertEqual(len(sashi_records), 1, "Student ID '25a51a4470' must appear EXACTLY ONCE.")
            self.assertIn("photo_count", sashi_records[0])
            self.assertGreaterEqual(sashi_records[0]["photo_count"], 1)

        logger.info("TEST 12 PASSED: /api/students returns unique student IDs without duplicates.")


def run_tests():
    """Run Student Management test suite."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStudentManagement)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
