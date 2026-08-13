import unittest
import os
import sys
import io
import shutil
from datetime import date

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import app
from config import Config

class TestStudentEnrollment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.app.testing = True
        app.initialize_components()
        cls.client = app.app.test_client()

        # Test paths
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.valid_photo_path = os.path.join(base_path, "students", "CSD", "25a51a4470_Sashi", "sashi_angle2.jpg")
        cls.double_face_path = os.path.join(base_path, "students", "CSD", "99A03_Double Face Test", "double_face.jpg")

    def setUp(self):
        # Login as admin
        self.client.post("/login", data={"username": "admin", "password": "admin@2026"}, follow_redirects=True)
        # Clean up test student
        self.test_sid = "T999"
        self.client.post("/api/students/delete", json={"department": "CSD", "student_id": self.test_sid})

    def test_01_enrollment_workflow(self):
        """Test full student enrollment workflow with metadata and photo."""
        if not os.path.isfile(self.valid_photo_path):
            self.skipTest(f"Test photo not found at {self.valid_photo_path}")

        with open(self.valid_photo_path, "rb") as img:
            data = {
                "department": "CSD",
                "student_id": self.test_sid,
                "student_name": "Test Student",
                "year_level": "3",
                "section": "A",
                "academic_year": "2026-2027",
                "semester": "1",
                "photo": (io.BytesIO(img.read()), "test.jpg")
            }
            res = self.client.post("/api/students/add", data=data, content_type='multipart/form-data')
            self.assertEqual(res.status_code, 200)
            self.assertTrue(res.get_json()["success"])

        # Verify DB metadata
        students = app.db_manager.get_students(department="CSD")
        test_student = next((s for s in students if s["student_id"] == self.test_sid), None)
        self.assertIsNotNone(test_student)
        self.assertEqual(test_student["year_level"], "3")
        self.assertEqual(test_student["section"], "A")

        # Verify FaceEngine reload
        details = app.face_engine_manager.get_student_details("CSD", db_manager=app.db_manager)
        test_details = next((s for s in details if s["student_id"] == self.test_sid), None)
        self.assertIsNotNone(test_details)
        self.assertEqual(test_details["photo_count"], 1)

    def test_02_multi_photo_enrollment(self):
        """Test adding multiple photos to the same student."""
        if not os.path.isfile(self.valid_photo_path):
            self.skipTest("Test photo not found")

        # 1. Add first photo
        with open(self.valid_photo_path, "rb") as img:
            self.client.post("/api/students/add", data={
                "department": "CSD", "student_id": self.test_sid, "student_name": "Test Student",
                "photo": (io.BytesIO(img.read()), "p1.jpg")
            }, content_type='multipart/form-data')

        # 2. Add second photo
        with open(self.valid_photo_path, "rb") as img:
            res = self.client.post("/api/students/add", data={
                "department": "CSD", "student_id": self.test_sid, "student_name": "Test Student",
                "photo": (io.BytesIO(img.read() + b"\0"), "p2.jpg")
            }, content_type='multipart/form-data')

        details = app.face_engine_manager.get_student_details("CSD", db_manager=app.db_manager)
        test_details = next((s for s in details if s["student_id"] == self.test_sid), None)
        self.assertIsNotNone(test_details)
        self.assertGreaterEqual(test_details["photo_count"], 1)

    def test_03_invalid_photo_rejection(self):
        """Test rejection of photos with multiple faces or no faces."""
        if not os.path.isfile(self.double_face_path):
            self.skipTest("Double face photo not found")

        # Multi-face rejection
        with open(self.double_face_path, "rb") as img:
            res = self.client.post("/api/students/add", data={
                "department": "CSD", "student_id": self.test_sid, "student_name": "Double Face",
                "photo": (io.BytesIO(img.read()), "double.jpg")
            }, content_type='multipart/form-data')
            self.assertEqual(res.status_code, 400)
            self.assertIn("Multiple faces", res.get_json()["error"])

    def test_04_hod_isolation(self):
        """HOD cannot enroll students in another department."""
        if not os.path.isfile(self.valid_photo_path): self.skipTest("Photo not found")

        # Login as CSE HOD
        self.client.get("/logout")
        self.client.post("/login", data={"username": "cse_hod", "password": "cse@hod2026"})

        with open(self.valid_photo_path, "rb") as img:
            # Try to add to EEE (cse_hod is CSD)
            res = self.client.post("/api/students/add", data={
                "department": "EEE", "student_id": "EEE_ST", "student_name": "EEE Boy",
                "photo": (io.BytesIO(img.read()), "eee.jpg")
            }, content_type='multipart/form-data')
            self.assertEqual(res.status_code, 403)

if __name__ == "__main__":
    unittest.main()
