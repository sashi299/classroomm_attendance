import unittest
import os
import sys
from datetime import date, datetime

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from database import DatabaseManager
from config import Config
from app import app, initialize_components

class TestMultiCamera(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = DatabaseManager(
            host=Config.MYSQL_HOST,
            port=Config.MYSQL_PORT,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DATABASE,
        )
        if not cls.db.connect():
            raise unittest.SkipTest("Database unavailable")
        initialize_components()

    def setUp(self):
        self.client = app.test_client()
        # Clear cameras
        with self.db._lock:
            cursor = self.db._connection.cursor()
            cursor.execute("DELETE FROM cameras")
            self.db._connection.commit()
            cursor.close()
        # Login as admin
        self.client.post("/login", data={"username": "admin", "password": "admin@2026"}, follow_redirects=True)

    def test_01_camera_registry_crud(self):
        """Admin can add and retrieve cameras via registry API."""
        cam_name = "Registry Test Cam"
        data = {
            "name": cam_name,
            "department": "CSD",
            "section": "B",
            "classroom": "Room 101",
            "source": "0",
            "is_active": True
        }
        res = self.client.post("/api/cameras/registry", json=data)
        self.assertEqual(res.status_code, 201)

        # Retrieve
        res = self.client.get("/api/cameras/registry?department=CSD&section=B")
        data = res.get_json()
        found = any(c["name"] == cam_name for c in data["cameras"])
        self.assertTrue(found)

    def test_02_hod_registry_isolation(self):
        """HOD can only see cameras for their own department in registry."""
        # 1. Admin adds EEE camera
        self.client.post("/api/cameras/registry", json={"name": "EEE Cam", "department": "EEE", "section": "A", "source": "3"})
        self.client.post("/api/cameras/registry", json={"name": "CSD Cam", "department": "CSD", "section": "B", "source": "0"})

        # 2. Mock login as CSD HOD
        self.client.get("/logout")
        with self.client.session_transaction() as sess:
            sess['logged_in'] = True
            sess['username'] = 'csd_hod'
            sess['role'] = 'hod'
            sess['department_code'] = 'CSD'

        # 3. Request EEE cameras
        res = self.client.get("/api/cameras/registry?department=EEE")
        data = res.get_json()
        # Should only return CSD cameras
        for c in data["cameras"]:
            self.assertEqual(c["department"], "CSD")
        self.assertFalse(any(c["name"] == "EEE Cam" for c in data["cameras"]))

    def test_03_legacy_status_api(self):
        """Legacy /api/cameras still returns status dict for compatibility."""
        self.client.post("/api/cameras/registry", json={"name": "Stat Cam", "department": "CSD", "section": "B", "source": "0"})

        res = self.client.get("/api/cameras")
        data = res.get_json()
        self.assertIn("cameras", data)
        self.assertIn("CSD", data["cameras"])
        self.assertTrue(data["cameras"]["CSD"]["configured"])

    def test_04_duplicate_attendance_protection(self):
        """Multiple cameras marking same student in same period is handled by DB unique constraint."""
        student_id = "MULTI_CAM_01"
        att_date = date.today()
        period = "09:15-10:20"

        # Clear existing
        with self.db._lock:
            cursor = self.db._connection.cursor()
            cursor.execute("DELETE FROM attendance WHERE student_id=%s AND attendance_date=%s AND hourly_period=%s",
                           (student_id, att_date, period))
            self.db._connection.commit()
            cursor.close()

        # 1. First camera marks student
        success1 = self.db.insert_attendance(
            student_id=student_id, student_name="Multi Cam Student",
            attendance_date=att_date, attendance_time=datetime.now().time(),
            department="CSD", section="B", period_number=1, hourly_period=period
        )
        self.assertTrue(success1)

        # 2. Second camera marks same student in same period
        success2 = self.db.insert_attendance(
            student_id=student_id, student_name="Multi Cam Student",
            attendance_date=att_date, attendance_time=datetime.now().time(),
            department="CSD", section="B", period_number=1, hourly_period=period
        )
        self.assertFalse(success2)

if __name__ == "__main__":
    unittest.main()
