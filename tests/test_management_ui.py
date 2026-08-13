import unittest
import os
import sys
from datetime import date, datetime

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from database import DatabaseManager
from config import Config
from app import app, initialize_components

class TestManagementUI(unittest.TestCase):
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
        # Real login as admin
        self.client.post("/login", data={"username": "admin", "password": "admin@2026"}, follow_redirects=True)

    def test_01_faculty_crud(self):
        """Admin can add and retrieve faculty."""
        fac_id = "FAC_UI_01"
        data = {
            "faculty_id": fac_id,
            "name": "UI Test Faculty",
            "department": "CSD",
            "phone": "1234567890",
            "email": "fac.ui@example.com",
            "is_active": True
        }
        res = self.client.post("/api/faculty", json=data)
        self.assertEqual(res.status_code, 201)

        # Retrieve
        res = self.client.get("/api/faculty?department=CSD")
        found = any(f["faculty_id"] == fac_id for f in res.get_json()["faculty"])
        self.assertTrue(found)

    def test_02_holiday_crud(self):
        """Admin can add and retrieve holidays."""
        h_date = "2030-12-25"
        data = {
            "holiday_date": h_date,
            "description": "Christmas 2030",
            "is_active": True
        }
        res = self.client.post("/api/holidays", json=data)
        self.assertEqual(res.status_code, 200)

        # Check is_holiday logic
        self.assertTrue(self.db.is_holiday(date(2030, 12, 25)))

        # Delete
        self.client.delete(f"/api/holidays/{h_date}")
        self.assertFalse(self.db.is_holiday(date(2030, 12, 25)))

    def test_03_timetable_dynamic_mapping(self):
        """Timetable entry uses faculty mapping correctly."""
        # 1. Ensure faculty exists
        fac_id = "FAC_TT_01"
        self.db.add_faculty(fac_id, "TT Faculty", "CSD", phone="9998887770")

        # 2. Add timetable entry referencing this faculty
        tt_data = {
            "academic_year": "2026-27",
            "year_level": "II",
            "semester": "I",
            "department": "CSD",
            "section": "B",
            "day_of_week": "MON",
            "period_number": 9, # High period num to avoid conflict
            "start_time": "17:00:00",
            "end_time": "18:00:00",
            "subject": "Dynamic UI Test",
            "faculty_id": fac_id,
            "class_type": "THEORY"
        }
        res = self.client.post("/api/timetable/entries", json=tt_data)
        self.assertEqual(res.status_code, 200)

        # 3. Check current slot mapping
        # Monday Jan 7 2030 is a Monday. 17:30 is during the 17:00-18:00 period.
        now = datetime(2030, 1, 7, 17, 30, 0)
        slot = self.db.get_current_timetable_slot("CSD", "B", now=now)
        self.assertEqual(slot["status"], "ACTIVE")
        self.assertEqual(slot["faculty_name"], "TT Faculty")
        self.assertEqual(slot["faculty_contact"], "9998887770")

    def test_04_hod_department_isolation(self):
        """HOD can only see faculty for their own department."""
        # Switch to HOD login
        with self.client.session_transaction() as sess:
            sess['user'] = {'username': 'csd_hod', 'role': 'hod', 'department_code': 'CSD'}

        # Try to get EEE faculty
        res = self.client.get("/api/faculty?department=EEE")
        data = res.get_json()
        # Even if they requested EEE, the API should return CSD (or force it)
        for f in data["faculty"]:
            self.assertEqual(f["department"], "CSD")

if __name__ == "__main__":
    unittest.main()
