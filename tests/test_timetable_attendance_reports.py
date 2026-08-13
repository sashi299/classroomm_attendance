"""
test_timetable_attendance_reports.py - Integration tests for timetable-aware HOD reports.
"""

import unittest
import os
import sys
import json
from datetime import date, datetime, timedelta

# Add src/ to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from app import app
from database import DatabaseManager
from config import Config

def _login(client, username="cse_hod", password="cse@hod2026"):
    return client.post("/login", data={"username": username, "password": password}, follow_redirects=True)

class TestTimetableAttendanceReports(unittest.TestCase):
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
            raise unittest.SkipTest("Database not available")

        # Ensure CSD-B timetable is seeded
        cls.db.seed_csd_b_timetable()

        # Insert a test attendance record for today P1 (if today is MON-SAT)
        cls.today = date.today()
        # Find a weekday for testing if today is Sunday
        cls.test_date = cls.today
        if cls.test_date.weekday() == 6: # Sunday
            cls.test_date -= timedelta(days=1)

        cls.db.insert_attendance(
            student_id="REPORT_TEST_01",
            student_name="Report Test Student",
            attendance_date=cls.test_date,
            attendance_time=datetime.now().time(),
            status="Present",
            department="CSD",
            section="B",
            period_number=1,
            subject="MFCS",
            class_type="THEORY"
        )

    def setUp(self):
        app.testing = True
        app.config["SECRET_KEY"] = "test-secret-key"
        self.client = app.test_client()

    def test_daily_report_timetable_aware(self):
        _login(self.client, "csd_hod", "csd@hod2026")
        date_str = self.test_date.isoformat()
        response = self.client.get(f"/api/reports/attendance/daily?date={date_str}&dept=CSD&section=B")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertTrue(data["success"])
        self.assertIn("summary", data)
        self.assertGreater(data["summary"]["total_periods"], 0)

        # Verify Matrix Structure and Status Values
        for rec in data["records"]:
            self.assertIn(rec["status"], ["PRESENT", "ABSENT"])
            self.assertIn("period_number", rec)
            self.assertIn("subject", rec)

    def test_department_isolation_hod(self):
        # HOD for CSD should NOT be able to access report for EEE (returns 403)
        _login(self.client, "csd_hod", "csd@hod2026")
        response = self.client.get("/api/reports/attendance/daily?dept=EEE")
        self.assertEqual(response.status_code, 403)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Unauthorized", data["error"])

    def test_csv_export_headers(self):
        _login(self.client, "admin", "admin@2026")
        response = self.client.get("/api/reports/attendance/export/csv?dept=CSD")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/csv")

        content = response.data.decode("utf-8")
        headers = content.splitlines()[0].split(",")
        expected = ["Date", "Period", "Time Slot", "Subject", "Class Type", "Student ID", "Student Name", "Section", "Status", "Attendance Time"]
        for exp in expected:
            self.assertIn(exp, headers)

    def test_excel_export(self):
        # Check it returns valid excel or expected error message
        _login(self.client, "admin", "admin@2026")
        response = self.client.get("/api/reports/attendance/export/excel?dept=CSD")
        self.assertEqual(response.status_code, 200)

        # Check if openpyxl is available in the environment
        try:
            import openpyxl
            has_openpyxl = True
        except ImportError:
            has_openpyxl = False

        if has_openpyxl:
            # Excel signature
            self.assertTrue(response.data.startswith(b"PK\x03\x04"))
        else:
            self.assertIn(b"Excel export unavailable", response.data)

if __name__ == "__main__":
    unittest.main()
