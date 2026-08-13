import unittest
import io
from datetime import datetime, date, time, timedelta
import os
import sys

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from database import DatabaseManager
from config import Config
from app import app, initialize_components

class TestSemesterReports(unittest.TestCase):
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

        # 1. Setup a test semester
        cls.db.add_semester("2026-27", "II", "I", "2026-08-01", "2026-08-31", True)

        # 2. Seed timetable for this semester
        # CSD_B_TIMETABLE_DATA in database.py already has 2026-27, II, I
        cls.db.seed_csd_b_timetable()

        # 3. Add some attendance
        # Inside range
        cls.db.insert_attendance("SEM_TEST_01", "Sem Student", date(2026, 8, 10), time(9, 30, 0),
                                department="CSD", section="B", period_number=1, hourly_period="09:15-10:20")
        # Outside range (should be ignored by filtered report)
        cls.db.insert_attendance("SEM_TEST_01", "Sem Student", date(2026, 7, 31), time(9, 30, 0),
                                department="CSD", section="B", period_number=1, hourly_period="09:15-10:20")

        initialize_components()

    def setUp(self):
        self.client = app.test_client()
        # Real login
        self.client.post("/login", data={"username": "admin", "password": "admin@2026"}, follow_redirects=True)

    def test_01_semester_date_clipping(self):
        """Report dates are clipped to semester boundaries."""
        # Request July to Sept, but semester is August only.
        res = self.client.get("/api/reports/attendance?start_date=2026-07-01&end_date=2026-09-30&academic_year=2026-27&year_level=II&semester=I")
        data = res.get_json()
        self.assertEqual(data["summary"]["start_date"], "2026-08-01")
        self.assertEqual(data["summary"]["end_date"], "2026-08-31")

    def test_02_sunday_exclusion(self):
        """Sundays are excluded from the report matrix."""
        # 2026-08-09 is a Sunday.
        res = self.client.get("/api/reports/attendance?start_date=2026-08-01&end_date=2026-08-31")
        data = res.get_json()
        dates = [r["attendance_date"] for r in data["all_records"]]
        self.assertNotIn("2026-08-09", dates)

    def test_03_time_range_filtering(self):
        """Report filters by from_time and to_time."""
        # Must specify department for admin user or it defaults to ALL
        res = self.client.get("/api/reports/attendance?dept=CSD&start_date=2026-08-10&end_date=2026-08-10&from_time=10:00&to_time=12:00")
        data = res.get_json()
        self.assertTrue(data["success"])

        # We expect P2 (10:20) and P3 (11:10) to be in the matrix
        periods = [r["period_number"] for r in data["all_records"]]
        self.assertIn(2, periods)
        self.assertIn(3, periods)
        self.assertNotIn(1, periods) # P1 is 09:15-10:20, starts before 10:00
        self.assertNotIn(5, periods) # P5 starts at 13:40

    def test_04_pdf_export_availability(self):
        """PDF export endpoint returns 200 and pdf mimetype."""
        res = self.client.get("/api/reports/attendance/export/pdf?start_date=2026-08-10&end_date=2026-08-10")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "application/pdf")

    def test_05_semester_api(self):
        """GET /api/semesters returns the configured semesters."""
        res = self.client.get("/api/semesters")
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertTrue(any(s["academic_year"] == "2026-27" for s in data["semesters"]))

if __name__ == "__main__":
    unittest.main()
