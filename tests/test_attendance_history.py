"""
test_attendance_history.py - Unit and Integration tests for Attendance History & CSV Export.

Tests:
  1. Today's attendance retrieval.
  2. Single-date filtering.
  3. Date-range filtering.
  4. Search by student ID.
  5. Search by student name.
  6. CSE department filtering.
  7. HOD department isolation.
  8. Admin ALL-department access.
  9. Unauthorized access protection.
 10. CSV generation and correct headers.
 11. HOD CSV department isolation.
 12. Admin CSV department filtering.
"""

import os
import io
import sys
import csv
import logging
import unittest
from datetime import date, timedelta, time, datetime

# Add src/ to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from database import DatabaseManager
from attendance import AttendanceManager
from app import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_attendance_history")


def _login(client, username="cse_hod", password="cse@hod2026"):
    """Helper to log in via POST."""
    return client.post("/login", data={
        "username": username,
        "password": password,
    }, follow_redirects=True)


class TestAttendanceHistory(unittest.TestCase):
    """Test suite for Attendance History and CSV Export endpoints."""

    @classmethod
    def setUpClass(cls):
        """Seed test database records across multiple dates and departments."""
        cls.db = DatabaseManager(
            host=Config.MYSQL_HOST,
            port=Config.MYSQL_PORT,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DATABASE,
        )
        logger.info("setUpClass: Connecting to DB...")
        if not cls.db.connect():
            raise unittest.SkipTest("MySQL database not connected.")
        logger.info("setUpClass: Connected successfully.")

        cls.today = date.today()
        cls.yesterday = cls.today - timedelta(days=1)
        cls.past_date = cls.today - timedelta(days=5)

        logger.info("setUpClass: Inserting test attendance records...")
        now_time = datetime.now().time()
        cls.db.insert_attendance("25a51a4470", "sashi", cls.today, now_time, "Present", department="CSD", hourly_period="09:15-10:20")
        logger.info("setUpClass: Inserted record 1")
        cls.db.insert_attendance("25a51a4470", "sashi", cls.yesterday, time(9, 30, 0), "Present", department="CSD", hourly_period="09:15-10:20")
        logger.info("setUpClass: Inserted record 2")
        cls.db.insert_attendance("25a51a4470", "sashi", cls.past_date, time(9, 0, 0), "Present", department="CSD", hourly_period="09:15-10:20")
        logger.info("setUpClass: Inserted record 3")

        cls.db.insert_attendance("99E01", "EEE Student Test", cls.today, now_time, "Present", department="CSD", hourly_period="09:15-10:20")
        logger.info("setUpClass: Inserted record 4")
        cls.db.insert_attendance("99E01", "EEE Student Test", cls.yesterday, time(10, 15, 0), "Present", department="CSD", hourly_period="09:15-10:20")
        logger.info("setUpClass: Inserted record 5")

    def setUp(self):
        app.testing = True
        app.config["SECRET_KEY"] = "test-secret-key"
        self.client = app.test_client()

    # ── TEST 1: Today's attendance retrieval ─────────────────
    def test_01_todays_attendance_retrieval(self):
        """GET /api/attendance/today returns today's attendance records."""
        _login(self.client, "cse_hod", "cse@hod2026")
        response = self.client.get("/api/attendance/today?period=ALL")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("attendance", data)
        self.assertGreaterEqual(data["count"], 1)
        logger.info("TEST 1 PASSED: Today's attendance retrieval works.")

    # ── TEST 2: Single-date filtering ────────────────────────
    def test_02_single_date_filtering(self):
        """GET /api/attendance/history with start_date==end_date filters exact date."""
        _login(self.client, "admin", "admin@2026")
        target_str = self.yesterday.isoformat()
        response = self.client.get(f"/api/attendance/history?start_date={target_str}&end_date={target_str}")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(all(r["attendance_date"] == target_str for r in data["records"]))
        logger.info("TEST 2 PASSED: Single-date filtering works.")

    # ── TEST 3: Date-range filtering ─────────────────────────
    def test_03_date_range_filtering(self):
        """GET /api/attendance/history with date range returns matching records."""
        _login(self.client, "admin", "admin@2026")
        start_str = self.past_date.isoformat()
        end_str = self.yesterday.isoformat()
        response = self.client.get(f"/api/attendance/history?start_date={start_str}&end_date={end_str}")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        for r in data["records"]:
            rec_date = date.fromisoformat(r["attendance_date"])
            self.assertTrue(self.past_date <= rec_date <= self.yesterday)
        logger.info("TEST 3 PASSED: Date-range filtering works.")

    # ── TEST 4: Search by student ID ─────────────────────────
    def test_04_search_by_student_id(self):
        """GET /api/attendance/history?search=25a51a4470 matches student ID."""
        _login(self.client, "admin", "admin@2026")
        response = self.client.get("/api/attendance/history?search=25a51a4470")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertGreaterEqual(data["count"], 1)
        self.assertTrue(all("25a51a4470" in r["student_id"] for r in data["records"]))
        logger.info("TEST 4 PASSED: Search by student ID works.")

    # ── TEST 5: Search by student name ───────────────────────
    def test_05_search_by_student_name(self):
        """GET /api/attendance/history?search=sashi matches student name."""
        _login(self.client, "admin", "admin@2026")
        response = self.client.get("/api/attendance/history?search=sashi")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertGreaterEqual(data["count"], 1)
        self.assertTrue(all("sashi" in r["student_name"].lower() for r in data["records"]))
        logger.info("TEST 5 PASSED: Search by student name works.")

    # ── TEST 6: CSD department filtering ─────────────────────
    def test_06_cse_department_filtering(self):
        """Admin GET /api/attendance/history?dept=CSD returns CSD records."""
        _login(self.client, "admin", "admin@2026")
        response = self.client.get("/api/attendance/history?dept=CSD")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(all(r["department"] == "CSD" for r in data["records"]))
        logger.info("TEST 6 PASSED: CSD department filtering works.")

    # ── TEST 7: HOD department isolation ────────────────────
    def test_07_hod_department_isolation(self):
        """HOD requesting ?dept=EEE is forced to CSD records only."""
        _login(self.client, "cse_hod", "cse@hod2026")
        response = self.client.get("/api/attendance/history?dept=EEE")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["department_code"], "CSD")
        self.assertTrue(all(r["department"] == "CSD" for r in data["records"]))
        logger.info("TEST 7 PASSED: HOD department isolation enforced on history API.")

    # ── TEST 8: Admin ALL-department access ──────────────────
    def test_08_admin_all_department_access(self):
        """Admin GET /api/attendance/history?dept=ALL returns records across departments."""
        _login(self.client, "admin", "admin@2026")
        response = self.client.get("/api/attendance/history?dept=ALL")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["department_code"], "ALL")
        logger.info("TEST 8 PASSED: Admin ALL-department access works.")

    # ── TEST 9: Unauthorized access protection ───────────────
    def test_09_unauthorized_access_protection(self):
        """Unauthenticated requests to history and export return 302 redirect to /login."""
        res_hist = self.client.get("/api/attendance/history")
        self.assertEqual(res_hist.status_code, 302)

        res_exp = self.client.get("/api/attendance/export")
        self.assertEqual(res_exp.status_code, 302)
        logger.info("TEST 9 PASSED: Unauthorized requests protected (302 redirect).")

    # ── TEST 10: CSV generation and correct headers ─────────
    def test_10_csv_generation_and_headers(self):
        """GET /api/attendance/export returns CSV with correct columns."""
        _login(self.client, "admin", "admin@2026")
        response = self.client.get("/api/attendance/export")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/csv")
        self.assertIn("Content-Disposition", response.headers)

        # Parse CSV output
        csv_text = response.data.decode("utf-8")
        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)

        self.assertGreater(len(rows), 0)
        expected_headers = [
            "Student ID",
            "Student Name",
            "Department",
            "Attendance Date",
            "Attendance Time",
            "Hourly Period",
            "Status",
        ]
        self.assertEqual(rows[0], expected_headers)
        logger.info("TEST 10 PASSED: CSV generation and header validation successful.")

    # ── TEST 11: HOD CSV department isolation ────────────────
    def test_11_hod_csv_department_isolation(self):
        """HOD exporting CSV receives only their department records."""
        _login(self.client, "cse_hod", "cse@hod2026")
        response = self.client.get("/api/attendance/export?dept=EEE")
        self.assertEqual(response.status_code, 200)

        csv_text = response.data.decode("utf-8")
        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)

        # Row 0 is header; data rows start at index 1
        for row in rows[1:]:
            self.assertEqual(row[2], "CSD")  # Department column must be CSD
        logger.info("TEST 11 PASSED: HOD CSV department isolation enforced.")

    # ── TEST 12: Admin CSV department filtering ──────────────
    def test_12_admin_csv_department_filtering(self):
        """Admin can export CSV for ALL or selected department."""
        _login(self.client, "admin", "admin@2026")
        response_all = self.client.get("/api/attendance/export?dept=ALL")
        self.assertEqual(response_all.status_code, 200)

        response_csd = self.client.get("/api/attendance/export?dept=CSD")
        self.assertEqual(response_csd.status_code, 200)

        reader_csd = csv.reader(io.StringIO(response_csd.data.decode("utf-8")))
        rows_csd = list(reader_csd)
        for row in rows_csd[1:]:
            self.assertEqual(row[2], "CSD")

        logger.info("TEST 12 PASSED: Admin CSV department filtering works.")


def run_tests():
    """Run Attendance History test suite."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAttendanceHistory)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
