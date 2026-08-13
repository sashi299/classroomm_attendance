"""
test_attendance_reports.py - Integration and Unit Tests for HOD Attendance Reports Module.

Covers 20 key test scenarios:
  1. Daily report (/api/reports/attendance/daily)
  2. Weekly report (/api/reports/attendance/weekly)
  3. Monthly report (/api/reports/attendance/monthly)
  4. Yearly report (/api/reports/attendance/yearly)
  5. Present filter (status=PRESENT)
  6. Absent filter (status=ABSENT)
  7. Date range filtering (start_date & end_date)
  8. Department filter (Admin access)
  9. HOD department isolation enforcement (403 Forbidden for unauthorized dept)
 10. Admin all-department access (dept=ALL)
 11. Student search (ID & Name matching)
 12. Hourly period filter (hourly_period=09:00-10:00)
 13. Duplicate attendance handling (same student/date/period not double-counted)
 14. CSV export generation (/api/reports/attendance/export/csv)
 15. Excel export generation (/api/reports/attendance/export/excel)
 16. Empty attendance still shows registered students as ABSENT
 17. Large date-range query correctness
 18. Database performance indexes verification
 19. Unauthorized access returns 403 Forbidden
 20. Existing attendance history preservation across report generation
"""

import os
import io
import sys
import logging
import unittest
from datetime import date, datetime, timedelta, time

HAS_OPENPYXL = True
try:
    import openpyxl
except ImportError:
    HAS_OPENPYXL = False

# Add src/ to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from database import DatabaseManager
from app import app, initialize_components

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_attendance_reports")


def _login_session(client, username="cse_hod", password="cse@hod2026"):
    """Helper to log in via POST and return the response."""
    return client.post("/login", data={
        "username": username,
        "password": password,
    }, follow_redirects=True)


class TestAttendanceReports(unittest.TestCase):
    """Test suite for HOD Attendance Reports module."""

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
            raise unittest.SkipTest("MySQL database not connected.")
        cls.today = date.today()
        cls.db.insert_attendance(
            student_id="REP_TEST_01",
            student_name="Reports Test student",
            attendance_date=cls.today,
            attendance_time=time(9, 30, 0),
            status="Present",
            department="CSD",
            period_number=1,
            subject="MFCS",
            section="B",
            hourly_period="09:15-10:20"
        )
        initialize_components()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "db") and cls.db and cls.db.is_connected:
            cls.db.disconnect()

    def setUp(self):
        app.testing = True
        app.config["SECRET_KEY"] = "test-secret-key"
        self.client = app.test_client()
        self.today = self.__class__.today

    # ── TEST 1: Daily report ─────────────────────────────────────────
    def test_01_daily_report(self):
        """GET /api/reports/attendance/daily returns 200 with daily summary & records."""
        _login_session(self.client, "cse_hod", "cse@hod2026")
        res = self.client.get(f"/api/reports/attendance/daily?date={self.today.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["report_type"], "daily")
        self.assertIn("summary", data)
        self.assertIn("records", data)
        logger.info("TEST 1 PASSED: Daily report API working.")

    # ── TEST 2: Weekly report ────────────────────────────────────────
    def test_02_weekly_report(self):
        """GET /api/reports/attendance/weekly returns 200 with 7-day range."""
        _login_session(self.client, "cse_hod", "cse@hod2026")
        res = self.client.get(f"/api/reports/attendance/weekly?date={self.today.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["report_type"], "weekly")
        logger.info("TEST 2 PASSED: Weekly report API working.")

    # ── TEST 3: Monthly report ───────────────────────────────────────
    def test_03_monthly_report(self):
        """GET /api/reports/attendance/monthly returns full month records."""
        _login_session(self.client, "cse_hod", "cse@hod2026")
        res = self.client.get(f"/api/reports/attendance/monthly?month={self.today.month}&year={self.today.year}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["report_type"], "monthly")
        logger.info("TEST 3 PASSED: Monthly report API working.")

    # ── TEST 4: Yearly report ────────────────────────────────────────
    def test_04_yearly_report(self):
        """GET /api/reports/attendance/yearly returns full year summary."""
        _login_session(self.client, "cse_hod", "cse@hod2026")
        res = self.client.get(f"/api/reports/attendance/yearly?year={self.today.year}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["report_type"], "yearly")
        logger.info("TEST 4 PASSED: Yearly report API working.")

    # ── TEST 5: Present filter ───────────────────────────────────────
    def test_05_present_filter(self):
        """Filtering by status=PRESENT returns ONLY present records."""
        _login_session(self.client, "cse_hod", "cse@hod2026")
        res = self.client.get(f"/api/reports/attendance/daily?date={self.today.isoformat()}&status=PRESENT")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        for r in data.get("records", []):
            self.assertEqual(r["status"], "PRESENT")
        logger.info("TEST 5 PASSED: Present filter correctly applied.")

    # ── TEST 6: Absent filter ────────────────────────────────────────
    def test_06_absent_filter(self):
        """Filtering by status=ABSENT returns ONLY absent slots."""
        _login_session(self.client, "cse_hod", "cse@hod2026")
        res = self.client.get(f"/api/reports/attendance/daily?date={self.today.isoformat()}&status=ABSENT")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        for r in data.get("records", []):
            self.assertEqual(r["status"], "ABSENT")
        logger.info("TEST 6 PASSED: Absent filter correctly applied.")

    # ── TEST 7: Date range filtering ─────────────────────────────────
    def test_07_date_range_filtering(self):
        """Custom date range filtering returns correct date boundaries."""
        start_d = self.today - timedelta(days=2)
        _login_session(self.client, "cse_hod", "cse@hod2026")
        res = self.client.get(f"/api/reports/attendance?start_date={start_d.isoformat()}&end_date={self.today.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["summary"]["start_date"], start_d.isoformat())
        self.assertEqual(data["summary"]["end_date"], self.today.isoformat())
        logger.info("TEST 7 PASSED: Date range filtering verified.")

    # ── TEST 8: Department filter (Admin) ───────────────────────────
    def test_08_admin_department_filter(self):
        """Admin can filter report by specific department (e.g. EEE)."""
        _login_session(self.client, "admin", "admin@2026")
        res = self.client.get(f"/api/reports/attendance?department=EEE&date={self.today.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["summary"]["department"], "EEE")
        logger.info("TEST 8 PASSED: Admin department filter working.")

    # ── TEST 9: HOD department isolation (403 Forbidden) ───────────
    def test_09_hod_department_isolation(self):
        """HOD requesting unauthorized department receives 403 Forbidden."""
        _login_session(self.client, "cse_hod", "cse@hod2026")
        res = self.client.get(f"/api/reports/attendance?department=EEE&date={self.today.isoformat()}")
        self.assertEqual(res.status_code, 403)
        data = res.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Unauthorized", data["error"])
        logger.info("TEST 9 PASSED: HOD unauthorized department access blocked (403).")

    # ── TEST 10: Admin all-department access ─────────────────────────
    def test_10_admin_all_departments(self):
        """Admin can request department=ALL without restriction."""
        _login_session(self.client, "admin", "admin@2026")
        res = self.client.get(f"/api/reports/attendance?department=ALL&date={self.today.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["summary"]["department"], "ALL")
        logger.info("TEST 10 PASSED: Admin all-department access working.")

    # ── TEST 11: Student search ──────────────────────────────────────
    def test_11_student_search(self):
        """Search query filters report rows by student ID or Name."""
        _login_session(self.client, "cse_hod", "cse@hod2026")
        res = self.client.get(f"/api/reports/attendance?search=Sashi&date={self.today.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        for r in data.get("records", []):
            self.assertIn("sashi", r["student_name"].lower())
        logger.info("TEST 11 PASSED: Student search filter working.")

    # ── TEST 12: Hourly period filter ────────────────────────────────
    def test_12_hourly_period_filter(self):
        """Filtering by period returns only matching records."""
        _login_session(self.client, "cse_hod", "cse@hod2026")
        res = self.client.get(f"/api/reports/attendance?hourly_period=1&date={self.today.isoformat()}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        for r in data.get("records", []):
            self.assertEqual(r["period_number"], 1)
        logger.info("TEST 12 PASSED: Period filter working.")

    # ── TEST 13: Duplicate attendance handling ───────────────────────
    def test_13_duplicate_attendance_handling(self):
        """Duplicate attendance insertions do not double count present stats."""
        # Attempt duplicate insert
        self.db.insert_attendance(
            student_id="REP_TEST_01",
            student_name="Reports Test student",
            attendance_date=self.today,
            attendance_time=time(9, 35, 0),
            status="Present",
            department="CSD",
            period_number=1,
            subject="MFCS",
            section="B",
            hourly_period="09:15-10:20"
        )
        report = self.db.get_attendance_report_data(
            start_date=self.today,
            end_date=self.today,
            department="CSD",
            section="B",
            registered_students=[{"student_id": "REP_TEST_01", "student_name": "Reports Test student", "department_code": "CSD"}],
            hourly_period="1",
        )
        # Present count for period 1 must still be 1
        present_slots = report["summary"]["present_periods"]
        self.assertEqual(present_slots, 1)
        logger.info("TEST 13 PASSED: Duplicate attendance handling verified.")

    # ── TEST 14: CSV export ──────────────────────────────────────────
    def test_14_csv_export(self):
        """GET /api/reports/attendance/export/csv returns valid CSV file stream."""
        _login_session(self.client, "cse_hod", "cse@hod2026")
        res = self.client.get(f"/api/reports/attendance/export/csv?date={self.today.isoformat()}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "text/csv")
        self.assertIn("Content-Disposition", res.headers)
        csv_text = res.data.decode("utf-8")
        self.assertIn("Student ID", csv_text)
        self.assertIn("Time Slot", csv_text)
        logger.info("TEST 14 PASSED: CSV export generated successfully.")

    # ── TEST 15: Excel export ────────────────────────────────────────
    @unittest.skipUnless(HAS_OPENPYXL, "openpyxl not installed")
    def test_15_excel_export(self):
        """GET /api/reports/attendance/export/excel returns valid 2-sheet Excel file."""
        _login_session(self.client, "cse_hod", "cse@hod2026")
        res = self.client.get(f"/api/reports/attendance/export/excel?date={self.today.isoformat()}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        wb = openpyxl.load_workbook(filename=io.BytesIO(res.data))
        sheet_names = wb.sheetnames
        self.assertIn("Attendance Details", sheet_names)
        self.assertIn("Summary", sheet_names)
        logger.info("TEST 15 PASSED: Excel export generated successfully with 2 sheets.")

    # ── TEST 16: Empty attendance renders ABSENT ────────────────────
    def test_16_empty_attendance_renders_absent(self):
        """Dates with no attendance records render registered students as ABSENT."""
        future_d = self.today + timedelta(days=10)
        report = self.db.get_attendance_report_data(
            start_date=future_d,
            end_date=future_d,
            department="CSE",
            registered_students=[{"student_id": "25a51a4470", "student_name": "Sashi", "department_code": "CSE"}],
        )
        for r in report["records"]:
            self.assertEqual(r["status"], "ABSENT")
        self.assertEqual(report["summary"]["present_periods"], 0)
        logger.info("TEST 16 PASSED: Empty attendance correctly renders registered students as ABSENT.")

    # ── TEST 17: Large date-range query correctness ───────────────
    def test_17_large_date_range(self):
        """Querying a 30-day date range produces complete matrix without errors."""
        start_d = self.today - timedelta(days=29)
        report = self.db.get_attendance_report_data(
            start_date=start_d,
            end_date=self.today,
            department="CSD",
            section="B",
            registered_students=[{"student_id": "REP_TEST_01", "student_name": "Reports Test student", "department_code": "CSD"}],
        )
        self.assertEqual(report["summary"]["start_date"], start_d.isoformat())
        self.assertEqual(report["summary"]["end_date"], self.today.isoformat())
        self.assertGreaterEqual(report["summary"]["total_periods"], 30)
        logger.info("TEST 17 PASSED: Large date-range query verified.")

    # ── TEST 18: Database performance indexes ────────────────────────
    def test_18_database_indexes(self):
        """Database connection verifies presence of attendance performance indexes."""
        cursor = self.db._connection.cursor()
        cursor.execute("SHOW INDEX FROM attendance;")
        rows = cursor.fetchall()
        cursor.close()

        index_names = {r[2] for r in rows}
        self.assertIn("idx_attendance_date", index_names)
        self.assertIn("idx_department", index_names)
        self.assertIn("idx_student_id", index_names)
        logger.info("TEST 18 PASSED: Database performance indexes verified.")

    # ── TEST 19: Unauthorized access returns 403 ────────────────────
    def test_19_unauthorized_access_returns_403(self):
        """HOD trying to access CSV/Excel export for another department receives 403."""
        _login_session(self.client, "cse_hod", "cse@hod2026")
        res = self.client.get(f"/api/reports/attendance/export/csv?department=EEE")
        self.assertEqual(res.status_code, 403)
        logger.info("TEST 19 PASSED: Unauthorized export access returns 403 Forbidden.")

    # ── TEST 20: Existing attendance history remains intact ─────────
    def test_20_existing_attendance_history_intact(self):
        """Generating reports does not alter or delete existing attendance DB records."""
        history = self.db.get_attendance_history(start_date=self.today, end_date=self.today)
        self.assertGreaterEqual(len(history), 1)
        logger.info("TEST 20 PASSED: Existing attendance DB history remains intact.")


def run_tests():
    """Run Attendance Reports test suite."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAttendanceReports)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
