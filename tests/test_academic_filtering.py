import unittest
import os
import sys
from datetime import date, datetime

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from database import DatabaseManager
from config import Config

class TestAcademicFiltering(unittest.TestCase):
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

        # Ensure some trial data exists via seeding
        cls.db.seed_csd_b_timetable()

    def test_01_timetable_filtering_academic_year(self):
        """get_timetable filters by academic_year."""
        # Seeding creates 2026-27 records
        recs = self.db.get_timetable(department="CSD", section="B", academic_year="2026-27")
        self.assertGreater(len(recs), 0)
        self.assertTrue(all(r["academic_year"] == "2026-27" for r in recs))

        # Non-existent year
        recs_empty = self.db.get_timetable(department="CSD", section="B", academic_year="2099-00")
        self.assertEqual(len(recs_empty), 0)

    def test_02_timetable_filtering_year_level(self):
        """get_timetable filters by year_level."""
        # Seeding creates Year Level 'II'
        recs = self.db.get_timetable(department="CSD", section="B", year_level="II")
        self.assertGreater(len(recs), 0)
        self.assertTrue(all(r["year_level"] == "II" for r in recs))

        # Non-existent year level
        recs_empty = self.db.get_timetable(department="CSD", section="B", year_level="IV")
        self.assertEqual(len(recs_empty), 0)

    def test_03_timetable_filtering_semester(self):
        """get_timetable filters by semester."""
        # Seeding creates Semester 'I'
        recs = self.db.get_timetable(department="CSD", section="B", semester="I")
        self.assertGreater(len(recs), 0)
        self.assertTrue(all(r["semester"] == "I" for r in recs))

        # Non-existent semester
        recs_empty = self.db.get_timetable(department="CSD", section="B", semester="II")
        self.assertEqual(len(recs_empty), 0)

    def test_04_timetable_filtering_combined(self):
        """get_timetable filters by multiple academic criteria."""
        recs = self.db.get_timetable(
            department="CSD",
            section="B",
            academic_year="2026-27",
            year_level="II",
            semester="I"
        )
        self.assertGreater(len(recs), 0)
        self.assertTrue(all(
            r["academic_year"] == "2026-27" and
            r["year_level"] == "II" and
            r["semester"] == "I" for r in recs
        ))

    def test_05_backward_compatibility(self):
        """get_timetable works without new filters (returns all years/sems for that dept/sec)."""
        recs = self.db.get_timetable(department="CSD", section="B")
        self.assertGreater(len(recs), 0)
        # Should include the seeded 2026-27 records
        self.assertTrue(any(r["academic_year"] == "2026-27" for r in recs))

    def test_06_report_filtering(self):
        """get_attendance_report_data respects academic filters."""
        # We need a date range that has timetable slots.
        # Monday Jan 7 2030 (referenced in other tests) has slots if we didn't change the seed year.
        # Wait, the seed year is 2026-27.
        # Today is Aug 11, 2026.
        today = date.today()

        # Ensure at least one student is "registered" for the report to have records
        # Fallback in report relies on db_records or registered_students list
        reg_students = [{"student_id": "ST_FILT_01", "student_name": "Filter Test student"}]

        report = self.db.get_attendance_report_data(
            start_date=today,
            end_date=today,
            department="CSD",
            section="B",
            academic_year="2026-27",
            year_level="II",
            semester="I",
            registered_students=reg_students
        )
        # If it's a weekday, we should have slots.
        if today.weekday() < 6: # Not Sunday
            self.assertIn("records", report)
            self.assertGreater(len(report["all_records"]), 0)

        # Empty report for wrong year
        report_empty = self.db.get_attendance_report_data(
            start_date=today,
            end_date=today,
            department="CSD",
            section="B",
            academic_year="2099-00"
        )
        self.assertEqual(len(report_empty["all_records"]), 0)

if __name__ == "__main__":
    unittest.main()
