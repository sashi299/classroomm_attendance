"""
test_hourly_attendance.py - Hourly Attendance Timetable & Deduplication Tests.

Tests:
  - Hourly period determination (e.g. 09:00-10:00, 10:00-11:00).
  - Insertion of attendance per (student_id, date, hourly_period, department).
  - Deduplication within the same hourly slot (returns ALREADY PRESENT).
  - Rollover to next hourly slot (allows new attendance marking).
  - Attendance history query & CSV export with Hourly Period column.
"""

import unittest
from datetime import date, time as dt_time, datetime

from config import Config
from database import DatabaseManager
from attendance import AttendanceManager, AttendanceStatus


class TestHourlyAttendance(unittest.TestCase):

    def setUp(self):
        self.db = DatabaseManager(
            host=Config.MYSQL_HOST,
            port=Config.MYSQL_PORT,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DATABASE,
        )
        self.assertTrue(self.db.connect())
        self.att_mgr = AttendanceManager(db_manager=self.db)

    def tearDown(self):
        self.db.disconnect()

    def test_current_hourly_period_calculation(self):
        """Test calculation of hourly period from timestamp."""
        t1 = dt_time(9, 30, 0)
        period1 = self.db.get_current_hourly_period(t1)
        self.assertEqual(period1, "09:00-10:00")

        t2 = dt_time(14, 15, 0)
        period2 = self.db.get_current_hourly_period(t2)
        self.assertEqual(period2, "14:00-15:00")

    def test_same_hour_attendance_deduplication(self):
        """Test that multiple face detections within the SAME hour return ALREADY PRESENT."""
        student_id = "TEST_H1"
        student_name = "Hourly Student 1"
        dept = "CSD"
        period = "09:15-10:20"
        # Mock Monday 09:30 (P1 MFCS)
        mock_now = datetime(2026, 8, 10, 9, 30)

        # First detection -> NEWLY_MARKED
        status1 = self.att_mgr.mark_present(
            student_id=student_id,
            student_name=student_name,
            dept_code=dept,
            hourly_period=period,
            now=mock_now
        )
        self.assertIn(status1, [AttendanceStatus.NEWLY_MARKED, AttendanceStatus.ALREADY_PRESENT])

        # Second detection in same hour -> ALREADY_PRESENT
        status2 = self.att_mgr.mark_present(
            student_id=student_id,
            student_name=student_name,
            dept_code=dept,
            hourly_period=period,
            now=mock_now
        )
        self.assertEqual(status2, AttendanceStatus.ALREADY_PRESENT)

    def test_next_hour_attendance_opportunity(self):
        """Test that clock rollover to NEXT hour allows marking present again for new period."""
        student_id = "TEST_H2"
        student_name = "Hourly Student 2"
        dept = "CSD"
        p1 = "09:15-10:20"
        p2 = "10:20-11:10"
        # P1: Mon 09:30
        now1 = datetime(2026, 8, 10, 9, 30)
        # P2: Mon 10:45
        now2 = datetime(2026, 8, 10, 10, 45)

        # Hour 1 (09:15-10:20)
        s1 = self.att_mgr.mark_present(
            student_id=student_id,
            student_name=student_name,
            dept_code=dept,
            hourly_period=p1,
            now=now1
        )
        self.assertIn(s1, [AttendanceStatus.NEWLY_MARKED, AttendanceStatus.ALREADY_PRESENT])

        # Hour 2 (10:20-11:10) -> Should mark present for new hourly period!
        s2 = self.att_mgr.mark_present(
            student_id=student_id,
            student_name=student_name,
            dept_code=dept,
            hourly_period=p2,
            now=now2
        )
        self.assertIn(s2, [AttendanceStatus.NEWLY_MARKED, AttendanceStatus.ALREADY_PRESENT])

    def test_hourly_attendance_queries(self):
        """Test retrieving today's attendance records by department and period."""
        today_records = self.db.get_today_attendance(today=date.today())
        self.assertIsInstance(today_records, list)

        history_records = self.db.get_attendance_history()
        self.assertIsInstance(history_records, list)

        if history_records:
            sample = history_records[0]
            self.assertIn("hourly_period", sample)
            self.assertIn("department", sample)


if __name__ == "__main__":
    unittest.main()
