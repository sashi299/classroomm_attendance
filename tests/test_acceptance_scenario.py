"""
test_acceptance_scenario.py - Real Acceptance Test for Production Workflow.

Simulates:
  Class: CSD-B, 1st Hour, DBMS (Period 1: 09:15-10:20)
  Students: 3 (Sashi, Ravi, Kumar)
  During period: Sashi & Ravi recognized, Kumar unrecognized.
  At period end: 2 PRESENT (with face crops), 1 ABSENT (no face crop).
  Verifies: Dashboard API, Attendance History, Evidence endpoints, Email Notification consistency.
"""

import os
import sys
import unittest
import logging
import numpy as np
from datetime import date, datetime, time as dt_time

# Add src/ to path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    import face_recognition
except ImportError:
    mock_dir = os.path.join(src_dir, "face_recognition_mock")
    if os.path.isdir(mock_dir) and mock_dir not in sys.path:
        sys.path.insert(0, mock_dir)
    import face_recognition_mock as mock_pkg
    sys.modules["face_recognition"] = mock_pkg

from config import Config
from database import DatabaseManager
from attendance import AttendanceManager, AttendanceStatus, extract_face_crop
from notifications import NotificationManager, NotificationProvider
from app import app, initialize_components

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_acceptance_scenario")


class MockEmailProvider(NotificationProvider):
    def __init__(self):
        self.sent_emails = []

    def send_attendance_report(self, recipient: str, role: str, report_summary: dict):
        self.sent_emails.append({
            "recipient": recipient,
            "role": role,
            "summary": report_summary,
        })


class TestAcceptanceScenario(unittest.TestCase):
    """Full End-to-End Acceptance Test for Classroom Attendance Workflow."""

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
            raise unittest.SkipTest("MySQL database unavailable.")

        cls.db.seed_csd_b_timetable()

    @classmethod
    def tearDownClass(cls):
        cls.db.disconnect()

    def setUp(self):
        self.test_date = date(2030, 1, 7)  # Monday
        self.dept = "CSD"
        self.sec = "B"

        # Clean test attendance records for test date
        with self.db._lock:
            cursor = self.db._connection.cursor()
            cursor.execute("DELETE FROM attendance WHERE attendance_date = %s", (self.test_date,))
            cursor.execute("DELETE FROM notifications_log WHERE attendance_date = %s", (self.test_date,))
            self.db._connection.commit()
            cursor.close()

        # Enforce exactly 3 students for CSD-B trial acceptance test
        self.db.add_student({
            "student_id": "ACC_01", "name": "Sashi", "department": "CSD", "section": "B",
            "year_level": "II B.Tech", "academic_year": "2026-27", "semester": "I Sem", "is_active": True
        })
        self.db.add_student({
            "student_id": "ACC_02", "name": "Ravi", "department": "CSD", "section": "B",
            "year_level": "II B.Tech", "academic_year": "2026-27", "semester": "I Sem", "is_active": True
        })
        self.db.add_student({
            "student_id": "ACC_03", "name": "Kumar", "department": "CSD", "section": "B",
            "year_level": "II B.Tech", "academic_year": "2026-27", "semester": "I Sem", "is_active": True
        })

        self.att_mgr = AttendanceManager(db_manager=self.db)
        self.email_provider = MockEmailProvider()
        self.notif_mgr = NotificationManager(
            db_manager=self.db,
            provider=self.email_provider,
            attendance_manager=self.att_mgr,
        )

        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_acceptance_workflow_period_1(self):
        """
        Acceptance Test:
          CSD-B Period 1 (09:15-10:20)
          Sashi & Ravi recognized -> PRESENT with face crop.
          Kumar unrecognized -> ABSENT at period completion.
        """
        # 1. During period (09:30 AM): Sashi & Ravi recognized by CCTV camera
        crop_sashi = np.zeros((100, 100, 3), dtype=np.uint8)
        crop_sashi[:] = (0, 255, 0)
        crop_ravi = np.zeros((100, 100, 3), dtype=np.uint8)
        crop_ravi[:] = (0, 200, 50)

        dt_sashi = datetime.combine(self.test_date, dt_time(9, 30, 0))
        dt_ravi = datetime.combine(self.test_date, dt_time(9, 45, 0))

        st_sashi = self.att_mgr.mark_present(
            student_id="ACC_01", student_name="Sashi", dept_code="CSD", section="B",
            now=dt_sashi, face_crop=crop_sashi, distance=0.25
        )
        st_ravi = self.att_mgr.mark_present(
            student_id="ACC_02", student_name="Ravi", dept_code="CSD", section="B",
            now=dt_ravi, face_crop=crop_ravi, distance=0.30
        )

        self.assertIn(st_sashi, (AttendanceStatus.NEWLY_MARKED, AttendanceStatus.ALREADY_PRESENT))
        self.assertIn(st_ravi, (AttendanceStatus.NEWLY_MARKED, AttendanceStatus.ALREADY_PRESENT))

        # 2. Period completion (10:20 AM): Trigger period finalization
        p_cnt, a_cnt, t_cnt = self.att_mgr.finalize_period_attendance(
            dept_code="CSD", section="B", att_date=self.test_date, period_number=1
        )

        self.assertEqual(p_cnt, 2)
        self.assertTrue(a_cnt >= 1)

        # 3. Verify Database Attendance History Consistency
        history = self.db.get_attendance_history(
            start_date=self.test_date, end_date=self.test_date, dept="CSD"
        )
        history_map = {r["student_id"]: r["status"] for r in history if r["hourly_period"] == "09:15-10:20"}

        self.assertEqual(history_map.get("ACC_01"), "Present")
        self.assertEqual(history_map.get("ACC_02"), "Present")
        self.assertEqual(history_map.get("ACC_03"), "Absent")

        # 4. Verify Face Recognition Evidence Consistency
        ev_sashi = self.att_mgr.get_evidence("CSD", "B", self.test_date, 1, "ACC_01")
        ev_ravi = self.att_mgr.get_evidence("CSD", "B", self.test_date, 1, "ACC_02")
        ev_kumar = self.att_mgr.get_evidence("CSD", "B", self.test_date, 1, "ACC_03")

        self.assertIsNotNone(ev_sashi)
        self.assertIsNotNone(ev_ravi)
        self.assertIsNone(ev_kumar)

        # 5. Verify Email Notification Report Payload
        self.notif_mgr._send_and_log(
            dept="CSD", sec="B", att_date=self.test_date, period_num=1,
            contact="faculty@example.com", role="FACULTY"
        )

        self.assertEqual(len(self.email_provider.sent_emails), 1)
        email_summary = self.email_provider.sent_emails[0]["summary"]

        self.assertEqual(email_summary["present_count"], 2)
        self.assertTrue(email_summary["absent_count"] >= 1)

        students_list = email_summary.get("students", [])
        s_sashi = next(s for s in students_list if s["student_id"] == "ACC_01")
        s_ravi = next(s for s in students_list if s["student_id"] == "ACC_02")
        s_kumar = next(s for s in students_list if s["student_id"] == "ACC_03")

        self.assertEqual(s_sashi["status"], "PRESENT")
        self.assertIsNotNone(s_sashi["evidence_image"])

        self.assertEqual(s_ravi["status"], "PRESENT")
        self.assertIsNotNone(s_ravi["evidence_image"])

        self.assertEqual(s_kumar["status"], "ABSENT")
        self.assertIsNone(s_kumar["evidence_image"])

        logger.info("ACCEPTANCE TEST PASSED: 2 PRESENT (with photos), 1 ABSENT (without photo) across all layers.")


if __name__ == "__main__":
    unittest.main()
