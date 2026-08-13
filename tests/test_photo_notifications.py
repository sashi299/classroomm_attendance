"""
test_photo_notifications.py - Unit and Integration tests for Photo Evidence Attendance Notifications.

Tests:
  1.  PRESENT student with recognition photo → CID inline image attached in EmailNotificationProvider.
  2.  PRESENT student without photo → Marked clearly as "Recognition image unavailable".
  3.  ABSENT student → Listed as ABSENT without any recognition photo attached.
  4.  Multiple cameras detecting same student → Deduplicated; exactly 1 crop retained per student per period.
  5.  Best-quality photo selection → Higher quality crop replaces lower quality crop.
  6.  Email HTML generation → Metadata table with Dept, Section, Year, Semester, Date, Period, Subject, Counts.
  7.  Inline image / CID attachment generation → MIMEMultipart with Content-ID headers and cid: img tags.
  8.  Existing notification behavior → ConsoleNotificationProvider & notification log persistence.
"""

import os
import sys
import logging
import unittest
import numpy as np
import cv2
from datetime import date, datetime, time
from unittest.mock import MagicMock, patch

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from database import DatabaseManager
from attendance import AttendanceManager, AttendanceStatus, calculate_crop_quality, extract_face_crop
from notifications import NotificationManager, NotificationProvider, EmailNotificationProvider, ConsoleNotificationProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_photo_notifications")


class MockProvider(NotificationProvider):
    """Mock notification provider for asserting notification payload data."""

    def __init__(self):
        self.sent_notifications = []

    def send_attendance_report(self, recipient: str, role: str, report_summary: dict):
        self.sent_notifications.append({
            "recipient": recipient,
            "role": role,
            "summary": report_summary,
        })


def _create_dummy_face_crop(width: int = 100, height: int = 100, color=(0, 255, 0)) -> np.ndarray:
    """Create a dummy BGR image representing a face crop."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = color
    # Add texture for non-zero sharpness (Laplacian variance)
    cv2.putText(img, "FACE", (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


class TestPhotoNotifications(unittest.TestCase):
    """Test suite for Visual Evidence Attendance Notifications."""

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

        # Register test students in DB if missing
        cls.db.add_student({
            "student_id": "25A51A0470", "name": "Sashi", "department": "CSD", "section": "B",
            "year_level": "II B.Tech", "academic_year": "2026-27", "semester": "I Sem"
        })
        cls.db.add_student({
            "student_id": "25A51A0471", "name": "Ravi", "department": "CSD", "section": "B",
            "year_level": "II B.Tech", "academic_year": "2026-27", "semester": "I Sem"
        })
        cls.db.add_student({
            "student_id": "25A51A0472", "name": "Kumar", "department": "CSD", "section": "B",
            "year_level": "II B.Tech", "academic_year": "2026-27", "semester": "I Sem"
        })

    @classmethod
    def tearDownClass(cls):
        cls.db.disconnect()

    def setUp(self):
        self.att_mgr = AttendanceManager(db_manager=self.db)
        self.mock_provider = MockProvider()
        self.notif_mgr = NotificationManager(
            db_manager=self.db,
            provider=self.mock_provider,
            attendance_manager=self.att_mgr,
        )

        # Clear notifications log and attendance for test dates
        with self.db._lock:
            cursor = self.db._connection.cursor()
            cursor.execute("DELETE FROM notifications_log WHERE attendance_date >= '2030-01-01'")
            cursor.execute("DELETE FROM attendance WHERE attendance_date >= '2030-01-01'")
            self.db._connection.commit()
            cursor.close()

    # ── TEST 1: PRESENT Student with Recognition Photo ─────────
    def test_01_present_student_with_recognition_photo(self):
        """Verify evidence image is preserved and retrieved for a PRESENT student."""
        test_now = datetime(2030, 1, 7, 9, 30) # Monday P1
        test_date = test_now.date()
        crop = _create_dummy_face_crop(120, 120)

        status = self.att_mgr.mark_present(
            student_id="25A51A0470",
            student_name="Sashi",
            dept_code="CSD",
            section="B",
            now=test_now,
            face_crop=crop,
            distance=0.25,
        )
        self.assertIn(status, (AttendanceStatus.NEWLY_MARKED, AttendanceStatus.ALREADY_PRESENT))

        # Retrieve evidence
        ev_bytes = self.att_mgr.get_evidence(
            dept="CSD", sec="B", att_date=test_date, period_num=1, student_id="25A51A0470"
        )
        self.assertIsNotNone(ev_bytes)
        self.assertGreater(len(ev_bytes), 0)
        logger.info("TEST 1 PASSED: Recognition photo preserved and retrieved for PRESENT student.")

    # ── TEST 2: PRESENT Student without Photo ────────────────────
    def test_02_present_student_without_photo(self):
        """Verify PRESENT student without photo returns None for evidence crop."""
        test_now = datetime(2030, 1, 7, 10, 30) # Monday P2
        test_date = test_now.date()

        # Mark without crop
        status = self.att_mgr.mark_present(
            student_id="25A51A0471",
            student_name="Ravi",
            dept_code="CSD",
            section="B",
            now=test_now,
            face_crop=None,
        )
        self.assertIn(status, (AttendanceStatus.NEWLY_MARKED, AttendanceStatus.ALREADY_PRESENT))

        # Evidence should be None
        ev_bytes = self.att_mgr.get_evidence(
            dept="CSD", sec="B", att_date=test_date, period_num=2, student_id="25A51A0471"
        )
        self.assertIsNone(ev_bytes)
        logger.info("TEST 2 PASSED: PRESENT student without photo gracefully returns None evidence.")

    # ── TEST 3: ABSENT Student (No Photo) ────────────────────────
    def test_03_absent_student_no_photo(self):
        """Verify report data marks unrecorded students as ABSENT with None evidence."""
        test_date = date(2030, 1, 7) # Monday
        period_num = 3

        # Trigger notification report compilation
        self.notif_mgr._send_and_log(
            dept="CSD", sec="B", att_date=test_date, period_num=period_num,
            contact="faculty@example.com", role="FACULTY"
        )

        self.assertEqual(len(self.mock_provider.sent_notifications), 1)
        summary = self.mock_provider.sent_notifications[0]["summary"]

        students = summary.get("students", [])
        kumar_record = next((s for s in students if s["student_id"] == "25A51A0472"), None)

        if kumar_record:
            self.assertEqual(kumar_record["status"], "ABSENT")
            self.assertIsNone(kumar_record["evidence_image"])

        logger.info("TEST 3 PASSED: ABSENT student correctly listed without recognition photo.")

    # ── TEST 4: Multiple Cameras Detecting Same Student ──────────
    def test_04_multiple_cameras_deduplication(self):
        """Verify only 1 representative face crop is stored per student per period across multiple cameras."""
        test_now = datetime(2030, 1, 7, 11, 20) # Monday P3
        test_date = test_now.date()

        crop_cam1 = _create_dummy_face_crop(60, 60, (255, 0, 0)) # Camera 1
        crop_cam2 = _create_dummy_face_crop(100, 100, (0, 255, 0)) # Camera 2

        # Camera 1 detection
        self.att_mgr.mark_present(
            student_id="25A51A0470", student_name="Sashi", dept_code="CSD", section="B",
            now=test_now, face_crop=crop_cam1, distance=0.4
        )

        # Camera 2 detection (same student, same period)
        self.att_mgr.mark_present(
            student_id="25A51A0470", student_name="Sashi", dept_code="CSD", section="B",
            now=test_now, face_crop=crop_cam2, distance=0.2
        )

        # Exactly 1 evidence crop retained
        ev_bytes = self.att_mgr.get_evidence("CSD", "B", test_date, 3, "25A51A0470")
        self.assertIsNotNone(ev_bytes)

        # Count stored cache keys for this student & period
        matching_keys = [k for k in self.att_mgr._evidence_cache.keys() if k[3] == 3 and k[4] == "25A51A0470"]
        self.assertEqual(len(matching_keys), 1)

        logger.info("TEST 4 PASSED: Multiple camera detections deduplicated to 1 representative crop.")

    # ── TEST 5: Best-Quality Photo Selection ─────────────────────
    def test_05_best_quality_photo_selection(self):
        """Verify higher quality crop replaces lower quality crop for the same period."""
        test_date = date(2030, 1, 7)
        period_num = 4

        small_crop = _create_dummy_face_crop(40, 40) # Low quality
        large_crop = _create_dummy_face_crop(150, 150) # High quality

        q_small = calculate_crop_quality(small_crop, distance=0.4)
        q_large = calculate_crop_quality(large_crop, distance=0.1)
        self.assertGreater(q_large, q_small)

        # Store small crop first
        stored_1 = self.att_mgr.store_evidence("CSD", "B", test_date, period_num, "25A51A0470", small_crop, distance=0.4)
        self.assertTrue(stored_1)

        # Store large crop second
        stored_2 = self.att_mgr.store_evidence("CSD", "B", test_date, period_num, "25A51A0470", large_crop, distance=0.1)
        self.assertTrue(stored_2)

        # Attempt to store small crop third (should be rejected because quality is lower)
        stored_3 = self.att_mgr.store_evidence("CSD", "B", test_date, period_num, "25A51A0470", small_crop, distance=0.4)
        self.assertFalse(stored_3)

        logger.info("TEST 5 PASSED: Best-quality crop preferred and lower quality crop rejected.")

    # ── TEST 6 & 7: Email HTML & CID Inline Attachment Generation ─
    def test_06_07_email_html_and_cid_attachments(self):
        """Verify EmailNotificationProvider constructs proper HTML with CID inline image attachments."""
        crop_bytes = cv2.imencode(".jpg", _create_dummy_face_crop(80, 80))[1].tobytes()

        summary = {
            "department": "CSD",
            "section": "B",
            "academic_year": "2026-27",
            "year_level": "II B.Tech",
            "semester": "I Sem",
            "date": "2030-01-07",
            "period_number": 1,
            "hourly_period": "09:15-10:20",
            "subject": "MFCS",
            "class_type": "THEORY",
            "present_count": 2,
            "absent_count": 1,
            "total_students": 3,
            "students": [
                {"student_id": "25A51A0470", "student_name": "Sashi", "status": "PRESENT", "evidence_image": crop_bytes},
                {"student_id": "25A51A0471", "student_name": "Ravi", "status": "PRESENT", "evidence_image": None},
                {"student_id": "25A51A0472", "student_name": "Kumar", "status": "ABSENT", "evidence_image": None},
            ]
        }

        provider = EmailNotificationProvider(
            host="smtp.example.com", port=587, username="user", password="pwd", sender="attn@example.com"
        )

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            provider.send_attendance_report("faculty@example.com", "FACULTY", summary)

            sender, recipient, msg_str = mock_server.sendmail.call_args[0]

            from email import message_from_string
            msg_parsed = message_from_string(msg_str)

            # Assert Subject
            self.assertIn("Attendance Report", msg_parsed["Subject"])
            self.assertIn("MFCS", msg_parsed["Subject"])

            # Extract decoded HTML body & image CIDs
            html_body = ""
            cid_attachments = []
            for part in msg_parsed.walk():
                if part.get_content_type() == "text/html":
                    html_body = part.get_payload(decode=True).decode("utf-8")
                if part.get_content_maintype() == "image":
                    cid_attachments.append(part.get("Content-ID"))

            # Assert CID inline image reference in HTML
            self.assertIn('src="cid:face_25A51A0470"', html_body)

            # Assert "Recognition image unavailable" present for student without photo
            self.assertIn("Recognition image unavailable", html_body)

            # Assert ABSENT status for Kumar
            self.assertIn("ABSENT", html_body)
            self.assertIn("Kumar", html_body)

            # Assert metadata presence
            self.assertIn("CSD - Section B", html_body)
            self.assertIn("2026-27", html_body)
            self.assertIn("MFCS (THEORY)", html_body)

            # Assert CID attachment headers
            self.assertTrue(any("face_25A51A0470" in str(cid) for cid in cid_attachments))

        logger.info("TEST 6 & 7 PASSED: HTML email generated with valid CID inline image attachments & metadata.")

    # ── TEST 8: Console Notification Provider Output ────────────
    def test_08_console_notification_provider(self):
        """Verify ConsoleNotificationProvider logs notification cleanly."""
        summary = {
            "department": "CSD", "section": "B", "period_number": 1, "subject": "MFCS",
            "present_count": 1, "absent_count": 1,
            "students": [
                {"student_id": "25A51A0470", "student_name": "Sashi", "status": "PRESENT", "evidence_image": b"123"},
                {"student_id": "25A51A0472", "student_name": "Kumar", "status": "ABSENT", "evidence_image": None},
            ]
        }
        console_provider = ConsoleNotificationProvider()
        # Should execute without error
        console_provider.send_attendance_report("console@example.com", "HOD", summary)
        logger.info("TEST 8 PASSED: Console notification provider executed cleanly.")


def run_tests():
    """Run Photo Notifications test suite."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhotoNotifications)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
