import unittest
from datetime import datetime, date, time, timedelta
from unittest.mock import MagicMock, patch
import os
import sys

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from database import DatabaseManager
from notifications import NotificationManager, NotificationProvider
from config import Config

class MockProvider(NotificationProvider):
    def __init__(self):
        self.sent_notifications = []

    def send_attendance_report(self, recipient, role, report_summary):
        self.sent_notifications.append({
            "recipient": recipient,
            "role": role,
            "summary": report_summary
        })

class TestNotifications(unittest.TestCase):
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
        cls.db.seed_csd_b_timetable()

    def setUp(self):
        # Clear all logs from the future for a clean test environment
        with self.db._lock:
            cursor = self.db._connection.cursor()
            cursor.execute("DELETE FROM notifications_log WHERE attendance_date >= '2030-01-01'")
            self.db._connection.commit()
            cursor.close()

        self.provider = MockProvider()
        self.manager = NotificationManager(self.db, provider=self.provider)

    def test_01_trigger_after_period_ends(self):
        """Notification is triggered for Faculty and HOD after a period ends."""
        # Monday P1 ends at 10:20. Let's pretend it's Monday 10:25 AM.
        monday_date = date(2030, 1, 7)
        monday_now = datetime.combine(monday_date, time(10, 25, 0))

        # Clear existing logs for this date just in case
        with self.db._lock:
            cursor = self.db._connection.cursor()
            cursor.execute("DELETE FROM notifications_log WHERE attendance_date = %s", (monday_date,))
            self.db._connection.commit()
            cursor.close()

        self.manager.process_pending_notifications(now=monday_now)

        # Should have sent 2 notifications: 1 for Faculty, 1 for HOD
        self.assertEqual(len(self.provider.sent_notifications), 2)

        roles = [n["role"] for n in self.provider.sent_notifications]
        self.assertIn("FACULTY", roles)
        self.assertIn("HOD", roles)

        # Verify it was logged in DB
        self.assertTrue(self.db.is_notification_sent("CSD", "B", monday_date, 1, "FACULTY"))
        self.assertTrue(self.db.is_notification_sent("CSD", "B", monday_date, 1, "HOD"))

    def test_02_duplicate_prevention(self):
        """Subsequent processing does not resend notifications for the same period."""
        monday_date = date(2030, 1, 14) # Also a Monday
        monday_now = datetime.combine(monday_date, time(10, 25, 0))

        # First run
        self.manager.process_pending_notifications(now=monday_now)
        self.assertEqual(len(self.provider.sent_notifications), 2)

        # Second run immediately after
        self.manager.process_pending_notifications(now=monday_now)
        self.assertEqual(len(self.provider.sent_notifications), 2) # Still 2, no more added

    def test_03_no_trigger_before_end(self):
        """Notification is not triggered before the period end time."""
        monday_date = date(2030, 1, 21) # Also a Monday
        # P1 ends at 10:20. Check at 10:15.
        monday_now = datetime.combine(monday_date, time(10, 15, 0))

        self.manager.process_pending_notifications(now=monday_now)
        self.assertEqual(len(self.provider.sent_notifications), 0)

    def test_04_recipient_resolution(self):
        """Notification contains correct recipient contact details."""
        monday_date = date(2030, 1, 28) # Also a Monday
        monday_now = datetime.combine(monday_date, time(10, 25, 0))

        self.manager.process_pending_notifications(now=monday_now)

        # Check Faculty contact (seeded in database.py for P1 MON)
        # ("...MFCS", "THEORY", "Dr. S. Kumar", "faculty.csd1@example.com")
        fac_notif = next(n for n in self.provider.sent_notifications if n["role"] == "FACULTY")
        self.assertEqual(fac_notif["recipient"], "faculty.csd1@example.com")

        # Check HOD contact (seeded in database.py for CSD)
        hod_notif = next(n for n in self.provider.sent_notifications if n["role"] == "HOD")
        self.assertEqual(hod_notif["recipient"], "hod.csd@example.com")

if __name__ == "__main__":
    unittest.main()
