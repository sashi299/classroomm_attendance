import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from notifications import EmailNotificationProvider

class TestEmailNotifications(unittest.TestCase):
    def setUp(self):
        self.host = "smtp.test.com"
        self.port = 587
        self.user = "test@test.com"
        self.pwd = "password"
        self.sender = "sender@test.com"
        self.provider = EmailNotificationProvider(self.host, self.port, self.user, self.pwd, self.sender)

    @patch("smtplib.SMTP")
    def test_send_email_calls_smtp(self, mock_smtp):
        """Verify that send_attendance_report calls smtplib with correct parameters."""
        instance = mock_smtp.return_value.__enter__.return_value

        report_summary = {
            "department": "CSD",
            "section": "B",
            "period_number": 1,
            "present_students": 25,
            "total_students": 30,
            "attendance_percentage": 83.3
        }

        recipient = "recipient@test.com"
        role = "FACULTY"

        self.provider.send_attendance_report(recipient, role, report_summary)

        # Check SMTP initialization
        mock_smtp.assert_called_with(self.host, self.port)

        # Check STARTTLS and Login
        instance.starttls.assert_called_once()
        instance.login.assert_called_once_with(self.user, self.pwd)

        # Check Sendmail
        instance.sendmail.assert_called_once()
        args = instance.sendmail.call_args[0]
        self.assertEqual(args[0], self.sender)
        self.assertEqual(args[1], recipient)
        self.assertIn("Subject: Attendance Report: CSD-B | Period 1", args[2])
        self.assertIn("FACULTY", args[2])
        self.assertIn("83.3%", args[2])

    def test_missing_config_skips_send(self):
        """Provider should log warning and return if not configured."""
        provider = EmailNotificationProvider("", 0, "", "", "")
        with self.assertLogs("notifications", level="WARNING") as cm:
            provider.send_attendance_report("test@test.com", "HOD", {})
            self.assertIn("Email provider not fully configured", cm.output[0])

if __name__ == "__main__":
    unittest.main()
