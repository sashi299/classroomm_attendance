"""
test_timetable_dashboard.py - Focused tests for Timetable integration on the Dashboard.
"""

import os
import sys
import unittest
from datetime import datetime

# Add src/ to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from app import app

def _login_session(client, username="cse_hod", password="cse@hod2026"):
    return client.post("/login", data={
        "username": username,
        "password": password,
    }, follow_redirects=True)

class TestTimetableDashboard(unittest.TestCase):
    def setUp(self):
        app.testing = True
        app.config["SECRET_KEY"] = "test-secret-key"
        self.client = app.test_client()

    def test_api_timetable_current_exists(self):
        _login_session(self.client)
        response = self.client.get("/api/timetable/current?department=CSD&section=B")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("status", data)

    def test_api_timetable_schedule_exists(self):
        _login_session(self.client)
        response = self.client.get("/api/timetable/schedule?department=CSD&section=B&day=MON")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("timetable", data)

    def test_api_today_attendance_structure_updated(self):
        _login_session(self.client)
        response = self.client.get("/api/attendance/today?dept=CSD")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("attendance", data)
        # If there are records, verify the status is uppercase
        for rec in data["attendance"]:
            self.assertEqual(rec["status"], "PRESENT")

    def test_dashboard_html_contains_timetable_elements(self):
        _login_session(self.client)
        response = self.client.get("/")
        html = response.get_data(as_text=True)
        self.assertIn('id="current-class-card"', html)
        self.assertIn('id="timetable-section"', html)
        self.assertIn('id="current-subject"', html)
        self.assertIn("Today's Timetable Schedule (CSD-B)", html)

if __name__ == "__main__":
    unittest.main()
