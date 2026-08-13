"""
test_timetable_attendance_integration.py - Integration tests for timetable-aware attendance.
(Standalone version without pytest)
"""

import os
import sys
import logging
from datetime import datetime, date, time as dt_time
from unittest.mock import MagicMock

# Add src/ to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from attendance import AttendanceManager, AttendanceStatus
from database import DatabaseManager
from system_state import system_state_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_integration")

class MockDB(DatabaseManager):
    def __init__(self):
        self._lock = MagicMock()
        self.records = []

    def _ensure_connection_unlocked(self): return True
    def connect(self): return True
    def disconnect(self): pass
    def get_current_hourly_period(self, time_val=None): return "09:00-10:00"

    def get_current_timetable_slot(self, department, section, now):
        day_code = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][now.weekday()]
        current_time = now.time()

        if day_code == "SUN":
            return {"status": "NO_CLASS"}

        if day_code == "MON":
            if current_time < dt_time(9, 15):
                return {"status": "BEFORE_CLASS"}
            if dt_time(9, 15) <= current_time < dt_time(10, 20):
                return {
                    "status": "ACTIVE", "period_number": 1, "subject": "MFCS",
                    "start_time": "09:15:00", "end_time": "10:20:00", "class_type": "THEORY"
                }
            if dt_time(13, 0) <= current_time < dt_time(13, 40):
                return {"status": "LUNCH"}
            if dt_time(13, 40) <= current_time < dt_time(14, 30):
                return {
                    "status": "ACTIVE", "period_number": 5, "subject": "Python Lab",
                    "start_time": "13:40:00", "end_time": "14:30:00", "class_type": "LAB"
                }

        return {"status": "NO_CLASS"}

    def insert_attendance(self, student_id, student_name, attendance_date, attendance_time, status, hourly_period, department, section, period_number, subject, class_type):
        for r in self.records:
            if (r['student_id'] == student_id and
                r['attendance_date'] == attendance_date and
                r['hourly_period'] == hourly_period and
                r['department'] == department):
                return False

        self.records.append({
            "student_id": student_id,
            "student_name": student_name,
            "attendance_date": attendance_date,
            "attendance_time": attendance_time,
            "status": status,
            "hourly_period": hourly_period,
            "department": department,
            "section": section,
            "period_number": period_number,
            "subject": subject,
            "class_type": class_type
        })
        return True

def run_test(name, func):
    try:
        db = MockDB()
        mgr = AttendanceManager(db)
        system_state_manager.disable_exam_mode()
        func(mgr, db)
        print(f"PASSED: {name}")
        return True
    except Exception as e:
        print(f"FAILED: {name} - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_active_period_allowed(mgr, db):
    now = datetime(2026, 8, 10, 9, 30)
    status = mgr.mark_present("S1", "Student One", now=now)
    assert status == AttendanceStatus.NEWLY_MARKED
    assert len(db.records) == 1
    assert db.records[0]['subject'] == "MFCS"

def test_before_class_blocked(mgr, db):
    now = datetime(2026, 8, 10, 8, 0)
    status = mgr.mark_present("S1", "Student One", now=now)
    assert status == AttendanceStatus.SKIPPED_BEFORE_CLASS
    assert len(db.records) == 0

def test_lunch_blocked(mgr, db):
    now = datetime(2026, 8, 10, 13, 15)
    status = mgr.mark_present("S1", "Student One", now=now)
    assert status == AttendanceStatus.SKIPPED_LUNCH
    assert len(db.records) == 0

def test_sunday_blocked(mgr, db):
    now = datetime(2026, 8, 9, 10, 0)
    status = mgr.mark_present("S1", "Student One", now=now)
    assert status == AttendanceStatus.SKIPPED_NO_CLASS
    assert len(db.records) == 0

def test_exam_mode_blocked(mgr, db):
    system_state_manager.enable_exam_mode()
    now = datetime(2026, 8, 10, 9, 30)
    status = mgr.mark_present("S1", "Student One", now=now)
    assert status == AttendanceStatus.SKIPPED_EXAM_MODE
    assert len(db.records) == 0

def test_duplicate_period_prevented(mgr, db):
    now1 = datetime(2026, 8, 10, 9, 30)
    mgr.mark_present("S1", "Student One", now=now1)
    now2 = datetime(2026, 8, 10, 9, 45)
    status = mgr.mark_present("S1", "Student One", now=now2)
    assert status == AttendanceStatus.ALREADY_PRESENT
    assert len(db.records) == 1

def test_next_period_allowed(mgr, db):
    now1 = datetime(2026, 8, 10, 9, 30)
    mgr.mark_present("S1", "Student One", now=now1)
    now2 = datetime(2026, 8, 10, 14, 0)
    status = mgr.mark_present("S1", "Student One", now=now2)
    assert status == AttendanceStatus.NEWLY_MARKED
    assert len(db.records) == 2
    assert db.records[1]['subject'] == "Python Lab"

if __name__ == "__main__":
    tests = [
        ("Active period allowed", test_active_period_allowed),
        ("Before class blocked", test_before_class_blocked),
        ("Lunch blocked", test_lunch_blocked),
        ("Sunday blocked", test_sunday_blocked),
        ("Exam Mode blocked", test_exam_mode_blocked),
        ("Duplicate period prevented", test_duplicate_period_prevented),
        ("Next period allowed", test_next_period_allowed),
    ]
    results = [run_test(n, f) for n, f in tests]
    print(f"\nSummary: {sum(results)}/{len(tests)} passed.")
    sys.exit(0 if all(results) else 1)
