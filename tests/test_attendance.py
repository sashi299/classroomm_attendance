"""
test_attendance.py - Test suite for AttendanceManager module.

Tests in-memory caching, duplicate attendance prevention,
rejection of unknown students, status text formatting, and date rollover.
"""

import os
import sys
import logging
from datetime import date, time as dt_time, datetime, timedelta

# Add src/ to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from attendance import AttendanceManager, AttendanceStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_attendance")


class MockDatabaseManager:
    """Mock DatabaseManager for standalone unit testing without MySQL."""

    def __init__(self, should_succeed=True):
        self.should_succeed = should_succeed
        self.records = set()  # set of (student_id, date)

    def insert_attendance(self, student_id, student_name, attendance_date=None, attendance_time=None, status="Present", *args, **kwargs):
        if not self.should_succeed:
            return False
        key = (student_id, attendance_date or date.today())
        if key in self.records:
            return False  # Duplicate IGNORED by DB
        self.records.add(key)
        return True

    def check_attendance_exists(self, student_id, attendance_date, hourly_period="09:00-10:00", department="CSE"):
        return (student_id, attendance_date) in self.records

    def get_current_hourly_period(self, time_val=None):
        return "09:00-10:00"

    def get_today_attendance(self, today=None, dept_code=None, period=None):
        return [{"student_id": s_id} for (s_id, d) in self.records]


def run_tests():
    """Run AttendanceManager unit tests."""
    test_results = []

    # ===================================================================
    # TEST 1: Rejection of Unknown & empty student IDs
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Rejection of Unknown/Empty Student IDs")
    logger.info("=" * 60)

    mock_db = MockDatabaseManager()
    att_mgr = AttendanceManager(db_manager=mock_db)

    status_unknown = att_mgr.mark_present("Unknown", "Unknown Person")
    status_empty = att_mgr.mark_present("", "No Name")
    status_none = att_mgr.mark_present(None, "None")

    if (
        status_unknown == AttendanceStatus.SKIPPED_UNKNOWN
        and status_empty == AttendanceStatus.SKIPPED_UNKNOWN
        and status_none == AttendanceStatus.SKIPPED_UNKNOWN
        and len(mock_db.records) == 0
    ):
        logger.info("TEST 1: PASSED - Unknown and empty student IDs skipped without DB insert.")
        test_results.append(("Unknown/Empty Student ID Rejection", True))
    else:
        logger.error("TEST 1: FAILED - Unknown student was not skipped properly.")
        test_results.append(("Unknown/Empty Student ID Rejection", False))

    # ===================================================================
    # TEST 2: Marking valid student & duplicate prevention
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Marking Attendance & In-Memory Cache Deduplication")
    logger.info("=" * 60)

    res_first = att_mgr.mark_present("22A01", "Sashi")
    res_second = att_mgr.mark_present("22A01", "Sashi")
    res_third = att_mgr.mark_present("22A01", "Sashi")

    if (
        res_first == AttendanceStatus.NEWLY_MARKED
        and res_second == AttendanceStatus.ALREADY_PRESENT
        and res_third == AttendanceStatus.ALREADY_PRESENT
        and att_mgr.get_today_count() == 1
    ):
        logger.info("TEST 2: PASSED - First attempt marked PRESENT; duplicates returned ALREADY PRESENT.")
        test_results.append(("Attendance Marking & Cache Deduplication", True))
    else:
        logger.error("TEST 2: FAILED - Unexpected status sequence: %s, %s", res_first, res_second)
        test_results.append(("Attendance Marking & Cache Deduplication", False))

    # ===================================================================
    # TEST 3: Multiple distinct students
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Multiple Distinct Students Attendance")
    logger.info("=" * 60)

    res_ravi = att_mgr.mark_present("22A02", "Ravi")
    res_priya = att_mgr.mark_present("22A03", "Priya")

    if (
        res_ravi == AttendanceStatus.NEWLY_MARKED
        and res_priya == AttendanceStatus.NEWLY_MARKED
        and att_mgr.get_today_count() == 3
    ):
        logger.info("TEST 3: PASSED - Multiple students marked successfully. Total today: %d", att_mgr.get_today_count())
        test_results.append(("Multiple Distinct Students", True))
    else:
        logger.error("TEST 3: FAILED - Count mismatch or insert failure.")
        test_results.append(("Multiple Distinct Students", False))

    # ===================================================================
    # TEST 4: App restart simulation (Cold cache, DB has records)
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: App Restart Simulation (Cold Cache)")
    logger.info("=" * 60)

    # Create a fresh AttendanceManager sharing the SAME database
    new_att_mgr = AttendanceManager(db_manager=mock_db)
    res_restart = new_att_mgr.mark_present("22A01", "Sashi")

    if res_restart == AttendanceStatus.ALREADY_PRESENT and new_att_mgr.is_marked_today("22A01"):
        logger.info("TEST 4: PASSED - Database unique constraint prevented duplicate after app restart.")
        test_results.append(("Cold Cache Restart Deduplication", True))
    else:
        logger.error("TEST 4: FAILED - Duplicate allowed after restart: %s", res_restart)
        test_results.append(("Cold Cache Restart Deduplication", False))

    # ===================================================================
    # TEST 5: Status text formatting for UI overlay
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Status Text Formatting for UI Overlay")
    logger.info("=" * 60)

    text_present = att_mgr.get_status_text(AttendanceStatus.NEWLY_MARKED, "Sashi")
    text_already = att_mgr.get_status_text(AttendanceStatus.ALREADY_PRESENT, "Sashi")

    if text_present == "PRESENT - Sashi" and text_already == "ALREADY PRESENT - Sashi":
        logger.info("TEST 5: PASSED - Status overlay strings formatted correctly.")
        test_results.append(("Status Text Formatting", True))
    else:
        logger.error("TEST 5: FAILED - Incorrect status strings: '%s', '%s'", text_present, text_already)
        test_results.append(("Status Text Formatting", False))

    # ===================================================================
    # TEST 6: Date rollover simulation
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Date Rollover Cache Reset")
    logger.info("=" * 60)

    # Clear mock DB records for new day simulation
    mock_db.records.clear()
    att_mgr._cache_date = date.today() - timedelta(days=1)
    att_mgr._marked_today = {"22A01", "22A02"}

    # Next call should detect date change, clear cache, and allow marking for new day
    res_next_day = att_mgr.mark_present("22A01", "Sashi")

    if res_next_day == AttendanceStatus.NEWLY_MARKED and att_mgr.get_today_count() == 1:
        logger.info("TEST 6: PASSED - Date rollover reset cache and allowed marking for new day.")
        test_results.append(("Date Rollover Cache Reset", True))
    else:
        logger.error("TEST 6: FAILED - Date rollover did not reset cache properly: %s", res_next_day)
        test_results.append(("Date Rollover Cache Reset", False))

    # ===================================================================
    # FINAL REPORT
    # ===================================================================
    print("\n" + "=" * 60)
    print("ATTENDANCE MANAGER TEST REPORT")
    print("=" * 60)
    passed = sum(1 for _, r in test_results if r is True)
    failed = sum(1 for _, r in test_results if r is False)
    skipped = sum(1 for _, r in test_results if r is None)

    for name, r in test_results:
        status = "PASSED" if r is True else ("SKIPPED" if r is None else "FAILED")
        print(f"  [{status:7s}] {name}")
    print("=" * 60)
    print(f"  Total: {len(test_results)}  |  Passed: {passed}  |  Failed: {failed}  |  Skipped: {skipped}")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
