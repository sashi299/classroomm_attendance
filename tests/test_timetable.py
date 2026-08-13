"""
test_timetable.py - Unit tests for CSD-B section timetable foundation.

Tests:
  1. CSD-B timetable insertion (42 records).
  2. No duplicate records on re-seeding.
  3. Monday retrieval (7 entries).
  4. Complete weekly retrieval (all 42 entries).
  5. Invalid department/section returns empty result.
"""

import os
import sys
import logging
import unittest

# Add src/ to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from database import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_timetable")

# Valid days for CSD-B timetable
VALID_DAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT"]
PERIODS_PER_DAY = 7
TOTAL_RECORDS = len(VALID_DAYS) * PERIODS_PER_DAY  # 42


class TestCsdbTimetable(unittest.TestCase):
    """Test suite for CSD-B section timetable foundation."""

    @classmethod
    def setUpClass(cls):
        """Create a shared DatabaseManager and connect once for all tests."""
        cls.db = DatabaseManager(
            host=Config.MYSQL_HOST,
            port=Config.MYSQL_PORT,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DATABASE,
        )
        if not cls.db.connect():
            raise unittest.SkipTest("MySQL database not connected.")

    @classmethod
    def tearDownClass(cls):
        """Disconnect after all tests."""
        cls.db.disconnect()

    # ── TEST 1: CSD-B timetable insertion ────────────────────────
    def test_01_csdb_timetable_insertion(self):
        """Seeding CSD-B timetable inserts exactly 42 records."""
        # First clear any existing CSD-B timetable entries to get a clean count
        cursor = None
        try:
            cursor = self.db._connection.cursor()
            cursor.execute(
                "DELETE FROM timetable WHERE department='CSD' AND section='B' AND academic_year='2026-27';"
            )
            self.db._connection.commit()
        finally:
            if cursor:
                cursor.close()

        inserted = self.db.seed_csd_b_timetable()
        self.assertEqual(inserted, TOTAL_RECORDS,
                         f"Expected {TOTAL_RECORDS} records inserted, got {inserted}")
        logger.info("TEST 1 PASSED: CSD-B timetable seeded with %d records.", inserted)

    # ── TEST 2: No duplicate records on re-seeding ───────────────
    def test_02_no_duplicate_records(self):
        """Re-seeding the same timetable inserts 0 new records (idempotent)."""
        # Ensure data exists first
        self.db.seed_csd_b_timetable()
        # Re-seed — should insert 0 due to UNIQUE constraint + INSERT IGNORE
        inserted = self.db.seed_csd_b_timetable()
        self.assertEqual(inserted, 0,
                         f"Expected 0 duplicate insertions, got {inserted}")

        # Verify total count is still exactly 42
        cursor = None
        try:
            cursor = self.db._connection.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM timetable WHERE department='CSD' AND section='B' AND academic_year='2026-27';"
            )
            count = cursor.fetchone()[0]
            self.assertEqual(count, TOTAL_RECORDS,
                             f"Expected {TOTAL_RECORDS} total records, got {count}")
        finally:
            if cursor:
                cursor.close()

        logger.info("TEST 2 PASSED: Re-seeding produced 0 duplicates, total remains %d.", TOTAL_RECORDS)

    # ── TEST 3: Monday retrieval ─────────────────────────────────
    def test_03_monday_retrieval(self):
        """Retrieving Monday timetable returns exactly 7 entries in correct order."""
        self.db.seed_csd_b_timetable()
        entries = self.db.get_timetable(department="CSD", section="B", day="MON")

        self.assertEqual(len(entries), PERIODS_PER_DAY,
                         f"Expected {PERIODS_PER_DAY} Monday entries, got {len(entries)}")

        # Verify period ordering
        period_numbers = [e["period_number"] for e in entries]
        self.assertEqual(period_numbers, list(range(1, PERIODS_PER_DAY + 1)),
                         f"Period numbers not in order: {period_numbers}")

        # Verify specific subjects
        self.assertEqual(entries[0]["subject"], "MFCS")
        self.assertEqual(entries[0]["class_type"], "THEORY")
        self.assertEqual(entries[4]["subject"], "Programming in Python Lab / Database Systems Lab")
        self.assertEqual(entries[4]["class_type"], "LAB")

        # Verify all entries are CSD-B
        for e in entries:
            self.assertEqual(e["department"], "CSD")
            self.assertEqual(e["section"], "B")
            self.assertEqual(e["day_of_week"], "MON")

        logger.info("TEST 3 PASSED: Monday retrieval returned %d correctly ordered entries.", len(entries))

    # ── TEST 4: Complete weekly retrieval ─────────────────────────
    def test_04_complete_weekly_retrieval(self):
        """Retrieving full week timetable returns all 42 entries across 6 days."""
        self.db.seed_csd_b_timetable()
        entries = self.db.get_timetable(department="CSD", section="B")

        self.assertEqual(len(entries), TOTAL_RECORDS,
                         f"Expected {TOTAL_RECORDS} weekly entries, got {len(entries)}")

        # Verify all 6 days are present
        days_found = set(e["day_of_week"] for e in entries)
        self.assertEqual(days_found, set(VALID_DAYS),
                         f"Missing days: {set(VALID_DAYS) - days_found}")

        # Verify each day has exactly 7 periods
        for day in VALID_DAYS:
            day_entries = [e for e in entries if e["day_of_week"] == day]
            self.assertEqual(len(day_entries), PERIODS_PER_DAY,
                             f"{day} has {len(day_entries)} entries, expected {PERIODS_PER_DAY}")

        logger.info("TEST 4 PASSED: Weekly retrieval returned all %d entries across %d days.",
                     TOTAL_RECORDS, len(VALID_DAYS))

    # ── TEST 5: Invalid department/section ───────────────────────
    def test_05_invalid_department_section(self):
        """Querying a non-existent department or section returns empty list."""
        self.db.seed_csd_b_timetable()

        # Invalid department
        entries = self.db.get_timetable(department="XYZ", section="B")
        self.assertEqual(len(entries), 0,
                         f"Expected 0 entries for department XYZ, got {len(entries)}")

        # Invalid section
        entries = self.db.get_timetable(department="CSD", section="Z")
        self.assertEqual(len(entries), 0,
                         f"Expected 0 entries for section Z, got {len(entries)}")

        # Both invalid
        entries = self.db.get_timetable(department="ABC", section="X")
        self.assertEqual(len(entries), 0,
                         f"Expected 0 entries for ABC/X, got {len(entries)}")

        logger.info("TEST 5 PASSED: Invalid department/section correctly returned empty results.")


def run_tests():
    """Run CSD-B timetable test suite."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCsdbTimetable)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
