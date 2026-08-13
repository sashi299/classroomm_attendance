"""
test_timetable_current.py - Unit tests for current timetable period detection.

Tests the get_current_timetable_slot() method against the existing CSD-B
timetable records in the database. Uses injectable `now` parameter for
deterministic time-based assertions.

Tests:
  1.  Monday 09:20 → P1 MFCS (THEORY)
  2.  Tuesday 10:35 → P2 MFCS (THEORY)
  3.  Wednesday 12:30 → P4 Library (OTHER)
  4.  Tuesday 13:20 → LUNCH
  5.  Thursday 14:45 → P6 Programming in Python Lab / Database Systems Lab (LAB)
  6.  Sunday → NO_CLASS
  7.  Before 09:15 → BEFORE_CLASS
  8.  Invalid department → NO_CLASS
  9.  Invalid section → NO_CLASS
  10. Boundary: exactly 09:15 → P1
  11. Boundary: exactly 10:20 → P2
  12. Boundary: exactly 11:10 → P3
  13. Boundary: exactly 13:00 → LUNCH
  14. Boundary: exactly 13:40 → P5
  15. Boundary: exactly 14:30 → P6
  16. Boundary: exactly 15:20 → P7
"""

import os
import sys
import logging
import unittest
from datetime import datetime

# Add src/ to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from database import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_timetable_current")

# ── Reference dates (August 2026) ─────────────────────────────
# Aug 10 = Monday, Aug 11 = Tuesday, Aug 12 = Wednesday,
# Aug 13 = Thursday, Aug 14 = Friday, Aug 15 = Saturday, Aug 16 = Sunday
MON = lambda h, m: datetime(2026, 8, 10, h, m, 0)
TUE = lambda h, m: datetime(2026, 8, 11, h, m, 0)
WED = lambda h, m: datetime(2026, 8, 12, h, m, 0)
THU = lambda h, m: datetime(2026, 8, 13, h, m, 0)
FRI = lambda h, m: datetime(2026, 8, 14, h, m, 0)
SAT = lambda h, m: datetime(2026, 8, 15, h, m, 0)
SUN = lambda h, m: datetime(2026, 8, 16, h, m, 0)


class TestCurrentTimetableSlot(unittest.TestCase):
    """Test suite for current timetable period detection."""

    @classmethod
    def setUpClass(cls):
        """Create a shared DatabaseManager, connect, and ensure timetable is seeded."""
        cls.db = DatabaseManager(
            host=Config.MYSQL_HOST,
            port=Config.MYSQL_PORT,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DATABASE,
        )
        if not cls.db.connect():
            raise unittest.SkipTest("MySQL database not connected.")
        cls.db.seed_csd_b_timetable()

    @classmethod
    def tearDownClass(cls):
        cls.db.disconnect()

    def _slot(self, now, dept="CSD", sec="B"):
        """Helper to call get_current_timetable_slot."""
        return self.db.get_current_timetable_slot(department=dept, section=sec, now=now)

    # ── TEST 1: Monday 09:20 → P1 MFCS ──────────────────────────
    def test_01_monday_0920_p1_mfcs(self):
        result = self._slot(MON(9, 20))
        self.assertEqual(result["status"], "ACTIVE")
        self.assertEqual(result["day"], "MON")
        self.assertEqual(result["period_number"], 1)
        self.assertEqual(result["subject"], "MFCS")
        self.assertEqual(result["class_type"], "THEORY")
        logger.info("TEST 1 PASSED: Monday 09:20 → P1 MFCS (THEORY)")

    # ── TEST 2: Tuesday 10:35 → P2 MFCS ─────────────────────────
    def test_02_tuesday_1035_p2_mfcs(self):
        result = self._slot(TUE(10, 35))
        self.assertEqual(result["status"], "ACTIVE")
        self.assertEqual(result["day"], "TUE")
        self.assertEqual(result["period_number"], 2)
        self.assertEqual(result["subject"], "MFCS")
        self.assertEqual(result["class_type"], "THEORY")
        logger.info("TEST 2 PASSED: Tuesday 10:35 → P2 MFCS (THEORY)")

    # ── TEST 3: Wednesday 12:30 → P4 Library ─────────────────────
    def test_03_wednesday_1230_p4_library(self):
        result = self._slot(WED(12, 30))
        self.assertEqual(result["status"], "ACTIVE")
        self.assertEqual(result["day"], "WED")
        self.assertEqual(result["period_number"], 4)
        self.assertEqual(result["subject"], "Library")
        self.assertEqual(result["class_type"], "OTHER")
        logger.info("TEST 3 PASSED: Wednesday 12:30 → P4 Library (OTHER)")

    # ── TEST 4: Tuesday 13:20 → LUNCH ────────────────────────────
    def test_04_tuesday_1320_lunch(self):
        result = self._slot(TUE(13, 20))
        self.assertEqual(result["status"], "LUNCH")
        self.assertEqual(result["department"], "CSD")
        self.assertEqual(result["section"], "B")
        self.assertNotIn("subject", result)
        logger.info("TEST 4 PASSED: Tuesday 13:20 → LUNCH")

    # ── TEST 5: Thursday 14:45 → P6 Lab ──────────────────────────
    def test_05_thursday_1445_p6_lab(self):
        result = self._slot(THU(14, 45))
        self.assertEqual(result["status"], "ACTIVE")
        self.assertEqual(result["day"], "THU")
        self.assertEqual(result["period_number"], 6)
        self.assertEqual(result["subject"], "Programming in Python Lab / Database Systems Lab")
        self.assertEqual(result["class_type"], "LAB")
        logger.info("TEST 5 PASSED: Thursday 14:45 → P6 Lab")

    # ── TEST 6: Sunday → NO_CLASS ────────────────────────────────
    def test_06_sunday_no_class(self):
        result = self._slot(SUN(10, 0))
        self.assertEqual(result["status"], "NO_CLASS")
        self.assertEqual(result["department"], "CSD")
        self.assertEqual(result["section"], "B")
        logger.info("TEST 6 PASSED: Sunday → NO_CLASS")

    # ── TEST 7: Before 09:15 → BEFORE_CLASS ──────────────────────
    def test_07_before_0915_before_class(self):
        result = self._slot(MON(8, 0))
        self.assertEqual(result["status"], "BEFORE_CLASS")
        self.assertEqual(result["department"], "CSD")
        self.assertEqual(result["section"], "B")
        logger.info("TEST 7 PASSED: Monday 08:00 → BEFORE_CLASS")

    # ── TEST 8: Invalid department → NO_CLASS ────────────────────
    def test_08_invalid_department(self):
        result = self._slot(MON(10, 0), dept="XYZ")
        self.assertEqual(result["status"], "NO_CLASS")
        self.assertEqual(result["department"], "XYZ")
        logger.info("TEST 8 PASSED: Invalid department XYZ → NO_CLASS")

    # ── TEST 9: Invalid section → NO_CLASS ───────────────────────
    def test_09_invalid_section(self):
        result = self._slot(MON(10, 0), sec="Z")
        self.assertEqual(result["status"], "NO_CLASS")
        self.assertEqual(result["section"], "Z")
        logger.info("TEST 9 PASSED: Invalid section Z → NO_CLASS")

    # ── TEST 10: Boundary — exactly 09:15 → P1 ───────────────────
    def test_10_boundary_0915_p1(self):
        result = self._slot(MON(9, 15))
        self.assertEqual(result["status"], "ACTIVE")
        self.assertEqual(result["period_number"], 1)
        logger.info("TEST 10 PASSED: Boundary 09:15 → P1")

    # ── TEST 11: Boundary — exactly 10:20 → P2 ───────────────────
    def test_11_boundary_1020_p2(self):
        result = self._slot(MON(10, 20))
        self.assertEqual(result["status"], "ACTIVE")
        self.assertEqual(result["period_number"], 2)
        logger.info("TEST 11 PASSED: Boundary 10:20 → P2")

    # ── TEST 12: Boundary — exactly 11:10 → P3 ───────────────────
    def test_12_boundary_1110_p3(self):
        result = self._slot(MON(11, 10))
        self.assertEqual(result["status"], "ACTIVE")
        self.assertEqual(result["period_number"], 3)
        logger.info("TEST 12 PASSED: Boundary 11:10 → P3")

    # ── TEST 13: Boundary — exactly 13:00 → LUNCH ────────────────
    def test_13_boundary_1300_lunch(self):
        result = self._slot(MON(13, 0))
        self.assertEqual(result["status"], "LUNCH")
        logger.info("TEST 13 PASSED: Boundary 13:00 → LUNCH")

    # ── TEST 14: Boundary — exactly 13:40 → P5 ───────────────────
    def test_14_boundary_1340_p5(self):
        result = self._slot(MON(13, 40))
        self.assertEqual(result["status"], "ACTIVE")
        self.assertEqual(result["period_number"], 5)
        logger.info("TEST 14 PASSED: Boundary 13:40 → P5")

    # ── TEST 15: Boundary — exactly 14:30 → P6 ───────────────────
    def test_15_boundary_1430_p6(self):
        result = self._slot(MON(14, 30))
        self.assertEqual(result["status"], "ACTIVE")
        self.assertEqual(result["period_number"], 6)
        logger.info("TEST 15 PASSED: Boundary 14:30 → P6")

    # ── TEST 16: Boundary — exactly 15:20 → P7 ───────────────────
    def test_16_boundary_1520_p7(self):
        result = self._slot(MON(15, 20))
        self.assertEqual(result["status"], "ACTIVE")
        self.assertEqual(result["period_number"], 7)
        logger.info("TEST 16 PASSED: Boundary 15:20 → P7")

    # ── TEST 17: Late evening on class day → still P7 (open-ended)
    def test_17_late_evening_still_p7(self):
        result = self._slot(MON(18, 0))
        self.assertEqual(result["status"], "ACTIVE")
        self.assertEqual(result["period_number"], 7)
        logger.info("TEST 17 PASSED: Monday 18:00 → P7 (open-ended)")

    # ── TEST 18: Saturday 09:30 → P1 DBS ─────────────────────────
    def test_18_saturday_0930_p1_dbs(self):
        result = self._slot(SAT(9, 30))
        self.assertEqual(result["status"], "ACTIVE")
        self.assertEqual(result["day"], "SAT")
        self.assertEqual(result["period_number"], 1)
        self.assertEqual(result["subject"], "DBS")
        logger.info("TEST 18 PASSED: Saturday 09:30 → P1 DBS")


def run_tests():
    """Run current timetable slot detection test suite."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCurrentTimetableSlot)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
