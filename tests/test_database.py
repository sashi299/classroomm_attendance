"""
test_database.py - Test suite for DatabaseManager module.

Tests MySQL schema creation logic, table initialization,
connection handling, parameterized queries, and unique constraints.
"""

import os
import sys
import logging
from datetime import date, time as dt_time

# Add src/ to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from database import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_database")


def run_tests():
    """Run DatabaseManager unit tests."""
    test_results = []

    # ===================================================================
    # TEST 1: DatabaseManager initialization
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: DatabaseManager Initialization")
    logger.info("=" * 60)

    db = DatabaseManager(
        host="localhost",
        port=3306,
        user="test_user",
        password="dummy_password",
        database="classroom_test_db",
    )

    if not db.is_connected and db.database == "classroom_test_db":
        logger.info("TEST 1: PASSED - DatabaseManager initialized cleanly.")
        test_results.append(("DatabaseManager Initialization", True))
    else:
        logger.error("TEST 1: FAILED - Initialization state mismatch.")
        test_results.append(("DatabaseManager Initialization", False))

    # ===================================================================
    # TEST 2: SQL Statement definitions
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Schema & SQL Statements Validation")
    logger.info("=" * 60)

    has_unique = ("uq_student_date" in db.CREATE_ATTENDANCE_TABLE_SQL or "uq_student_date" in getattr(db, "CREATE_TABLE_SQL", ""))
    has_params = "%s" in db.INSERT_ATTENDANCE_SQL and "%s" in db.CHECK_ATTENDANCE_SQL

    if has_unique and has_params:
        logger.info("TEST 2: PASSED - UNIQUE constraint & parameterized SQL verified.")
        test_results.append(("Schema & SQL Validation", True))
    else:
        logger.error("TEST 2: FAILED - SQL statements missing unique key or parameter placeholders.")
        test_results.append(("Schema & SQL Validation", False))

    # ===================================================================
    # TEST 3: Connection with real MySQL (if credentials available in .env)
    # ===================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Real MySQL Connection & Table Operations")
    logger.info("=" * 60)

    config = Config()
    real_db = DatabaseManager(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        database="classroom_test_db",
    )

    connected = real_db.connect()

    if not connected:
        logger.info("TEST 3: SKIPPED - Could not connect to local MySQL (check credentials in .env).")
        test_results.append(("Real MySQL Connection & Table Ops", None))
    else:
        try:
            logger.info("Connected to MySQL. Testing schema creation & unique constraint...")
            test_date = date(2026, 8, 10)
            test_time = dt_time(10, 30, 0)

            # Insert first record
            res1 = real_db.insert_attendance(
                student_id="TEST_01",
                student_name="Test Student 1",
                attendance_date=test_date,
                attendance_time=test_time,
                status="Present",
                hourly_period="09:00-10:00",
                department="CSE",
            )

            # Check exists
            exists = real_db.check_attendance_exists("TEST_01", test_date, hourly_period="09:00-10:00", department="CSE")

            # Insert duplicate record (should be ignored due to UNIQUE constraint)
            res2 = real_db.insert_attendance(
                student_id="TEST_01",
                student_name="Test Student 1",
                attendance_date=test_date,
                attendance_time=test_time,
                status="Present",
                hourly_period="09:00-10:00",
                department="CSE",
            )

            if res1 and exists and not res2:
                logger.info("TEST 3: PASSED - Insert, check, and duplicate prevention verified.")
                test_results.append(("Real MySQL Connection & Table Ops", True))
            else:
                logger.error("TEST 3: FAILED - Duplicate record was not blocked properly.")
                test_results.append(("Real MySQL Connection & Table Ops", False))

        except Exception as e:
            logger.error("TEST 3: FAILED with error: %s", e)
            test_results.append(("Real MySQL Connection & Table Ops", False))
        finally:
            real_db.drop_table("attendance")
            real_db.disconnect()

    # ===================================================================
    # FINAL REPORT
    # ===================================================================
    print("\n" + "=" * 60)
    print("DATABASE MANAGER TEST REPORT")
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
