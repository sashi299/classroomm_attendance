"""
database.py - MySQL database operations.

Handles connection management, automatic database and schema migration,
thread-safe query execution, dynamic department & hourly period management,
and attendance record insertion with parameterized queries.

All database credentials are loaded from environment configuration.
No credentials are hardcoded in this module.
"""
import os
import math
import logging
import threading
from datetime import date, datetime, time as dt_time, timedelta
from typing import Optional, Tuple, List, Dict, Any

import mysql.connector
from mysql.connector import Error as MySQLError

# Module-level logger
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    MySQL database manager for the Classroom Attendance System.

    Handles:
      - Safe connection and automatic reconnection to MySQL server
      - Auto-creation and migration of database tables:
          - `attendance` (hourly period & department isolated)
          - `departments` (dynamic department registry & camera config)
          - `hourly_periods` (configurable timetable slots)
      - Thread-safe parameterized queries preventing SQL injection
      - Safe cursor lifecycle management (try/finally) preventing Commands out of sync
    """

    CREATE_ATTENDANCE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS attendance (
        id INT AUTO_INCREMENT PRIMARY KEY,
        student_id VARCHAR(50) NOT NULL,
        student_name VARCHAR(100) NOT NULL,
        department VARCHAR(20) NOT NULL DEFAULT 'CSD',
        attendance_date DATE NOT NULL,
        attendance_time TIME NOT NULL,
        hourly_period VARCHAR(30) NOT NULL DEFAULT '09:00-10:00',
        status VARCHAR(30) NOT NULL DEFAULT 'Present',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_student_date_period_dept (student_id, attendance_date, hourly_period, department)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    CREATE_DEPARTMENTS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS departments (
        id INT AUTO_INCREMENT PRIMARY KEY,
        code VARCHAR(20) NOT NULL UNIQUE,
        name VARCHAR(100) NOT NULL,
        is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
        camera_source VARCHAR(255) NOT NULL DEFAULT '',
        hod_contact VARCHAR(100) DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    CREATE_PERIODS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS hourly_periods (
        id INT AUTO_INCREMENT PRIMARY KEY,
        period_label VARCHAR(30) NOT NULL UNIQUE,
        start_time TIME NOT NULL,
        end_time TIME NOT NULL,
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    CREATE_TIMETABLE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS timetable (
        id INT AUTO_INCREMENT PRIMARY KEY,
        academic_year VARCHAR(20) NOT NULL,
        year_level VARCHAR(10) NOT NULL,
        semester VARCHAR(50) NOT NULL,
        department VARCHAR(20) NOT NULL,
        section VARCHAR(10) NOT NULL,
        day_of_week VARCHAR(10) NOT NULL,
        period_number INT NOT NULL,
        start_time TIME NOT NULL,
        end_time TIME NOT NULL,
        subject VARCHAR(100) NOT NULL,
        class_type ENUM('THEORY','LAB','OTHER') NOT NULL DEFAULT 'THEORY',
        faculty_id VARCHAR(50) DEFAULT NULL,
        faculty_name VARCHAR(100) DEFAULT '',
        faculty_contact VARCHAR(100) DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_timetable_slot (department, section, academic_year, year_level, semester, day_of_week, period_number)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    CREATE_NOTIFICATIONS_LOG_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS notifications_log (
        id INT AUTO_INCREMENT PRIMARY KEY,
        department VARCHAR(20) NOT NULL,
        section VARCHAR(10) NOT NULL,
        attendance_date DATE NOT NULL,
        period_number INT NOT NULL,
        recipient_role VARCHAR(20) NOT NULL,
        recipient_contact VARCHAR(100) NOT NULL,
        status VARCHAR(20) NOT NULL,
        sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_period_notif (department, section, attendance_date, period_number, recipient_role)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    CREATE_FACULTY_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS faculty (
        id INT AUTO_INCREMENT PRIMARY KEY,
        faculty_id VARCHAR(50) NOT NULL UNIQUE,
        name VARCHAR(100) NOT NULL,
        department VARCHAR(20) NOT NULL,
        phone VARCHAR(20),
        email VARCHAR(100),
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    CREATE_HOLIDAYS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS holidays (
        id INT AUTO_INCREMENT PRIMARY KEY,
        holiday_date DATE NOT NULL UNIQUE,
        description VARCHAR(255),
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    CREATE_CAMERAS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS cameras (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        department VARCHAR(20) NOT NULL,
        section VARCHAR(10) NOT NULL,
        classroom VARCHAR(50),
        source VARCHAR(255) NOT NULL,
        username VARCHAR(100),
        password VARCHAR(100),
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_dept_sec (department, section)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    CREATE_SEMESTERS_CONFIG_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS semesters_config (
        id INT AUTO_INCREMENT PRIMARY KEY,
        academic_year VARCHAR(20) NOT NULL,
        year_level VARCHAR(10) NOT NULL,
        semester VARCHAR(10) NOT NULL,
        start_date DATE NOT NULL,
        end_date DATE NOT NULL,
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_semester (academic_year, year_level, semester)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    CREATE_STUDENTS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS students (
        id INT AUTO_INCREMENT PRIMARY KEY,
        student_id VARCHAR(50) NOT NULL UNIQUE,
        name VARCHAR(100) NOT NULL,
        department VARCHAR(20) NOT NULL,
        year_level VARCHAR(10),
        section VARCHAR(10),
        academic_year VARCHAR(20),
        semester VARCHAR(10),
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_dept_sec (department, section)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    CREATE_ATTENDANCE_EVIDENCE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS attendance_evidence (
        id INT AUTO_INCREMENT PRIMARY KEY,
        attendance_id INT DEFAULT NULL,
        student_id VARCHAR(50) NOT NULL,
        department VARCHAR(20) NOT NULL,
        section VARCHAR(10) NOT NULL,
        attendance_date DATE NOT NULL,
        period_number INT NOT NULL,
        image_path VARCHAR(255) NOT NULL,
        quality_score FLOAT NOT NULL DEFAULT 0.0,
        captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_period_student_evidence (department, section, attendance_date, period_number, student_id),
        INDEX idx_student_date (student_id, attendance_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    CREATE_SYSTEM_SETTINGS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS system_settings (
        setting_key VARCHAR(100) PRIMARY KEY,
        setting_value TEXT NOT NULL,
        description VARCHAR(255),
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    CREATE_NOTIFICATION_SETTINGS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS notification_settings (
        id INT AUTO_INCREMENT PRIMARY KEY,
        setting_key VARCHAR(100) UNIQUE NOT NULL,
        setting_value TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    CREATE_ROLES_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS roles (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(50) UNIQUE NOT NULL,
        description VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    CREATE_SUBJECTS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS subjects (
        id INT AUTO_INCREMENT PRIMARY KEY,
        code VARCHAR(50) UNIQUE NOT NULL,
        name VARCHAR(100) NOT NULL,
        department VARCHAR(20) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    CREATE_ACADEMIC_YEARS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS academic_years (
        id INT AUTO_INCREMENT PRIMARY KEY,
        year_code VARCHAR(20) UNIQUE NOT NULL,
        is_current BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    DEFAULT_DEPARTMENTS = [
        ("CSD", "CSD", True, "0", "hod.csd@example.com"),
    ]

    DEFAULT_HOURLY_PERIODS = [
        ("09:00-10:00", "09:00:00", "10:00:00"),
        ("10:00-11:00", "10:00:00", "11:00:00"),
        ("11:00-12:00", "11:00:00", "12:00:00"),
        ("12:00-13:00", "12:00:00", "13:00:00"),
        ("13:00-14:00", "13:00:00", "14:00:00"),
        ("14:00-15:00", "14:00:00", "15:00:00"),
        ("15:00-16:00", "15:00:00", "16:00:00"),
        ("16:00-17:00", "16:00:00", "17:00:00"),
    ]

    INSERT_ATTENDANCE_SQL = """
    INSERT IGNORE INTO attendance
        (student_id, student_name, department, attendance_date, attendance_time,
         hourly_period, status, section, period_number, subject, class_type)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """

    CHECK_ATTENDANCE_SQL = """
    SELECT id FROM attendance
    WHERE student_id = %s AND attendance_date = %s AND hourly_period = %s AND department = %s
    LIMIT 1;
    """

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self._connection: Optional[mysql.connector.MySQLConnection] = None
        self._lock = threading.RLock()

        logger.info("DatabaseManager initialized.")
        logger.info("  Host: %s:%d", self.host, self.port)
        logger.info("  User: %s", self.user)
        logger.info("  Database: %s", self.database)

    def connect(self) -> bool:
        """Alias for _connect_unlocked wrapped with RLock."""
        with self._lock:
            return self._connect_unlocked()

    def _connect_unlocked(self) -> bool:
        """
        Connect to MySQL server, create database if not exists, and run schema setup.
        Thread-unsafe internal call. Must be called under self._lock.
        """
        try:
            if self._connection is None or not self._connection.is_connected():
                self._connection = mysql.connector.connect(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    autocommit=True
                )
                cursor = self._connection.cursor()
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{self.database}` "
                    f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
                )
                cursor.close()
                self._connection.database = self.database

            if not self._connection.is_connected():
                logger.error("MySQL connection failed (not connected).")
                return False

            try:
                server_info = getattr(self._connection, 'server_info', None) or self._connection.get_server_info()
            except Exception:
                server_info = "MySQL Server"

            logger.info("Connected to MySQL Server version: %s (database: %s)", server_info, self.database)

            cursor = None
            try:
                cursor = self._connection.cursor()
                # Ensure tables
                cursor.execute(self.CREATE_DEPARTMENTS_TABLE_SQL)
                cursor.execute(self.CREATE_PERIODS_TABLE_SQL)
                cursor.execute(self.CREATE_ATTENDANCE_TABLE_SQL)
                cursor.execute(self.CREATE_TIMETABLE_TABLE_SQL)
                cursor.execute(self.CREATE_NOTIFICATIONS_LOG_TABLE_SQL)
                cursor.execute(self.CREATE_SEMESTERS_CONFIG_TABLE_SQL)
                cursor.execute(self.CREATE_FACULTY_TABLE_SQL)
                cursor.execute(self.CREATE_HOLIDAYS_TABLE_SQL)
                cursor.execute(self.CREATE_CAMERAS_TABLE_SQL)
                cursor.execute(self.CREATE_STUDENTS_TABLE_SQL)
                cursor.execute(self.CREATE_ATTENDANCE_EVIDENCE_TABLE_SQL)
                cursor.execute(self.CREATE_SYSTEM_SETTINGS_TABLE_SQL)
                cursor.execute(self.CREATE_NOTIFICATION_SETTINGS_TABLE_SQL)
                cursor.execute(self.CREATE_ROLES_TABLE_SQL)
                cursor.execute(self.CREATE_SUBJECTS_TABLE_SQL)
                cursor.execute(self.CREATE_ACADEMIC_YEARS_TABLE_SQL)
                self._connection.commit()

                # Run database migrations FIRST
                self._migrate_attendance_table_unlocked(cursor)
                self._migrate_notification_columns_unlocked(cursor)

                # Ensure CSD is present and enabled; disable any other departments safely for CCTV trial
                logger.info("Ensuring CSD as the only active department for CCTV trial...")
                cursor.execute(
                    "INSERT INTO departments (code, name, is_enabled, camera_source, hod_contact) "
                    "VALUES ('CSD', 'CSD', TRUE, '0', 'hod.csd@example.com') "
                    "ON DUPLICATE KEY UPDATE is_enabled = TRUE, name = 'CSD', hod_contact = 'hod.csd@example.com';"
                )
                cursor.execute("UPDATE departments SET is_enabled = FALSE WHERE code != 'CSD';")
                self._connection.commit()

                # Seed default hourly periods if table empty
                cursor.execute("SELECT COUNT(*) FROM hourly_periods;")
                count_p_row = cursor.fetchone()
                if count_p_row and count_p_row[0] == 0:
                    logger.info("Seeding default hourly timetable periods...")
                    for label, stime, etime in self.DEFAULT_HOURLY_PERIODS:
                        cursor.execute(
                            "INSERT IGNORE INTO hourly_periods (period_label, start_time, end_time, is_active) "
                            "VALUES (%s, %s, %s, %s);",
                            (label, stime, etime, True)
                        )
                    self._connection.commit()

                # Seed CSD-B timetable trial data
                self.seed_csd_b_timetable()

                # Seed default semester config if empty
                cursor.execute("SELECT COUNT(*) FROM semesters_config;")
                row_count = cursor.fetchone()
                if row_count and row_count[0] == 0:
                    logger.info("Seeding default semester configuration...")
                    cursor.execute(
                        "INSERT INTO semesters_config (academic_year, year_level, semester, start_date, end_date, is_active) "
                        "VALUES ('2026-27', 'II', 'I', '2026-06-01', '2026-12-31', TRUE);"
                    )
                    self._connection.commit()

                logger.info("Database schema and migrations ensured successfully.")
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass

            return True

        except MySQLError as e:
            logger.error("MySQL connection error: %s", e)
            self._connection = None
            return False

    def _migrate_attendance_table_unlocked(self, cursor):
        """Safely migrate legacy attendance table to include hourly_period & department columns."""
        try:
            # Check existing columns
            cursor.execute("SHOW COLUMNS FROM attendance LIKE 'hourly_period';")
            has_hourly = cursor.fetchone() is not None

            cursor.execute("SHOW COLUMNS FROM attendance LIKE 'department';")
            has_dept = cursor.fetchone() is not None

            if not has_hourly:
                logger.info("Migrating schema: Adding 'hourly_period' column to attendance table...")
                cursor.execute(
                    "ALTER TABLE attendance ADD COLUMN hourly_period VARCHAR(30) NOT NULL DEFAULT '09:00-10:00';"
                )

            if not has_dept:
                logger.info("Migrating schema: Adding 'department' column to attendance table...")
                cursor.execute(
                    "ALTER TABLE attendance ADD COLUMN department VARCHAR(20) NOT NULL DEFAULT 'CSE';"
                )

            if not has_hourly or not has_dept:
                # Update existing records to populate hourly_period based on attendance_time
                logger.info("Populating hourly_period for existing attendance records...")
                cursor.execute(
                    "UPDATE attendance SET hourly_period = "
                    "CONCAT(LPAD(HOUR(attendance_time), 2, '0'), ':00-', "
                    "LPAD(HOUR(attendance_time) + 1, 2, '0'), ':00') "
                    "WHERE hourly_period = '09:00-10:00' OR hourly_period IS NULL;"
                )

                # Drop legacy unique index if present and add new hourly unique constraint
                try:
                    cursor.execute("SHOW INDEX FROM attendance WHERE Key_name = 'uq_student_date';")
                    rows_old = cursor.fetchall()
                    if rows_old:
                        cursor.execute("ALTER TABLE attendance DROP INDEX uq_student_date;")
                except Exception as e:
                    logger.debug("Note dropping uq_student_date index: %s", e)

                try:
                    cursor.execute("SHOW INDEX FROM attendance WHERE Key_name = 'uq_student_date_period_dept';")
                    rows_new = cursor.fetchall()
                    if not rows_new:
                        cursor.execute(
                            "ALTER TABLE attendance ADD UNIQUE KEY uq_student_date_period_dept "
                            "(student_id, attendance_date, hourly_period, department);"
                        )
                except Exception as e:
                    logger.debug("Note creating uq_student_date_period_dept index: %s", e)

                self._connection.commit()
                logger.info("Attendance schema migration complete.")

        except Exception as e:
            logger.warning("Attendance migration notice: %s", e)

    def _migrate_notification_columns_unlocked(self, cursor):
        """Safely add notification-related columns to departments and timetable tables."""
        try:
            # 1. departments table
            cursor.execute("SHOW COLUMNS FROM departments LIKE 'hod_contact';")
            if not cursor.fetchone():
                logger.info("Migrating schema: Adding 'hod_contact' column to departments table...")
                cursor.execute("ALTER TABLE departments ADD COLUMN hod_contact VARCHAR(100) DEFAULT '';")

            # 2. timetable table
            cursor.execute("SHOW COLUMNS FROM timetable LIKE 'faculty_name';")
            if not cursor.fetchone():
                logger.info("Migrating schema: Adding 'faculty_name' column to timetable table...")
                cursor.execute("ALTER TABLE timetable ADD COLUMN faculty_name VARCHAR(100) DEFAULT '';")

            cursor.execute("SHOW COLUMNS FROM timetable LIKE 'faculty_contact';")
            if not cursor.fetchone():
                logger.info("Migrating schema: Adding 'faculty_contact' column to timetable table...")
                cursor.execute("ALTER TABLE timetable ADD COLUMN faculty_contact VARCHAR(100) DEFAULT '';")

            self._connection.commit()

            # Academic filtering columns for timetable
            cursor.execute("SHOW COLUMNS FROM timetable LIKE 'year_level';")
            if not cursor.fetchone():
                logger.info("Migrating schema: Adding 'year_level' column to timetable table...")
                # Add it after academic_year
                cursor.execute("ALTER TABLE timetable ADD COLUMN year_level VARCHAR(10) NOT NULL AFTER academic_year;")
                # Populate existing with 'II' for trial and clean up old trial labels
                cursor.execute("UPDATE timetable SET year_level = 'II' WHERE year_level IS NULL OR year_level = '';")
                cursor.execute("DELETE FROM timetable WHERE semester = 'II B.Tech I Sem' AND department = 'CSD' AND section = 'B';")

            # Update unique key to include semester and year_level
            try:
                cursor.execute("SHOW INDEX FROM timetable WHERE Key_name = 'uq_timetable_slot';")
                rows = cursor.fetchall()
                if rows:
                    # Check if it already includes year_level/semester
                    cols = [r['Column_name'] for r in rows] if isinstance(rows[0], dict) else [r[4] for r in rows]
                    if 'year_level' not in cols or 'semester' not in cols:
                        logger.info("Migrating schema: Updating uq_timetable_slot index...")
                        cursor.execute("ALTER TABLE timetable DROP INDEX uq_timetable_slot;")
                        cursor.execute(
                            "ALTER TABLE timetable ADD UNIQUE KEY uq_timetable_slot "
                            "(department, section, academic_year, year_level, semester, day_of_week, period_number);"
                        )
            except Exception as e:
                logger.debug("Note updating timetable index: %s", e)

            self._connection.commit()
        except MySQLError as e:
            logger.warning("Migration warning (notification/academic columns): %s", e)

        # Faculty link for timetable
        try:
            cursor.execute("SHOW COLUMNS FROM timetable LIKE 'faculty_id';")
            if not cursor.fetchone():
                logger.info("Migrating schema: Adding 'faculty_id' column to timetable table...")
                cursor.execute("ALTER TABLE timetable ADD COLUMN faculty_id VARCHAR(50) DEFAULT NULL AFTER class_type;")
            self._connection.commit()
        except MySQLError as e:
            logger.warning("Migration warning (faculty_id column): %s", e)

        # Timetable-aware attendance columns migration
        try:
            timetable_columns = [
                ("section", "VARCHAR(10) DEFAULT NULL"),
                ("period_number", "INT DEFAULT NULL"),
                ("subject", "VARCHAR(100) DEFAULT NULL"),
                ("class_type", "VARCHAR(20) DEFAULT NULL"),
            ]
            for col_name, col_def in timetable_columns:
                cursor.execute(f"SHOW COLUMNS FROM attendance LIKE '{col_name}';")
                if cursor.fetchone() is None:
                    logger.info("Migrating schema: Adding '%s' column to attendance table...", col_name)
                    cursor.execute(f"ALTER TABLE attendance ADD COLUMN {col_name} {col_def};")
            self._connection.commit()
        except Exception as e:
            logger.warning("Timetable attendance migration notice: %s", e)

        self.ensure_reporting_indexes()

    def ensure_reporting_indexes(self):
        """Ensure all database performance indexes required for reporting exist."""
        with self._lock:
            if self._connection is None or not self._connection.is_connected():
                return
            cursor = None
            try:
                cursor = self._connection.cursor()
                indexes_to_create = [
                    ("idx_attendance_date", "CREATE INDEX idx_attendance_date ON attendance(attendance_date);"),
                    ("idx_department", "CREATE INDEX idx_department ON attendance(department);"),
                    ("idx_student_id", "CREATE INDEX idx_student_id ON attendance(student_id);"),
                    ("idx_hourly_period", "CREATE INDEX idx_hourly_period ON attendance(hourly_period);"),
                    ("idx_date_dept", "CREATE INDEX idx_date_dept ON attendance(attendance_date, department);"),
                ]
                for idx_name, idx_sql in indexes_to_create:
                    try:
                        cursor.execute(f"SHOW INDEX FROM attendance WHERE Key_name = '{idx_name}';")
                        rows = cursor.fetchall()
                        if not rows:
                            cursor.execute(idx_sql)
                            logger.info("Created reporting index: %s", idx_name)
                    except Exception as ie:
                        logger.warning("Index creation notice (%s): %s", idx_name, ie)
                self._connection.commit()
            except Exception as e:
                logger.warning("Error ensuring reporting indexes: %s", e)
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass

    def disconnect(self):
        """Close the MySQL connection safely."""
        with self._lock:
            if self._connection is not None:
                try:
                    if self._connection.is_connected():
                        self._connection.close()
                        logger.info("MySQL connection closed.")
                except MySQLError as e:
                    logger.warning("Error closing MySQL connection: %s", e)
                finally:
                    self._connection = None

    @property
    def is_connected(self) -> bool:
        """Check if the MySQL connection is alive."""
        with self._lock:
            if self._connection is None:
                return False
            try:
                return self._connection.is_connected()
            except Exception:
                return False

    def _ensure_connection_unlocked(self) -> bool:
        """Ensure database connection is alive and clean (caller must hold _lock)."""
        if self._connection is not None:
            try:
                if self._connection.is_connected():
                    if getattr(self._connection, "unread_result", False):
                        try:
                            self._connection.consume_results()
                        except Exception:
                            pass
                    return True
            except Exception:
                pass

        logger.warning("MySQL connection lost or uninitialized. Attempting to reconnect...")
        return self._connect_internal()

    # ── Dynamic Department CRUD Methods ────────────────────────────

    def get_departments(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """Return list of registered departments."""
        with self._lock:
            if not self._ensure_connection_unlocked():
                return []
            cursor = None
            try:
                sql = "SELECT id, code, name, is_enabled, camera_source, hod_contact, created_at FROM departments"
                if enabled_only:
                    sql += " WHERE is_enabled = TRUE"
                sql += " ORDER BY code ASC;"

                cursor = self._connection.cursor(dictionary=True)
                cursor.execute(sql)
                rows = cursor.fetchall()
                return rows if rows is not None else []
            except MySQLError as e:
                logger.error("Error fetching departments: %s", e)
                return []
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass

    def get_department_by_code(self, dept_code: str) -> Optional[Dict[str, Any]]:
        """Fetch details for a single department code."""
        code = (dept_code or "").strip().upper()
        with self._lock:
            if not self._ensure_connection_unlocked():
                return None
            cursor = None
            try:
                cursor = self._connection.cursor(dictionary=True)
                cursor.execute(
                    "SELECT id, code, name, is_enabled, camera_source, created_at "
                    "FROM departments WHERE code = %s LIMIT 1;",
                    (code,)
                )
                return cursor.fetchone()
            except MySQLError as e:
                logger.error("Error getting department '%s': %s", code, e)
                return None
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass

    def add_department(
        self,
        code: str,
        name: str,
        camera_source: str = "",
        is_enabled: bool = True,
    ) -> Tuple[bool, str]:
        """Add a new department dynamically."""
        clean_code = (code or "").strip().upper()
        clean_name = (name or "").strip()

        if not clean_code or not clean_code.isalnum():
            return False, "Department code must be alphanumeric (e.g. CSE, AIML)."
        if not clean_name:
            return False, "Department name cannot be empty."

        with self._lock:
            if not self._ensure_connection_unlocked():
                return False, "No database connection."
            cursor = None
            try:
                cursor = self._connection.cursor()
                cursor.execute(
                    "INSERT INTO departments (code, name, camera_source, is_enabled) "
                    "VALUES (%s, %s, %s, %s);",
                    (clean_code, clean_name, camera_source.strip(), is_enabled),
                )
                self._connection.commit()
                logger.info("Added department: [%s] %s", clean_code, clean_name)
                return True, f"Department '{clean_code}' created successfully."
            except MySQLError as e:
                logger.error("Error adding department '%s': %s", clean_code, e)
                if e.errno == 1062:
                    return False, f"Department code '{clean_code}' already exists."
                return False, f"Database error: {e}"
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass

    def update_department(
        self,
        code: str,
        name: Optional[str] = None,
        camera_source: Optional[str] = None,
        is_enabled: Optional[bool] = None,
    ) -> Tuple[bool, str]:
        """Update an existing department's name, camera source, or enabled state."""
        clean_code = (code or "").strip().upper()
        with self._lock:
            if not self._ensure_connection_unlocked():
                return False, "No database connection."

            cursor = None
            try:
                cursor = self._connection.cursor()
                updates = []
                params = []

                if name is not None:
                    updates.append("name = %s")
                    params.append(name.strip())

                if camera_source is not None:
                    updates.append("camera_source = %s")
                    params.append(camera_source.strip())

                if is_enabled is not None:
                    updates.append("is_enabled = %s")
                    params.append(is_enabled)

                if not updates:
                    return True, "No fields to update."

                params.append(clean_code)
                sql = f"UPDATE departments SET {', '.join(updates)} WHERE code = %s;"
                cursor.execute(sql, tuple(params))
                self._connection.commit()

                if cursor.rowcount > 0:
                    logger.info("Updated department [%s]", clean_code)
                    return True, f"Department '{clean_code}' updated successfully."
                else:
                    return False, f"Department '{clean_code}' not found."

            except MySQLError as e:
                logger.error("Error updating department '%s': %s", clean_code, e)
                return False, f"Database error: {e}"
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass

    def toggle_department(self, code: str, is_enabled: bool) -> Tuple[bool, str]:
        """Enable or disable a department."""
        return self.update_department(code=code, is_enabled=is_enabled)

    # ── Hourly Timetable Period Methods ────────────────────────────

    def get_hourly_periods(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Return list of configured hourly timetable periods."""
        with self._lock:
            if not self._ensure_connection_unlocked():
                return []
            cursor = None
            try:
                sql = "SELECT id, period_label, start_time, end_time, is_active FROM hourly_periods"
                if active_only:
                    sql += " WHERE is_active = TRUE"
                sql += " ORDER BY start_time ASC;"

                cursor = self._connection.cursor(dictionary=True)
                cursor.execute(sql)
                rows = cursor.fetchall()
                formatted = []
                for r in (rows or []):
                    formatted.append({
                        "id": r["id"],
                        "period_label": r["period_label"],
                        "start_time": str(r["start_time"]),
                        "end_time": str(r["end_time"]),
                        "is_active": bool(r["is_active"]),
                    })
                return formatted
            except MySQLError as e:
                logger.error("Error fetching hourly periods: %s", e)
                return []
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass

    def add_hourly_period(
        self,
        period_label: str,
        start_time: str,
        end_time: str,
        is_active: bool = True,
    ) -> Tuple[bool, str]:
        """Add a new hourly timetable period."""
        clean_label = (period_label or "").strip()
        with self._lock:
            if not self._ensure_connection_unlocked():
                return False, "No database connection."
            cursor = None
            try:
                cursor = self._connection.cursor()
                cursor.execute(
                    "INSERT INTO hourly_periods (period_label, start_time, end_time, is_active) "
                    "VALUES (%s, %s, %s, %s);",
                    (clean_label, start_time, end_time, is_active)
                )
                self._connection.commit()
                return True, f"Period '{clean_label}' created successfully."
            except MySQLError as e:
                logger.error("Error adding period '%s': %s", clean_label, e)
                return False, f"Database error: {e}"
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass

    def get_current_hourly_period(self, time_val: Optional[dt_time] = None) -> str:
        """
        Determine the active hourly timetable period label for a given time.
        If no configured slot matches, formats dynamically as HH:00-(HH+1):00.
        """
        if time_val is None:
            time_val = datetime.now().time()

        periods = self.get_hourly_periods(active_only=True)
        t_sec = time_val.hour * 3600 + time_val.minute * 60 + time_val.second

        for p in periods:
            try:
                # Parse start/end times
                sh, sm, ss = map(int, p["start_time"].split(":"))
                eh, em, es = map(int, p["end_time"].split(":"))
                s_sec = sh * 3600 + sm * 60 + ss
                e_sec = eh * 3600 + em * 60 + es

                if s_sec <= t_sec < e_sec:
                    return p["period_label"]
            except Exception:
                continue

        # Dynamic fallback: HH:00 - (HH+1):00
        cur_hour = time_val.hour
        next_hour = (cur_hour + 1) % 24
        return f"{cur_hour:02d}:00-{next_hour:02d}:00"

    # ── Attendance Query & Insertion Methods ───────────────────────

    def check_attendance_exists(
        self,
        student_id: str,
        attendance_date: date,
        hourly_period: Optional[str] = None,
        department: str = "CSD",
    ) -> bool:
        """Check if a student already has an attendance record for a date, period, and department."""
        if not hourly_period:
            hourly_period = self.get_current_hourly_period()
        with self._lock:
            if not self._ensure_connection_unlocked():
                logger.error("Cannot check attendance: no database connection.")
                return False

            cursor = None
            try:
                cursor = self._connection.cursor()
                cursor.execute(
                    self.CHECK_ATTENDANCE_SQL,
                    (student_id, attendance_date, hourly_period, department)
                )
                result = cursor.fetchone()
                return result is not None
            except MySQLError as e:
                logger.error("Error checking attendance: %s", e)
                if "Commands out of sync" in str(e) or e.errno == 2014:
                    self._connect_internal()
                return False
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass

    def insert_attendance(
        self,
        student_id: str,
        student_name: str,
        attendance_date: date,
        attendance_time: dt_time,
        status: str = "Present",
        hourly_period: Optional[str] = None,
        department: str = "CSD",
        section: Optional[str] = None,
        period_number: Optional[int] = None,
        subject: Optional[str] = None,
        class_type: Optional[str] = None,
    ) -> bool:
        """Insert an attendance record using INSERT IGNORE.

        Supports optional timetable-aware fields (section, period_number, subject, class_type)
        for CSD-B trial. Historical records without these fields remain valid.
        """
        if not hourly_period:
            hourly_period = self.get_current_hourly_period(attendance_time)

        dept_code = (department or "CSD").strip().upper()

        with self._lock:
            if not self._ensure_connection_unlocked():
                logger.error("Cannot insert attendance: no database connection.")
                return False

            cursor = None
            try:
                cursor = self._connection.cursor()
                cursor.execute(
                    self.INSERT_ATTENDANCE_SQL,
                    (student_id, student_name, dept_code, attendance_date, attendance_time,
                     hourly_period, status, section, period_number, subject, class_type),
                )
                self._connection.commit()
                rows_affected = cursor.rowcount

                if rows_affected > 0:
                    logger.info(
                        "ATTENDANCE MARKED: [%s] %s (%s) for period %s on %s at %s",
                        student_id, student_name, dept_code, hourly_period, attendance_date, attendance_time,
                    )
                    return True
                else:
                    logger.debug(
                        "Attendance already exists: [%s] %s (%s) period %s on %s",
                        student_id, student_name, dept_code, hourly_period, attendance_date,
                    )
                    return False

            except MySQLError as e:
                logger.error("Error inserting attendance: %s", e)
                if "Commands out of sync" in str(e) or e.errno == 2014:
                    self._connect_internal()
                return False
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass

    def get_today_attendance(
        self,
        today: Optional[date] = None,
        dept_code: Optional[str] = None,
        period: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve all attendance records for a given date, optionally filtered by dept & period."""
        if today is None:
            today = date.today()

        with self._lock:
            if not self._ensure_connection_unlocked():
                logger.error("Cannot query attendance: no database connection.")
                return []

            cursor = None
            try:
                sql = (
                    "SELECT student_id, student_name, department, attendance_date, attendance_time, "
                    "hourly_period, status, section, period_number, subject, class_type "
                    "FROM attendance WHERE attendance_date = %s"
                )
                params = [today]

                if dept_code and dept_code != "ALL":
                    sql += " AND department = %s"
                    params.append(dept_code.strip().upper())

                if period and period != "ALL":
                    if str(period).isdigit():
                        sql += " AND (period_number = %s OR hourly_period = %s)"
                        params.extend([int(period), str(period).strip()])
                    else:
                        sql += " AND hourly_period = %s"
                        params.append(str(period).strip())

                sql += " ORDER BY attendance_time DESC, id DESC;"

                cursor = self._connection.cursor(dictionary=True)
                cursor.execute(sql, tuple(params))
                rows = cursor.fetchall()
                return rows if rows is not None else []
            except MySQLError as e:
                logger.error("Error querying attendance: %s", e)
                if "Commands out of sync" in str(e) or e.errno == 2014:
                    self._connect_internal()
                return []
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass

    def get_attendance_history(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        search: Optional[str] = None,
        dept: Optional[str] = None,
        period: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve historical attendance records with parameterized multi-filter support."""
        with self._lock:
            if not self._ensure_connection_unlocked():
                logger.error("Cannot query attendance history: no database connection.")
                return []

            cursor = None
            try:
                sql = (
                    "SELECT student_id, student_name, department, attendance_date, attendance_time, "
                    "hourly_period, status, section, period_number, subject, class_type "
                    "FROM attendance WHERE 1=1"
                )
                params = []

                if start_date is not None:
                    sql += " AND attendance_date >= %s"
                    params.append(start_date)

                if end_date is not None:
                    sql += " AND attendance_date <= %s"
                    params.append(end_date)

                if search and search.strip():
                    clean_search = f"%{search.strip()}%"
                    sql += " AND (student_id LIKE %s OR student_name LIKE %s)"
                    params.extend([clean_search, clean_search])

                if dept and dept != "ALL":
                    sql += " AND department = %s"
                    params.append(dept.strip().upper())

                if period and period != "ALL":
                    sql += " AND hourly_period = %s"
                    params.append(period.strip())

                if status and status != "ALL":
                    sql += " AND status = %s"
                    params.append(status.strip())

                sql += " ORDER BY attendance_date DESC, attendance_time DESC;"

                cursor = self._connection.cursor(dictionary=True)
                cursor.execute(sql, tuple(params))
                rows = cursor.fetchall()
                return rows if rows is not None else []
            except MySQLError as e:
                logger.error("Error querying attendance history: %s", e)
                if "Commands out of sync" in str(e) or e.errno == 2014:
                    self._connect_internal()
                return []
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass

    def get_attendance_report_data(
        self,
        start_date: date,
        end_date: date,
        department: str = "CSD",
        section: str = "B",
        academic_year: Optional[str] = None,
        year_level: Optional[str] = None,
        semester: Optional[str] = None,
        registered_students: Optional[List[Dict[str, Any]]] = None,
        search: Optional[str] = None,
        status: Optional[str] = "ALL",
        hourly_period: Optional[str] = "ALL",
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
        page: int = 1,
        per_page: int = 50,
    ) -> Dict[str, Any]:
        """
        Generate timetable-aware attendance report with PRESENT/ABSENT calculation
        based on the scheduled periods for CSD-B trial.
        """
        if registered_students is None:
            registered_students = []

        # 1. Fetch timetable entries for the target context
        dept = (department or "CSD").strip().upper()
        sec = (section or "B").strip().upper()

        # 1a. Respect Semester Dates if context provided
        sem_start = None
        sem_end = None
        if academic_year and year_level and semester:
            sem_config = self.get_semester_by_params(academic_year, year_level, semester)
            if sem_config:
                sem_start = datetime.strptime(str(sem_config["start_date"]), "%Y-%m-%d").date()
                sem_end = datetime.strptime(str(sem_config["end_date"]), "%Y-%m-%d").date()

                # Clip report dates to semester range
                if start_date < sem_start: start_date = sem_start
                if end_date > sem_end: end_date = sem_end

                if start_date > end_date:
                    # Context provided but dates out of bounds
                    return self._empty_report_result(start_date, end_date, dept, sec, page, per_page)

        timetable = self.get_timetable(
            department=dept,
            section=sec,
            academic_year=academic_year,
            year_level=year_level,
            semester=semester
        )

        # Group timetable by day
        tt_by_day = {}
        for entry in timetable:
            day = entry["day_of_week"]
            if day not in tt_by_day:
                tt_by_day[day] = []
            tt_by_day[day].append(entry)

        # 2. Build date list (Excluding Sundays and Holidays)
        delta = (end_date - start_date).days
        date_list = []
        for i in range(delta + 1):
            curr_d = start_date + timedelta(days=i)
            if not self.is_holiday(curr_d):
                date_list.append(curr_d)

        # Parse time filters
        f_time = None
        t_time = None
        if from_time and from_time.strip():
            try:
                parts = from_time.strip().split(":")
                f_time = dt_time(int(parts[0]), int(parts[1]))
            except Exception: pass
        if to_time and to_time.strip():
            try:
                parts = to_time.strip().split(":")
                t_time = dt_time(int(parts[0]), int(parts[1]))
            except Exception: pass

        # 3. Query existing attendance records from database
        db_records = self.get_attendance_history(
            start_date=start_date,
            end_date=end_date,
            dept=dept,
        )

        # Lookup dict: (student_id.upper(), date_str, period_number) -> record
        present_map = {}
        for r in db_records:
            s_id = (r.get("student_id") or "").strip().upper()
            d_str = str(r.get("attendance_date"))
            # We prioritize period_number for matching if available
            p_num = r.get("period_number")
            if p_num is not None:
                try:
                    p_num = int(p_num)
                except (ValueError, TypeError):
                    pass
                key = (s_id, d_str, p_num)
                if key not in present_map:
                    present_map[key] = r

        # 4. Filter registered students
        filtered_students = []
        for st in registered_students:
            st_id = (st.get("student_id") or "").strip()
            st_name = (st.get("student_name") or "").strip()

            if search and search.strip():
                q = search.strip().lower()
                if q not in st_id.lower() and q not in st_name.lower():
                    continue
            filtered_students.append({"student_id": st_id, "student_name": st_name})

        # Fallback if no students provided
        if not filtered_students and db_records:
            seen = set()
            for r in db_records:
                st_id = r.get("student_id", "")
                st_name = r.get("student_name", "")
                if st_id.upper() not in seen:
                    seen.add(st_id.upper())
                    if search and search.strip():
                        q = search.strip().lower()
                        if q not in st_id.lower() and q not in st_name.lower():
                            continue
                    filtered_students.append({"student_id": st_id, "student_name": st_name})

        # 5. Build full report matrix based on timetable slots
        full_matrix = []
        clean_status_filter = (status or "ALL").strip().upper()

        for d in date_list:
            day_code = self._WEEKDAY_MAP.get(d.weekday(), "SUN")
            if day_code not in tt_by_day:
                continue # Skip Sunday or days without timetable

            slots = tt_by_day[day_code]
            for slot in slots:
                # Apply period filter if requested
                if hourly_period and hourly_period != "ALL":
                    try:
                        if int(slot["period_number"]) != int(hourly_period):
                            continue
                    except (ValueError, TypeError):
                        # Fallback to string comparison if not numeric
                        if str(slot["period_number"]) != str(hourly_period):
                            continue

                # Apply time filters
                slot_start = self._parse_time_safe(slot["start_time"])
                if f_time and slot_start < f_time:
                    continue
                if t_time and slot_start > t_time:
                    continue

                for st in filtered_students:
                    st_id = st["student_id"]
                    try:
                        p_num = int(slot["period_number"])
                    except (ValueError, TypeError):
                        p_num = slot["period_number"]
                    key = (st_id.upper(), str(d), p_num)

                    if key in present_map:
                        rec = present_map[key]
                        att_time = str(rec.get("attendance_time") or "")
                        rec_status = (rec.get("status") or "PRESENT").strip().upper()
                    else:
                        att_time = "-"
                        rec_status = "ABSENT"

                    full_matrix.append({
                        "attendance_date": str(d),
                        "attendance_time": att_time,
                        "period_number": slot["period_number"],
                        "hourly_period": f"{slot['start_time'][:5]}-{slot['end_time'][:5]}",
                        "subject": slot["subject"],
                        "class_type": slot["class_type"],
                        "student_id": st_id,
                        "student_name": st["student_name"],
                        "department": dept,
                        "section": sec,
                        "status": rec_status,
                    })

        # 6. Calculate Summary Metrics
        total_students = len(filtered_students)
        total_periods = len(full_matrix)
        present_periods = sum(1 for m in full_matrix if m["status"] == "PRESENT")
        absent_periods = sum(1 for m in full_matrix if m["status"] == "ABSENT")
        att_percentage = round((present_periods / total_periods) * 100, 2) if total_periods > 0 else 0.0

        students_present_set = {m["student_id"].upper() for m in full_matrix if m["status"] == "PRESENT"}
        present_students_count = len(students_present_set)
        absent_students_count = max(0, total_students - present_students_count)

        summary = {
            "total_students": total_students,
            "present_students": present_students_count,
            "absent_students": absent_students_count,
            "total_periods": total_periods,
            "present_periods": present_periods,
            "absent_periods": absent_periods,
            "attendance_percentage": att_percentage,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "department": dept,
            "section": sec,
            "hourly_period": hourly_period,
            "academic_year": academic_year,
            "year_level": year_level,
            "semester": semester,
            "from_time": from_time,
            "to_time": to_time,
        }

        # Filter matrix by status
        if clean_status_filter != "ALL":
            display_matrix = [m for m in full_matrix if m["status"] == clean_status_filter]
        else:
            display_matrix = full_matrix

        # 7. Apply Pagination
        page = max(1, page)
        per_page = max(1, per_page)
        total_records = len(display_matrix)
        total_pages = math.ceil(total_records / per_page) if total_records > 0 else 1

        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_records = display_matrix[start_idx:end_idx]

        return {
            "summary": summary,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_records": total_records,
                "total_pages": total_pages,
            },
            "records": paginated_records,
            "all_records": display_matrix,
        }

    def _empty_report_result(self, start_date, end_date, dept, sec, page, per_page):
        summary = {
            "total_students": 0, "present_students": 0, "absent_students": 0,
            "total_periods": 0, "present_periods": 0, "absent_periods": 0,
            "attendance_percentage": 0.0, "start_date": str(start_date), "end_date": str(end_date),
            "department": dept, "section": sec
        }
        return {
            "summary": summary,
            "pagination": {"page": page, "per_page": per_page, "total_records": 0, "total_pages": 1},
            "records": [], "all_records": []
        }

    def _parse_time_safe(self, val: Any) -> dt_time:
        if isinstance(val, timedelta):
            s = int(val.total_seconds())
            return dt_time(s // 3600, (s % 3600) // 60)
        if not isinstance(val, str):
            return dt_time(0, 0)
        try:
            parts = val.strip().split(":")
            return dt_time(int(parts[0]), int(parts[1]))
        except Exception:
            return dt_time(0, 0)

    # ── Semester Configuration Methods ────────────────────────────

    def get_semesters(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """Return list of semester configurations."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return []
            cursor = None
            try:
                sql = "SELECT id, academic_year, year_level, semester, start_date, end_date, is_active FROM semesters_config"
                if active_only: sql += " WHERE is_active = TRUE"
                sql += " ORDER BY academic_year DESC, year_level ASC, semester ASC;"
                cursor = self._connection.cursor(dictionary=True)
                cursor.execute(sql)
                return cursor.fetchall() or []
            except MySQLError as e:
                logger.error("Error fetching semesters: %s", e)
                return []
            finally:
                if cursor: cursor.close()

    def get_semester_by_params(self, academic_year: str, year_level: str, semester: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific semester config."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return None
            cursor = None
            try:
                sql = "SELECT * FROM semesters_config WHERE academic_year=%s AND year_level=%s AND semester=%s LIMIT 1"
                cursor = self._connection.cursor(dictionary=True)
                cursor.execute(sql, (academic_year, year_level, semester))
                return cursor.fetchone()
            except MySQLError: return None
            finally:
                if cursor: cursor.close()

    def add_semester(self, academic_year, year_level, semester, start_date, end_date, is_active=True):
        """Add or update a semester config."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return False, "DB Error"
            cursor = None
            try:
                sql = ("INSERT INTO semesters_config (academic_year, year_level, semester, start_date, end_date, is_active) "
                       "VALUES (%s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE "
                       "start_date=VALUES(start_date), end_date=VALUES(end_date), is_active=VALUES(is_active)")
                cursor = self._connection.cursor()
                cursor.execute(sql, (academic_year, year_level, semester, start_date, end_date, is_active))
                self._connection.commit()
                return True, "Semester configured successfully."
            except MySQLError as e:
                return False, str(e)
            finally:
                if cursor: cursor.close()

    # ── Section Timetable Methods ──────────────────────────────────

    CSD_B_TIMETABLE_DATA = [
        # MONDAY
        ("2026-27", "II", "I", "CSD", "B", "MON", 1, "09:15:00", "10:20:00", "MFCS", "THEORY", "Dr. S. Kumar", "faculty.csd1@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "MON", 2, "10:20:00", "11:10:00", "General Aptitude (TPC)", "THEORY", "Mr. V. Rao", "faculty.csd2@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "MON", 3, "11:10:00", "12:00:00", "General Aptitude (TPC)", "THEORY", "Mr. V. Rao", "faculty.csd2@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "MON", 4, "12:00:00", "13:00:00", "FOS/SEC", "THEORY", "Ms. R. Priya", "faculty.csd3@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "MON", 5, "13:40:00", "14:30:00", "Programming in Python Lab / Database Systems Lab", "LAB", "Dr. P. Sharma", "faculty.csd4@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "MON", 6, "14:30:00", "15:20:00", "Programming in Python Lab / Database Systems Lab", "LAB", "Dr. P. Sharma", "faculty.csd4@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "MON", 7, "15:20:00", "16:10:00", "Programming in Python Lab / Database Systems Lab", "LAB", "Dr. P. Sharma", "faculty.csd4@example.com"),
        # TUESDAY
        ("2026-27", "II", "I", "CSD", "B", "TUE", 1, "09:15:00", "10:20:00", "DBS", "THEORY", "Dr. M. Reddy", "faculty.csd5@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "TUE", 2, "10:20:00", "11:10:00", "MFCS", "THEORY", "Dr. S. Kumar", "faculty.csd1@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "TUE", 3, "11:10:00", "12:00:00", "PP", "THEORY", "Mr. K. Sai", "faculty.csd6@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "TUE", 4, "12:00:00", "13:00:00", "FOS/SEC", "THEORY", "Ms. R. Priya", "faculty.csd3@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "TUE", 5, "13:40:00", "14:30:00", "COUN", "OTHER", "Admin", "admin@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "TUE", 6, "14:30:00", "15:20:00", "UHV", "THEORY", "Dr. G. Rao", "faculty.csd7@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "TUE", 7, "15:20:00", "16:10:00", "Sports", "OTHER", "Mr. P. Coach", "faculty.sports@example.com"),
        # WEDNESDAY
        ("2026-27", "II", "I", "CSD", "B", "WED", 1, "09:15:00", "10:20:00", "PP", "THEORY", "Mr. K. Sai", "faculty.csd6@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "WED", 2, "10:20:00", "11:10:00", "MFCS", "THEORY", "Dr. S. Kumar", "faculty.csd1@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "WED", 3, "11:10:00", "12:00:00", "FOS", "THEORY", "Ms. R. Priya", "faculty.csd3@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "WED", 4, "12:00:00", "13:00:00", "Library", "OTHER", "Librarian", "lib@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "WED", 5, "13:40:00", "14:30:00", "DBS", "THEORY", "Dr. M. Reddy", "faculty.csd5@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "WED", 6, "14:30:00", "15:20:00", "DBS", "THEORY", "Dr. M. Reddy", "faculty.csd5@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "WED", 7, "15:20:00", "16:10:00", "UHV", "THEORY", "Dr. G. Rao", "faculty.csd7@example.com"),
        # THURSDAY
        ("2026-27", "II", "I", "CSD", "B", "THU", 1, "09:15:00", "10:20:00", "FOS", "THEORY", "Ms. R. Priya", "faculty.csd3@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "THU", 2, "10:20:00", "11:10:00", "DBS", "THEORY", "Dr. M. Reddy", "faculty.csd5@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "THU", 3, "11:10:00", "12:00:00", "MFCS", "THEORY", "Dr. S. Kumar", "faculty.csd1@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "THU", 4, "12:00:00", "13:00:00", "Library", "OTHER", "Librarian", "lib@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "THU", 5, "13:40:00", "14:30:00", "Programming in Python Lab / Database Systems Lab", "LAB", "Dr. P. Sharma", "faculty.csd4@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "THU", 6, "14:30:00", "15:20:00", "Programming in Python Lab / Database Systems Lab", "LAB", "Dr. P. Sharma", "faculty.csd4@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "THU", 7, "15:20:00", "16:10:00", "Programming in Python Lab / Database Systems Lab", "LAB", "Dr. P. Sharma", "faculty.csd4@example.com"),
        # FRIDAY
        ("2026-27", "II", "I", "CSD", "B", "FRI", 1, "09:15:00", "10:20:00", "PP", "THEORY", "Mr. K. Sai", "faculty.csd6@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "FRI", 2, "10:20:00", "11:10:00", "Tinkering Lab", "LAB", "Mr. T. Expert", "faculty.tinker@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "FRI", 3, "11:10:00", "12:00:00", "Tinkering Lab", "LAB", "Mr. T. Expert", "faculty.tinker@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "FRI", 4, "12:00:00", "13:00:00", "Tinkering Lab", "LAB", "Mr. T. Expert", "faculty.tinker@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "FRI", 5, "13:40:00", "14:30:00", "PP", "THEORY", "Mr. K. Sai", "faculty.csd6@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "FRI", 6, "14:30:00", "15:20:00", "EC/CC", "OTHER", "Admin", "admin@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "FRI", 7, "15:20:00", "16:10:00", "EC/CC", "OTHER", "Admin", "admin@example.com"),
        # SATURDAY
        ("2026-27", "II", "I", "CSD", "B", "SAT", 1, "09:15:00", "10:20:00", "DBS", "THEORY", "Dr. M. Reddy", "faculty.csd5@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "SAT", 2, "10:20:00", "11:10:00", "FOS", "THEORY", "Ms. R. Priya", "faculty.csd3@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "SAT", 3, "11:10:00", "12:00:00", "PP", "THEORY", "Mr. K. Sai", "faculty.csd6@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "SAT", 4, "12:00:00", "13:00:00", "MFCS", "THEORY", "Dr. S. Kumar", "faculty.csd1@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "SAT", 5, "13:40:00", "14:30:00", "Professional Communication Skills Lab", "LAB", "Ms. E. Speak", "faculty.comm@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "SAT", 6, "14:30:00", "15:20:00", "Professional Communication Skills Lab", "LAB", "Ms. E. Speak", "faculty.comm@example.com"),
        ("2026-27", "II", "I", "CSD", "B", "SAT", 7, "15:20:00", "16:10:00", "Professional Communication Skills Lab", "LAB", "Ms. E. Speak", "faculty.comm@example.com"),
    ]

    def seed_csd_b_timetable(self) -> int:
        """Idempotently seed the CSD-B section timetable. Returns count of newly inserted records."""
        with self._lock:
            if not self._ensure_connection_unlocked():
                logger.error("Cannot seed timetable: no database connection.")
                return 0

            cursor = None
            inserted = 0
            try:
                cursor = self._connection.cursor()
                for row in self.CSD_B_TIMETABLE_DATA:
                    cursor.execute(
                        "INSERT INTO timetable "
                        "(academic_year, year_level, semester, department, section, day_of_week, "
                        "period_number, start_time, end_time, subject, class_type, faculty_name, faculty_contact) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                        "ON DUPLICATE KEY UPDATE "
                        "start_time=VALUES(start_time), end_time=VALUES(end_time), "
                        "subject=VALUES(subject), class_type=VALUES(class_type), "
                        "faculty_name=VALUES(faculty_name), faculty_contact=VALUES(faculty_contact);",
                        row,
                    )
                    inserted += cursor.rowcount
                self._connection.commit()
                logger.info("Seeded CSD-B timetable: %d new records inserted.", inserted)
                return inserted
            except MySQLError as e:
                logger.error("Error seeding CSD-B timetable: %s", e)
                return 0
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass

    def get_timetable(
        self,
        department: str,
        section: str,
        day: Optional[str] = None,
        academic_year: Optional[str] = None,
        year_level: Optional[str] = None,
        semester: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve timetable entries for a department and section, optionally filtered by day/academic context."""
        dept = (department or "").strip().upper()
        sec = (section or "").strip().upper()

        with self._lock:
            if not self._ensure_connection_unlocked():
                logger.error("Cannot query timetable: no database connection.")
                return []

            cursor = None
            try:
                sql = (
                    "SELECT id, academic_year, year_level, semester, department, section, "
                    "day_of_week, period_number, start_time, end_time, subject, class_type, faculty_id, faculty_name, faculty_contact "
                    "FROM timetable WHERE department = %s AND section = %s"
                )
                params: list = [dept, sec]

                if day:
                    sql += " AND day_of_week = %s"
                    params.append(day.strip().upper())
                if academic_year:
                    sql += " AND academic_year = %s"
                    params.append(academic_year.strip())
                if year_level:
                    sql += " AND year_level = %s"
                    params.append(year_level.strip())
                if semester:
                    sql += " AND semester = %s"
                    params.append(semester.strip())

                sql += " ORDER BY FIELD(day_of_week, 'MON','TUE','WED','THU','FRI','SAT','SUN'), period_number ASC;"

                cursor = self._connection.cursor(dictionary=True)
                cursor.execute(sql, tuple(params))
                rows = cursor.fetchall()

                formatted = []
                for r in (rows or []):
                    # Helper to format TIME/timedelta to HH:MM:SS
                    def _fmt_t(val):
                        if isinstance(val, timedelta):
                            s = int(val.total_seconds())
                            return f"{s//3600:02d}:{(s%3600)//60:02d}:00"
                        return str(val)

                    formatted.append({
                        "id": r["id"],
                        "academic_year": r["academic_year"],
                        "year_level": r["year_level"],
                        "semester": r["semester"],
                        "department": r["department"],
                        "section": r["section"],
                        "day_of_week": r["day_of_week"],
                        "period_number": r["period_number"],
                        "start_time": _fmt_t(r["start_time"]),
                        "end_time": _fmt_t(r["end_time"]),
                        "subject": r["subject"],
                        "class_type": r["class_type"],
                        "faculty_id": r.get("faculty_id", ""),
                        "faculty_name": r.get("faculty_name", ""),
                        "faculty_contact": r.get("faculty_contact", ""),
                    })
                return formatted
            except MySQLError as e:
                logger.error("Error querying timetable: %s", e)
                return []
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass

    # Day-of-week mapping from Python weekday() to timetable day codes
    _WEEKDAY_MAP = {0: "MON", 1: "TUE", 2: "WED", 3: "THU", 4: "FRI", 5: "SAT", 6: "SUN"}

    def get_current_timetable_slot(
        self,
        department: str = "CSD",
        section: str = "B",
        academic_year: Optional[str] = None,
        year_level: Optional[str] = None,
        semester: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Determine the active timetable slot for the given department/section at a point in time.

        Args:
            department: Department code (default CSD).
            section: Section letter (default B).
            academic_year: Academic Year (e.g. 2026-27).
            year_level: Year Level (e.g. II).
            semester: Semester (e.g. I).
            now: Datetime to evaluate. If None, uses server's current local time.

        Returns:
            Dict with 'status' key ('ACTIVE', 'BEFORE_CLASS', 'LUNCH', 'NO_CLASS', 'AFTER_CLASS')
            and timetable entry fields when status is ACTIVE.
        """
        if now is None:
            now = datetime.now()

        dept = (department or "").strip().upper()
        sec = (section or "").strip().upper()
        day_code = self._WEEKDAY_MAP.get(now.weekday(), "SUN")
        current_time = now.time()

        base = {"department": dept, "section": sec}

        # Fetch timetable entries for this day from the database
        entries = self.get_timetable(
            department=dept,
            section=sec,
            day=day_code,
            academic_year=academic_year,
            year_level=year_level,
            semester=semester
        )

        if not entries:
            return {**base, "status": "NO_CLASS"}

        # Parse a time string "H:MM:SS" or "HH:MM:SS" into a datetime.time object
        def _parse_time(t_str: str) -> dt_time:
            parts = t_str.strip().split(":")
            return dt_time(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)

        first_start = _parse_time(entries[0]["start_time"])
        last_entry = entries[-1]
        last_start = _parse_time(last_entry["start_time"])
        last_end = _parse_time(last_entry["end_time"])

        # Before first period starts
        if current_time < first_start:
            return {**base, "status": "BEFORE_CLASS"}

        # Lunch break: 13:00-13:40
        lunch_start = dt_time(13, 0, 0)
        lunch_end = dt_time(13, 40, 0)
        if lunch_start <= current_time < lunch_end:
            return {**base, "status": "LUNCH"}

        # Last period is open-ended ("P7 = 15:20 onward")
        if current_time >= last_start:
            return {
                **base,
                "status": "ACTIVE",
                "day": day_code,
                "period_number": last_entry["period_number"],
                "start_time": last_entry["start_time"],
                "end_time": last_entry["end_time"],
                "subject": last_entry["subject"],
                "class_type": last_entry["class_type"],
                "faculty_id": last_entry.get("faculty_id", ""),
                "faculty_name": last_entry.get("faculty_name", ""),
                "faculty_contact": last_entry.get("faculty_contact", ""),
            }

        # Check each period (except last, already handled above)
        for entry in entries[:-1]:
            start = _parse_time(entry["start_time"])
            end = _parse_time(entry["end_time"])
            if start <= current_time < end:
                return {
                    **base,
                    "status": "ACTIVE",
                    "day": day_code,
                    "period_number": entry["period_number"],
                    "start_time": entry["start_time"],
                    "end_time": entry["end_time"],
                    "subject": entry["subject"],
                    "class_type": entry["class_type"],
                    "faculty_id": entry.get("faculty_id", ""),
                    "faculty_name": entry.get("faculty_name", ""),
                    "faculty_contact": entry.get("faculty_contact", ""),
                }

        # Fallback — shouldn't normally be reached with contiguous periods
        return {**base, "status": "NO_CLASS"}

        # Fallback — shouldn't normally be reached with contiguous periods
        return {**base, "status": "NO_CLASS"}

    def get_hod_contact(self, dept_code: str) -> str:
        """Fetch HOD contact for a department."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return ""
            cursor = None
            try:
                cursor = self._connection.cursor()
                cursor.execute("SELECT hod_contact FROM departments WHERE code = %s", (dept_code.upper(),))
                row = cursor.fetchone()
                return row[0] if row else ""
            except MySQLError:
                return ""
            finally:
                if cursor: cursor.close()

    def is_notification_sent(self, dept: str, sec: str, att_date: date, period_num: int, role: str) -> bool:
        """Check if a notification has already been sent for this period/role."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return False
            cursor = None
            try:
                cursor = self._connection.cursor()
                cursor.execute(
                    "SELECT 1 FROM notifications_log WHERE department=%s AND section=%s AND attendance_date=%s AND period_number=%s AND recipient_role=%s AND status='SENT'",
                    (dept, sec, att_date, period_num, role)
                )
                return cursor.fetchone() is not None
            except MySQLError:
                return False
            finally:
                if cursor: cursor.close()

    def log_notification(self, dept: str, sec: str, att_date: date, period_num: int, role: str, contact: str, status: str):
        """Log a notification attempt."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return
            cursor = None
            try:
                cursor = self._connection.cursor()
                cursor.execute(
                    "INSERT INTO notifications_log (department, section, attendance_date, period_number, recipient_role, recipient_contact, status) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE status=%s, sent_at=CURRENT_TIMESTAMP",
                    (dept, sec, att_date, period_num, role, contact, status, status)
                )
                self._connection.commit()
            except MySQLError as e:
                logger.error("Error logging notification: %s", e)
            finally:
                if cursor: cursor.close()

    def drop_table(self, table_name: str = "attendance"):
        """Drop a table from the database (Dangerous - use with caution)."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return
            cursor = None
            try:
                cursor = self._connection.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
                self._connection.commit()
                logger.warning("Table dropped: %s", table_name)
            except MySQLError as e:
                logger.error("Error dropping table: %s", e)
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass

    # ── Faculty Management ────────────────────────────────────────

    def get_faculty(self, department: Optional[str] = None, active_only: bool = False) -> List[Dict[str, Any]]:
        """Return list of faculty members."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return []
            cursor = None
            try:
                sql = "SELECT id, faculty_id, name, department, phone, email, is_active FROM faculty WHERE 1=1"
                params = []
                if department and department != "ALL":
                    sql += " AND department = %s"
                    params.append(department.strip().upper())
                if active_only:
                    sql += " AND is_active = TRUE"
                sql += " ORDER BY name ASC;"
                cursor = self._connection.cursor(dictionary=True)
                cursor.execute(sql, tuple(params))
                return cursor.fetchall() or []
            except MySQLError as e:
                logger.error("Error fetching faculty: %s", e)
                return []
            finally:
                if cursor: cursor.close()

    def add_faculty(self, faculty_id, name, department, phone=None, email=None, is_active=True):
        """Add or update a faculty member."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return False, "DB Error"
            cursor = None
            try:
                sql = ("INSERT INTO faculty (faculty_id, name, department, phone, email, is_active) "
                       "VALUES (%s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE "
                       "name=VALUES(name), department=VALUES(department), phone=VALUES(phone), "
                       "email=VALUES(email), is_active=VALUES(is_active)")
                cursor = self._connection.cursor()
                cursor.execute(sql, (faculty_id, name, department.strip().upper(), phone, email, is_active))
                self._connection.commit()
                return True, "Faculty member saved successfully."
            except MySQLError as e:
                return False, str(e)
            finally:
                if cursor: cursor.close()

    def delete_faculty(self, faculty_id):
        """Delete a faculty member if not in use by timetable."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return False, "DB Error"
            cursor = None
            try:
                cursor = self._connection.cursor()
                # Check if used in timetable
                cursor.execute("SELECT COUNT(*) FROM timetable WHERE faculty_id = %s", (faculty_id,))
                if cursor.fetchone()[0] > 0:
                    return False, "Cannot delete faculty: In use by timetable."
                cursor.execute("DELETE FROM faculty WHERE faculty_id = %s", (faculty_id,))
                self._connection.commit()
                return True, "Faculty deleted."
            except MySQLError as e:
                return False, str(e)
            finally:
                if cursor: cursor.close()

    # ── Holiday Management ────────────────────────────────────────

    def get_holidays(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """Return list of configured holidays."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return []
            cursor = None
            try:
                sql = "SELECT id, holiday_date, description, is_active FROM holidays"
                if active_only: sql += " WHERE is_active = TRUE"
                sql += " ORDER BY holiday_date DESC;"
                cursor = self._connection.cursor(dictionary=True)
                cursor.execute(sql)
                rows = cursor.fetchall() or []
                for r in rows:
                    r["holiday_date"] = str(r["holiday_date"])
                return rows
            except MySQLError as e:
                logger.error("Error fetching holidays: %s", e)
                return []
            finally:
                if cursor: cursor.close()

    def add_holiday(self, holiday_date, description, is_active=True):
        """Add or update a holiday."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return False, "DB Error"
            cursor = None
            try:
                sql = ("INSERT INTO holidays (holiday_date, description, is_active) "
                       "VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE "
                       "description=VALUES(description), is_active=VALUES(is_active)")
                cursor = self._connection.cursor()
                cursor.execute(sql, (holiday_date, description, is_active))
                self._connection.commit()
                return True, "Holiday saved."
            except MySQLError as e:
                return False, str(e)
            finally:
                if cursor: cursor.close()

    def delete_holiday(self, holiday_date):
        """Remove a holiday."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return False
            cursor = None
            try:
                cursor = self._connection.cursor()
                cursor.execute("DELETE FROM holidays WHERE holiday_date = %s", (holiday_date,))
                self._connection.commit()
                return True
            except MySQLError: return False
            finally:
                if cursor: cursor.close()

    def is_holiday(self, check_date: date) -> bool:
        """Check if a date is a holiday (Sunday or in holidays table)."""
        if check_date.weekday() == 6: return True # Sunday
        with self._lock:
            if not self._ensure_connection_unlocked(): return False
            cursor = None
            try:
                cursor = self._connection.cursor()
                cursor.execute("SELECT 1 FROM holidays WHERE holiday_date = %s AND is_active = TRUE LIMIT 1", (check_date,))
                return cursor.fetchone() is not None
            except MySQLError: return False
            finally:
                if cursor: cursor.close()

    # ── Camera Management ─────────────────────────────────────────

    def get_cameras(self, department: Optional[str] = None, section: Optional[str] = None, active_only: bool = False) -> List[Dict[str, Any]]:
        """Return list of configured cameras with passwords masked."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return []
            cursor = None
            try:
                sql = "SELECT id, name, department, section, classroom, source, username, '***' as password, is_active, created_at, updated_at FROM cameras WHERE 1=1"
                params = []
                if department and department != "ALL":
                    sql += " AND department = %s"
                    params.append(department.strip().upper())
                if section and section != "ALL":
                    sql += " AND section = %s"
                    params.append(section.strip().upper())
                if active_only:
                    sql += " AND is_active = TRUE"
                sql += " ORDER BY department ASC, section ASC, name ASC;"

                cursor = self._connection.cursor(dictionary=True)
                cursor.execute(sql, tuple(params))
                rows = cursor.fetchall() or []
                for r in rows:
                    r["created_at"] = str(r["created_at"])
                    r["updated_at"] = str(r["updated_at"])
                return rows
            except MySQLError as e:
                logger.error("Error fetching cameras: %s", e)
                return []
            finally:
                if cursor: cursor.close()

    def get_camera_by_id(self, cam_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single camera by ID."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return None
            cursor = None
            try:
                cursor = self._connection.cursor(dictionary=True)
                cursor.execute("SELECT * FROM cameras WHERE id = %s", (cam_id,))
                return cursor.fetchone()
            except MySQLError: return None
            finally:
                if cursor: cursor.close()

    def add_camera(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Add a new camera configuration."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return False, "DB Error"
            cursor = None
            try:
                sql = ("INSERT INTO cameras (name, department, section, classroom, source, username, password, is_active) "
                       "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)")
                params = (
                    data["name"], data["department"].strip().upper(), data["section"].strip().upper(),
                    data.get("classroom"), data["source"], data.get("username"), data.get("password"),
                    data.get("is_active", True)
                )
                cursor = self._connection.cursor()
                cursor.execute(sql, params)
                self._connection.commit()
                return True, "Camera configured successfully."
            except MySQLError as e:
                return False, str(e)
            finally:
                if cursor: cursor.close()

    def update_camera(self, cam_id: int, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Update an existing camera configuration."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return False, "DB Error"
            cursor = None
            try:
                update_fields = []
                params = []
                for field in ["name", "department", "section", "classroom", "source", "username", "is_active"]:
                    if field in data:
                        val = data[field]
                        if field in ["department", "section"]: val = val.strip().upper()
                        update_fields.append(f"{field}=%s")
                        params.append(val)

                # Password only updated if provided and not masked
                if "password" in data and data["password"] != "***":
                    update_fields.append("password=%s")
                    params.append(data["password"])

                if not update_fields: return True, "No changes."

                sql = f"UPDATE cameras SET {', '.join(update_fields)} WHERE id=%s"
                params.append(cam_id)

                cursor = self._connection.cursor()
                cursor.execute(sql, tuple(params))
                self._connection.commit()
                return True, "Camera updated."
            except MySQLError as e:
                return False, str(e)
            finally:
                if cursor: cursor.close()

    def delete_camera(self, cam_id: int) -> bool:
        """Deactivate or remove a camera."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return False
            cursor = None
            try:
                cursor = self._connection.cursor()
                cursor.execute("DELETE FROM cameras WHERE id = %s", (cam_id,))
                self._connection.commit()
                return True
            except MySQLError: return False
            finally:
                if cursor: cursor.close()

    # ── Timetable Management ──────────────────────────────────────

    def add_timetable_entry(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Add or update a timetable entry."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return False, "DB Error"
            cursor = None
            try:
                # Resolve faculty info if faculty_id provided
                f_id = data.get("faculty_id")
                f_name = data.get("faculty_name", "")
                f_contact = data.get("faculty_contact", "")

                if f_id:
                    f_cursor = self._connection.cursor(dictionary=True)
                    f_cursor.execute("SELECT name, phone, email FROM faculty WHERE faculty_id = %s", (f_id,))
                    f_row = f_cursor.fetchone()
                    if f_row:
                        f_name = f_row["name"]
                        f_contact = f_row["phone"] or f_row["email"] or ""
                    f_cursor.close()

                sql = ("INSERT INTO timetable (academic_year, year_level, semester, department, section, "
                       "day_of_week, period_number, start_time, end_time, subject, class_type, "
                       "faculty_id, faculty_name, faculty_contact) "
                       "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                       "ON DUPLICATE KEY UPDATE start_time=VALUES(start_time), end_time=VALUES(end_time), "
                       "subject=VALUES(subject), class_type=VALUES(class_type), faculty_id=VALUES(faculty_id), "
                       "faculty_name=VALUES(faculty_name), faculty_contact=VALUES(faculty_contact)")

                params = (
                    data["academic_year"], data["year_level"], data["semester"],
                    data["department"].strip().upper(), data["section"].strip().upper(),
                    data["day_of_week"].strip().upper(), int(data["period_number"]),
                    data["start_time"], data["end_time"], data["subject"], data["class_type"],
                    f_id, f_name, f_contact
                )

                cursor = self._connection.cursor()
                cursor.execute(sql, params)
                self._connection.commit()
                return True, "Timetable entry saved."
            except Exception as e:
                logger.error("Error adding timetable entry: %s", e)
                return False, str(e)
            finally:
                if cursor: cursor.close()

    def delete_timetable_entry(self, entry_id: int) -> bool:
        """Remove a timetable entry by ID."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return False
            cursor = None
            try:
                cursor = self._connection.cursor()
                cursor.execute("DELETE FROM timetable WHERE id = %s", (entry_id,))
                self._connection.commit()
                return True
            except MySQLError: return False
            finally:
                if cursor: cursor.close()

    # ── Student Management ────────────────────────────────────────

    def add_student(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Register or update a student's metadata."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return False, "DB Error"
            cursor = None
            try:
                sql = ("INSERT INTO students (student_id, name, department, year_level, section, academic_year, semester, is_active) "
                       "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                       "ON DUPLICATE KEY UPDATE name=VALUES(name), department=VALUES(department), year_level=VALUES(year_level), "
                       "section=VALUES(section), academic_year=VALUES(academic_year), semester=VALUES(semester), is_active=VALUES(is_active)")

                params = (
                    data["student_id"].strip(), data["name"].strip(), data["department"].strip().upper(),
                    data.get("year_level"), data.get("section", "B").strip().upper(),
                    data.get("academic_year"), data.get("semester"), data.get("is_active", True)
                )

                cursor = self._connection.cursor()
                cursor.execute(sql, params)
                self._connection.commit()
                return True, "Student record saved."
            except MySQLError as e:
                return False, str(e)
            finally:
                if cursor: cursor.close()

    def get_students(self, department: Optional[str] = None, section: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch students filtered by department/section."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return []
            cursor = None
            try:
                sql = "SELECT * FROM students WHERE 1=1"
                params = []
                if department and department != "ALL":
                    sql += " AND department = %s"
                    params.append(department.strip().upper())
                if section and section != "ALL":
                    sql += " AND section = %s"
                    params.append(section.strip().upper())

                sql += " ORDER BY department ASC, name ASC;"
                cursor = self._connection.cursor(dictionary=True)
                cursor.execute(sql, tuple(params))
                rows = cursor.fetchall() or []
                for r in rows:
                    r["created_at"] = str(r["created_at"])
                    r["updated_at"] = str(r["updated_at"])
                return rows
            except MySQLError: return []
            finally:
                if cursor: cursor.close()

    def get_students_by_department(self, department_code: str) -> List[Dict[str, Any]]:
        """Alias for get_students(department=department_code)."""
        return self.get_students(department=department_code)

    def get_student_by_id(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Fetch student record by student_id."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return None
            cursor = None
            try:
                cursor = self._connection.cursor(dictionary=True)
                cursor.execute("SELECT * FROM students WHERE student_id = %s LIMIT 1", (student_id.strip(),))
                return cursor.fetchone()
            except MySQLError: return None
            finally:
                if cursor: cursor.close()

    def get_attendance_by_id(self, attendance_id: int) -> Optional[Dict[str, Any]]:
        """Fetch attendance record by id."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return None
            cursor = None
            try:
                cursor = self._connection.cursor(dictionary=True)
                cursor.execute("SELECT * FROM attendance WHERE id = %s LIMIT 1", (attendance_id,))
                return cursor.fetchone()
            except MySQLError: return None
            finally:
                if cursor: cursor.close()

    def delete_student_record(self, student_id: str) -> bool:
        """Remove a student record from DB."""
        with self._lock:
            if not self._ensure_connection_unlocked(): return False
            cursor = None
            try:
                cursor = self._connection.cursor()
                cursor.execute("DELETE FROM students WHERE student_id = %s", (student_id,))
                self._connection.commit()
                return True
            except MySQLError: return False
            finally:
                if cursor: cursor.close()
        """Drop a table (used only for testing cleanup)."""
        with self._lock:
            if not self._ensure_connection_unlocked():
                return

            cursor = None
            try:
                cursor = self._connection.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`;")
                self._connection.commit()
                logger.info("Table '%s' dropped.", table_name)
            except MySQLError as e:
                logger.error("Error dropping table: %s", e)
            finally:
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
