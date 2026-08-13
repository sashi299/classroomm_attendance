"""
attendance.py - Timetable-aware attendance recording manager with visual evidence.

Manages the logic for marking student attendance per timetable period with:
  - Automatic timetable slot detection via get_current_timetable_slot().
  - Visual face evidence capture and quality-based deduplication per student & period.
  - In-memory cache per (student_id, attendance_date, hourly_period, department).
  - MySQL UNIQUE constraint (student_id, attendance_date, hourly_period, department) as source of truth.
  - Timetable-aware blocking: no attendance before class, during lunch, or on no-class days.
  - Clear status feedback for UI and live video overlays.
"""

import os
import cv2
import logging
import numpy as np
from datetime import date, datetime, time as dt_time
from enum import Enum
from typing import Optional, Set, Tuple, Dict, Any

from database import DatabaseManager
from system_state import system_state_manager

# Module-level logger
logger = logging.getLogger(__name__)

# Base directory for saving face crop evidence images
EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "evidence")


class AttendanceStatus(Enum):
    """Result status for an attendance marking attempt."""
    NEWLY_MARKED = "PRESENT"
    ALREADY_PRESENT = "ALREADY PRESENT"
    FAILED = "FAILED"
    SKIPPED_UNKNOWN = "SKIPPED"
    SKIPPED_EXAM_MODE = "EXAM MODE PAUSED"
    SKIPPED_NO_CLASS = "NO CLASS"
    SKIPPED_BEFORE_CLASS = "BEFORE CLASS"
    SKIPPED_LUNCH = "LUNCH BREAK"
    SKIPPED_HOLIDAY = "HOLIDAY"


def extract_face_crop(
    frame: np.ndarray,
    face_location: Tuple[int, int, int, int],
    margin_pct: float = 0.2,
) -> Optional[np.ndarray]:
    """
    Extract face crop from an OpenCV BGR frame with margin padding.

    Args:
        frame: The full camera frame (BGR).
        face_location: (top, right, bottom, left).
        margin_pct: Margin fraction to expand bounding box.

    Returns:
        Cropped face image as BGR numpy array, or None.
    """
    if frame is None or frame.size == 0 or not face_location:
        return None

    top, right, bottom, left = face_location
    h, w = frame.shape[:2]
    fh = max(1, bottom - top)
    fw = max(1, right - left)

    pad_h = int(fh * margin_pct)
    pad_w = int(fw * margin_pct)

    crop_top = max(0, top - pad_h)
    crop_bottom = min(h, bottom + pad_h)
    crop_left = max(0, left - pad_w)
    crop_right = min(w, right + pad_w)

    crop = frame[crop_top:crop_bottom, crop_left:crop_right]
    return crop.copy() if crop.size > 0 else None


def calculate_crop_quality(crop: np.ndarray, distance: float = 0.0) -> float:
    """
    Calculate quality score for a face crop.

    Combines:
      - Face crop area (resolution / detail).
      - Match confidence (lower face distance = higher quality).
      - Image sharpness (Laplacian variance).

    Higher score indicates a clearer, higher-quality recognition crop.
    """
    if crop is None or crop.size == 0:
        return 0.0

    h, w = crop.shape[:2]
    area = h * w

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness <= 0:
        sharpness = 1.0

    match_factor = max(0.1, 1.0 - min(float(distance), 1.0))
    return float(area * (sharpness ** 0.5) * match_factor)


class AttendanceManager:
    """
    Manages timetable-aware student attendance recording with in-memory caching
    and visual face evidence preservation.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self._cache_date: Optional[date] = None
        # Set of tuples: (student_id, attendance_date, hourly_period, department)
        self._marked_cache: Set[Tuple[str, date, str, str]] = set()

        # Evidence cache: (dept, section, date_str, period_num, student_id_upper) -> (jpeg_bytes, quality_score)
        self._evidence_cache: Dict[Tuple[str, str, str, int, str], Tuple[bytes, float]] = {}

        os.makedirs(EVIDENCE_DIR, exist_ok=True)
        logger.info("AttendanceManager initialized with hourly timetable & evidence support.")

    def _ensure_cache_date(self):
        """Reset cache if date changes."""
        today = date.today()
        if self._cache_date != today:
            if self._cache_date is not None:
                logger.info(
                    "Date changed from %s to %s. Resetting attendance cache.",
                    self._cache_date, today,
                )
            self._cache_date = today
            self._marked_cache.clear()

    @staticmethod
    def _format_period_label(start_time_str: str, end_time_str: str) -> str:
        """Format timetable start/end time strings into an hourly_period label like '09:15-10:20'."""
        def _fmt(t: str) -> str:
            parts = t.strip().split(":")
            return f"{int(parts[0]):02d}:{int(parts[1]):02d}"
        return f"{_fmt(start_time_str)}-{_fmt(end_time_str)}"

    def store_evidence(
        self,
        dept: str,
        sec: str,
        att_date: date,
        period_num: int,
        student_id: str,
        crop: np.ndarray,
        distance: float = 0.0,
    ) -> bool:
        """
        Store or update face recognition evidence image for a student and period.

        Keeps only ONE crop per student per period, preferring the highest-quality crop.

        Returns:
            True if evidence was stored or updated (higher quality), False otherwise.
        """
        if crop is None or crop.size == 0 or not student_id:
            return False

        dept_code = (dept or "CSD").strip().upper()
        sec_code = (sec or "B").strip().upper()
        s_id = student_id.strip().upper()
        d_str = str(att_date)

        quality = calculate_crop_quality(crop, distance=distance)

        key = (dept_code, sec_code, d_str, int(period_num), s_id)

        # Check existing quality
        if key in self._evidence_cache:
            _, existing_q = self._evidence_cache[key]
            if quality <= existing_q:
                logger.debug("Evidence crop for [%s] P%d quality (%.1f <= %.1f) not updated.", s_id, period_num, quality, existing_q)
                return False

        # Encode crop to JPEG
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return False

        jpeg_bytes = buf.tobytes()
        self._evidence_cache[key] = (jpeg_bytes, quality)

        # Save to disk for persistence
        filename = f"{dept_code}_{sec_code}_{d_str}_P{period_num}_{s_id}.jpg"
        file_path = os.path.join(EVIDENCE_DIR, filename)
        try:
            with open(file_path, "wb") as f:
                f.write(jpeg_bytes)
        except Exception as e:
            logger.warning("Could not write evidence to disk: %s", e)

        logger.info("Saved best-quality face evidence for [%s] P%d (quality: %.1f)", s_id, period_num, quality)
        return True

    def get_evidence(
        self,
        dept: str,
        sec: str,
        att_date: date,
        period_num: int,
        student_id: str,
    ) -> Optional[bytes]:
        """
        Retrieve JPEG bytes of recognized face evidence for a student in a period.

        Returns:
            bytes of JPEG image, or None if evidence unavailable.
        """
        dept_code = (dept or "CSD").strip().upper()
        sec_code = (sec or "B").strip().upper()
        s_id = (student_id or "").strip().upper()
        d_str = str(att_date)

        key = (dept_code, sec_code, d_str, int(period_num), s_id)

        # 1. Check in-memory cache
        if key in self._evidence_cache:
            return self._evidence_cache[key][0]

        # 2. Check disk persistence
        filename = f"{dept_code}_{sec_code}_{d_str}_P{period_num}_{s_id}.jpg"
        file_path = os.path.join(EVIDENCE_DIR, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "rb") as f:
                    jpeg_bytes = f.read()
                    self._evidence_cache[key] = (jpeg_bytes, 1.0)
                    return jpeg_bytes
            except Exception:
                pass

        return None

    def mark_present(
        self,
        student_id: str,
        student_name: str,
        dept_code: str = "CSD",
        hourly_period: Optional[str] = None,
        section: str = "B",
        now: Optional[datetime] = None,
        face_crop: Optional[np.ndarray] = None,
        frame: Optional[np.ndarray] = None,
        face_location: Optional[Tuple[int, int, int, int]] = None,
        distance: float = 0.0,
    ) -> AttendanceStatus:
        """
        Mark a recognized student present for the current date & timetable period.

        Checks the CSD-B timetable to determine the active slot. Blocks attendance
        outside of active periods (before class, lunch, no class days).
        Preserves the best-quality face crop evidence when available.
        """
        if system_state_manager.is_exam_mode_enabled():
            logger.debug("Skipping attendance insertion for [%s]: Exam Mode is active.", student_id)
            return AttendanceStatus.SKIPPED_EXAM_MODE

        if not student_id or student_id == "Unknown":
            return AttendanceStatus.SKIPPED_UNKNOWN

        self._ensure_cache_date()
        if now is None:
            now = datetime.now()
        today = now.date()
        current_time = now.time()
        dept = (dept_code or "CSD").strip().upper()
        sec = (section or "B").strip().upper()

        # ── Holiday check ─────────────────────────────────────────
        if self.db.is_holiday(today):
            logger.debug("Skipping attendance for [%s]: today is a holiday/Sunday.", student_id)
            return AttendanceStatus.SKIPPED_HOLIDAY

        # ── Timetable slot detection ──────────────────────────────
        slot = self.db.get_current_timetable_slot(department=dept, section=sec, now=now)
        slot_status = slot.get("status", "NO_CLASS")

        if slot_status == "BEFORE_CLASS":
            logger.debug("Skipping attendance for [%s]: before class hours.", student_id)
            return AttendanceStatus.SKIPPED_BEFORE_CLASS

        if slot_status == "LUNCH":
            logger.debug("Skipping attendance for [%s]: lunch break.", student_id)
            return AttendanceStatus.SKIPPED_LUNCH

        if slot_status == "NO_CLASS":
            logger.debug("Skipping attendance for [%s]: no class scheduled.", student_id)
            return AttendanceStatus.SKIPPED_NO_CLASS

        # ── ACTIVE period: extract timetable context ──────────────
        period_number = slot.get("period_number", 1)
        subject = slot.get("subject")
        class_type = slot.get("class_type")
        slot_start = slot.get("start_time", "")
        slot_end = slot.get("end_time", "")

        # Process face crop evidence if frame/crop provided
        crop_to_save = face_crop
        if crop_to_save is None and frame is not None and face_location is not None:
            crop_to_save = extract_face_crop(frame, face_location)

        if crop_to_save is not None:
            self.store_evidence(
                dept=dept,
                sec=sec,
                att_date=today,
                period_num=period_number,
                student_id=student_id,
                crop=crop_to_save,
                distance=distance,
            )

        # Derive hourly_period from timetable slot (e.g. "09:15-10:20")
        if not hourly_period:
            hourly_period = self._format_period_label(slot_start, slot_end)

        cache_key = (student_id, today, hourly_period, dept)

        # Fast path — in-memory cache check
        if cache_key in self._marked_cache:
            logger.debug("Cache hit: [%s] %s (%s) already marked for period %s.", student_id, student_name, dept, hourly_period)
            return AttendanceStatus.ALREADY_PRESENT

        # Insert into database with timetable context
        inserted = self.db.insert_attendance(
            student_id=student_id,
            student_name=student_name,
            attendance_date=today,
            attendance_time=current_time,
            status="Present",
            hourly_period=hourly_period,
            department=dept,
            section=sec,
            period_number=period_number,
            subject=subject,
            class_type=class_type,
        )

        self._marked_cache.add(cache_key)

        if inserted:
            logger.info("TIMETABLE ATTENDANCE MARKED: [%s] %s (%s-%s) P%s %s for %s",
                         student_id, student_name, dept, sec, period_number, subject, hourly_period)
            return AttendanceStatus.NEWLY_MARKED
        else:
            logger.info("Already in DB: [%s] %s (%s) for period %s", student_id, student_name, dept, hourly_period)
            return AttendanceStatus.ALREADY_PRESENT

    def get_status_text(self, status: AttendanceStatus, student_name: str) -> str:
        """Generate a human-readable status string for the video overlay."""
        if status == AttendanceStatus.NEWLY_MARKED:
            return f"PRESENT - {student_name}"
        elif status == AttendanceStatus.ALREADY_PRESENT:
            return f"ALREADY PRESENT - {student_name}"
        elif status == AttendanceStatus.FAILED:
            return f"DB ERROR - {student_name}"
        elif status == AttendanceStatus.SKIPPED_BEFORE_CLASS:
            return f"BEFORE CLASS - {student_name}"
        elif status == AttendanceStatus.SKIPPED_LUNCH:
            return f"LUNCH BREAK - {student_name}"
        elif status == AttendanceStatus.SKIPPED_NO_CLASS:
            return f"NO CLASS - {student_name}"
        else:
            return ""

    def get_today_count(self, dept_code: Optional[str] = None, hourly_period: Optional[str] = None) -> int:
        """Return the count of distinct students marked present today for a dept/period."""
        self._ensure_cache_date()
        today = date.today()
        dept = (dept_code or "").strip().upper() if dept_code else None

        records = self.db.get_today_attendance(today=today, dept_code=dept, period=hourly_period)
        distinct_ids = {r["student_id"] for r in records}
        return len(distinct_ids)

    def is_marked_today(self, student_id: str, dept_code: str = "CSD", hourly_period: Optional[str] = None) -> bool:
        """Check if student is in today's attendance cache for period."""
        self._ensure_cache_date()
        today = date.today()
        dept = (dept_code or "CSD").strip().upper()
        if not hourly_period:
            hourly_period = self.db.get_current_hourly_period()
        return (student_id, today, hourly_period, dept) in self._marked_cache

    def finalize_period_attendance(
        self,
        dept_code: str = "CSD",
        section: str = "B",
        att_date: Optional[date] = None,
        period_number: int = 1,
    ) -> Tuple[int, int, int]:
        """
        Finalize period attendance:
        Compare registered section students with recognized present records.
        Automatically mark unrecognized students as ABSENT in the database.

        Returns:
            Tuple of (present_count, absent_count, total_students)
        """
        if att_date is None:
            att_date = date.today()
        dept = (dept_code or "CSD").strip().upper()
        sec = (section or "B").strip().upper()

        # 1. Fetch timetable slot details
        day_code = getattr(self.db, "_WEEKDAY_MAP", {}).get(att_date.weekday(), "SUN")
        timetable_entries = self.db.get_timetable(department=dept, section=sec, day=day_code)
        slot_info = next((t for t in timetable_entries if t.get("period_number") == period_number), None)

        subject = slot_info.get("subject", "N/A") if slot_info else "N/A"
        class_type = slot_info.get("class_type", "THEORY") if slot_info else "THEORY"
        slot_start = slot_info.get("start_time", "09:00:00") if slot_info else "09:00:00"
        slot_end = slot_info.get("end_time", "10:00:00") if slot_info else "10:00:00"
        hourly_period = self._format_period_label(slot_start, slot_end)

        # 2. Get registered students
        all_students = self.db.get_students_by_department(dept)
        section_students = [
            s for s in all_students
            if s.get("section", "B").strip().upper() == sec or not s.get("section")
        ]

        # 3. Get present attendance records
        today_records = self.db.get_today_attendance(today=att_date, dept_code=dept, period=hourly_period)
        present_student_ids = {
            r["student_id"].strip().upper()
            for r in today_records
            if r.get("status", "").upper() == "PRESENT"
        }

        # 4. Insert ABSENT for unrecognized students
        absent_count = 0
        end_time_obj = dt_time(10, 0)
        if isinstance(slot_end, dt_time):
            end_time_obj = slot_end
        elif isinstance(slot_end, str) and ":" in slot_end:
            try:
                parts = slot_end.split(":")
                end_time_obj = dt_time(int(parts[0]), int(parts[1]))
            except Exception:
                pass

        for student in section_students:
            s_id = student.get("student_id", "").strip().upper()
            s_name = student.get("name", s_id)
            if s_id and s_id not in present_student_ids:
                inserted = self.db.insert_attendance(
                    student_id=s_id,
                    student_name=s_name,
                    attendance_date=att_date,
                    attendance_time=end_time_obj,
                    status="Absent",
                    hourly_period=hourly_period,
                    department=dept,
                    section=sec,
                    period_number=period_number,
                    subject=subject,
                    class_type=class_type,
                )
                if inserted:
                    absent_count += 1

        present_count = len(present_student_ids)
        total_students = len(section_students)
        logger.info(
            "Finalized Period %d (%s-%s): %d Present, %d Absent out of %d total.",
            period_number, dept, sec, present_count, absent_count, total_students
        )
        return present_count, absent_count, total_students

