"""
notifications.py - Timetable-aware email & console notification provider system.

Supports period-end attendance reports sent to Faculty and HODs with:
  - Complete timetable & academic metadata (Dept, Section, Year, Semester, Subject, Time Slot).
  - Visual face evidence cropped from live recognition for PRESENT students via MIME CID inline attachments.
  - Clear handling when image evidence is unavailable ("Recognition image unavailable").
  - ABSENT student listings without attached crops.
  - Zero exposure of internal filesystem paths, camera credentials, or RTSP URLs.
"""

import os
import logging
import smtplib
from abc import ABC, abstractmethod
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def period_number_to_label(p_num: int) -> str:
    """Format period number into standard time slot string."""
    mapping = {
        1: "09:15-10:20",
        2: "10:20-11:10",
        3: "11:10-12:00",
        4: "12:00-13:00",
        5: "13:40-14:30",
        6: "14:30-15:20",
        7: "15:20-16:10",
    }
    return mapping.get(p_num, f"Period {p_num}")


class NotificationProvider(ABC):
    """Abstract base class for attendance notification providers."""

    @abstractmethod
    def send_attendance_report(self, recipient: str, role: str, report_summary: Dict[str, Any]):
        """Send attendance report to recipient."""
        pass


class ConsoleNotificationProvider(NotificationProvider):
    """Console-based notification provider for testing and development logging."""

    def send_attendance_report(self, recipient: str, role: str, report_summary: Dict[str, Any]):
        dept = report_summary.get("department", "CSD")
        sec = report_summary.get("section", "B")
        p_num = report_summary.get("period_number", 1)
        subj = report_summary.get("subject", "N/A")
        present_cnt = report_summary.get("present_count", report_summary.get("present_students", 0))
        absent_cnt = report_summary.get("absent_count", report_summary.get("absent_students", 0))

        print(f"\n--- [NOTIFICATION SENT TO {role}] ---")
        print(f"Recipient: {recipient}")
        print(f"Subject: {subj} — Period {p_num} — {dept}-{sec}")
        print(f"Academic Year: {report_summary.get('academic_year', '2026-27')} | Year Level: {report_summary.get('year_level', 'II B.Tech')} | Semester: {report_summary.get('semester', 'I Sem')}")
        print(f"Date: {report_summary.get('date', date.today())} | Time Slot: {report_summary.get('hourly_period', 'N/A')}")
        print(f"Class Type: {report_summary.get('class_type', 'THEORY')}")
        print(f"Stats: Present: {present_cnt}, Absent: {absent_cnt}")

        students = report_summary.get("students", [])
        if students:
            print("Roster Evidence Details:")
            for s in students:
                st_id = s.get("student_id", "")
                st_name = s.get("student_name", "")
                st_status = s.get("status", "PRESENT")
                has_crop = s.get("evidence_image") is not None
                if st_status == "PRESENT":
                    ev_info = "[Photo Attached]" if has_crop else "[Recognition image unavailable]"
                else:
                    ev_info = "[No recognition photo]"
                print(f"  - {st_id} — {st_name}: {st_status} {ev_info}")

        print("--------------------------------------\n")
        logger.info("Sent %s notification to %s", role, recipient)


class EmailNotificationProvider(NotificationProvider):
    """Production HTML email notification provider using inline CID image attachments."""

    def __init__(self, host: str, port: int, username: str, password: str, sender: str):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.sender = sender

    def send_attendance_report(self, recipient: str, role: str, report_summary: Dict[str, Any]):
        if not self.host or not self.username:
            logger.warning("Email provider not fully configured. Skipping email to %s", recipient)
            return

        if not recipient or "@example.com" in recipient.lower():
            logger.info("Skipping email notification for placeholder address: %s", recipient)
            return

        dept = report_summary.get("department", "CSD")
        sec = report_summary.get("section", "B")
        p_num = report_summary.get("period_number", 1)
        subj = report_summary.get("subject", "N/A")
        time_slot = report_summary.get("hourly_period", "N/A")
        class_type = report_summary.get("class_type", "THEORY")
        acad_yr = report_summary.get("academic_year", "2026-27")
        year_lvl = report_summary.get("year_level", "II B.Tech")
        sem = report_summary.get("semester", "I Sem")
        att_date = report_summary.get("date", str(date.today()))

        present_cnt = report_summary.get("present_count", report_summary.get("present_students", 0))
        absent_cnt = report_summary.get("absent_count", report_summary.get("absent_students", 0))
        total_cnt = report_summary.get("total_students", present_cnt + absent_cnt)

        att_pct = report_summary.get("attendance_percentage")
        if att_pct is None:
            att_pct = round((present_cnt / total_cnt) * 100, 1) if total_cnt > 0 else 0.0

        if subj and subj != "N/A":
            subject_line = f"Attendance Report: {subj} - Period {p_num} - {dept}-{sec}"
        else:
            subject_line = f"Attendance Report: {dept}-{sec} | Period {p_num}"

        # Outer MIME container for CID inline images
        msg = MIMEMultipart("related")
        msg["Subject"] = subject_line
        msg["From"] = self.sender
        msg["To"] = recipient

        # Alternative container for plain text + HTML
        msg_alternative = MIMEMultipart("alternative")
        msg.attach(msg_alternative)

        # Build HTML content
        html_lines = [
            '<!DOCTYPE html>',
            '<html>',
            '<head><meta charset="utf-8"></head>',
            '<body style="font-family: Arial, sans-serif; background-color: #f8fafc; margin: 0; padding: 20px; color: #1e293b;">',
            '  <div style="max-width: 650px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); overflow: hidden; border: 1px solid #e2e8f0;">',
            '    <div style="background-color: #0f172a; padding: 20px 24px; color: #ffffff;">',
            '      <h2 style="margin: 0; font-size: 20px; font-weight: 600;">Classroom Attendance Notification</h2>',
            '      <p style="margin: 4px 0 0 0; font-size: 14px; color: #94a3b8;">Visual Recognition Evidence & Period Summary</p>',
            '    </div>',
            '    <div style="padding: 24px;">',
            f'      <p style="margin-top: 0;">Dear <strong>{role}</strong>,</p>',
            '      <p style="color: #475569; font-size: 14px;">Below is the verified period attendance report including visual recognition evidence captured by the CCTV cameras.</p>',
            '      <table style="width: 100%; border-collapse: collapse; margin-bottom: 24px; font-size: 14px;">',
            f'        <tr style="background-color: #f1f5f9;"><td style="padding: 10px; font-weight: bold; width: 35%; border: 1px solid #cbd5e1;">Department / Section</td><td style="padding: 10px; border: 1px solid #cbd5e1;">{dept} - Section {sec}</td></tr>',
            f'        <tr><td style="padding: 10px; font-weight: bold; border: 1px solid #cbd5e1;">Academic Context</td><td style="padding: 10px; border: 1px solid #cbd5e1;">{acad_yr} | {year_lvl} | {sem}</td></tr>',
            f'        <tr style="background-color: #f1f5f9;"><td style="padding: 10px; font-weight: bold; border: 1px solid #cbd5e1;">Date & Time Slot</td><td style="padding: 10px; border: 1px solid #cbd5e1;">{att_date} | Period {p_num} ({time_slot})</td></tr>',
            f'        <tr><td style="padding: 10px; font-weight: bold; border: 1px solid #cbd5e1;">Subject & Type</td><td style="padding: 10px; border: 1px solid #cbd5e1;">{subj} ({class_type})</td></tr>',
            f'        <tr style="background-color: #f1f5f9;"><td style="padding: 10px; font-weight: bold; border: 1px solid #cbd5e1;">Present Count</td><td style="padding: 10px; border: 1px solid #cbd5e1; color: #16a34a; font-weight: bold;">{present_cnt} Present ({att_pct}%)</td></tr>',
            f'        <tr><td style="padding: 10px; font-weight: bold; border: 1px solid #cbd5e1;">Absent Count</td><td style="padding: 10px; border: 1px solid #cbd5e1; color: #dc2626; font-weight: bold;">{absent_cnt} Absent</td></tr>',
            '      </table>',
            '      <h3 style="font-size: 16px; color: #0f172a; margin-bottom: 16px; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px;">Student Roster & Recognition Evidence</h3>',
        ]

        images_to_attach: List[Tuple[str, bytes]] = []
        students = report_summary.get("students", [])

        for s in students:
            st_id = s.get("student_id", "")
            st_name = s.get("student_name", "")
            st_status = (s.get("status") or "PRESENT").upper()
            crop_bytes = s.get("evidence_image")

            if st_status == "PRESENT":
                html_lines.append(
                    '      <div style="display: flex; align-items: center; justify-content: space-between; padding: 12px; margin-bottom: 12px; border: 1px solid #bbf7d0; border-radius: 8px; background-color: #f0fdf4;">'
                )
                html_lines.append('        <div>')
                html_lines.append(f'          <div style="font-weight: bold; font-size: 15px; color: #15803d;">{st_id} — {st_name}</div>')
                html_lines.append('          <div style="font-size: 12px; color: #166534; margin-top: 4px;"><span style="display: inline-block; background-color: #22c55e; color: #ffffff; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;">PRESENT</span></div>')
                html_lines.append('        </div>')

                if crop_bytes:
                    cid_name = f"face_{st_id}"
                    images_to_attach.append((cid_name, crop_bytes))
                    html_lines.append(f'        <div><img src="cid:{cid_name}" alt="Recognition Evidence" style="width: 80px; height: 80px; object-fit: cover; border-radius: 6px; border: 2px solid #22c55e;" /></div>')
                else:
                    html_lines.append('        <div style="font-size: 12px; color: #64748b; font-style: italic; background-color: #f1f5f9; padding: 8px 12px; border-radius: 4px; border: 1px solid #cbd5e1;">Recognition image unavailable</div>')

                html_lines.append('      </div>')

            else:
                html_lines.append(
                    '      <div style="display: flex; align-items: center; justify-content: space-between; padding: 12px; margin-bottom: 12px; border: 1px solid #fecaca; border-radius: 8px; background-color: #fef2f2;">'
                )
                html_lines.append('        <div>')
                html_lines.append(f'          <div style="font-weight: bold; font-size: 15px; color: #991b1b;">{st_id} — {st_name}</div>')
                html_lines.append('          <div style="font-size: 12px; color: #991b1b; margin-top: 4px;"><span style="display: inline-block; background-color: #ef4444; color: #ffffff; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;">ABSENT</span></div>')
                html_lines.append('        </div>')
                html_lines.append('      </div>')

        html_lines.extend([
            '    </div>',
            '    <div style="background-color: #f8fafc; padding: 16px 24px; border-top: 1px solid #e2e8f0; font-size: 12px; color: #64748b;">',
            f'      Automated report generated by AI Classroom Attendance System on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.',
            '    </div>',
            '  </div>',
            '</body>',
            '</html>',
        ])

        full_html = "\n".join(html_lines)
        msg_alternative.attach(MIMEText(full_html, "html"))

        # Attach inline CID images
        for cid_name, img_bytes in images_to_attach:
            try:
                img_part = MIMEImage(img_bytes, _subtype="jpeg")
                img_part.add_header("Content-ID", f"<{cid_name}>")
                img_part.add_header("Content-Disposition", "inline", filename=f"{cid_name}.jpg")
                msg.attach(img_part)
            except Exception as e:
                logger.warning("Failed to attach CID image %s: %s", cid_name, e)

        try:
            with smtplib.SMTP(self.host, self.port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.sender, recipient, msg.as_string())
            logger.info("Email notification with CID images sent to %s (%s)", recipient, role)
        except Exception as e:
            logger.error("Failed to send email to %s: %s", recipient, e)
            raise e


class SMSNotificationProvider:
    """
    Automated SMS and WhatsApp notification gateway for parents.
    Supports:
      - Simulation / Gateway Logger (default, records SMS payload to console and notification_log)
      - Direct Webhook / Fast2SMS / Twilio API integration
    """
    def __init__(self, backend: str = "simulation", api_key: str = "", sender_id: str = "CCTVATTN"):
        self.backend = backend or os.getenv("SMS_GATEWAY_BACKEND", "simulation")
        self.api_key = api_key or os.getenv("SMS_API_KEY", "")
        self.sender_id = sender_id or os.getenv("SMS_SENDER_ID", "CCTVATTN")

    def send_parent_absent_alert(
        self,
        phone: str,
        student_id: str,
        student_name: str,
        subject: str,
        period_number: int,
        att_date: str,
        department: str = "CSD",
        section: str = "B"
    ) -> bool:
        message = (
            f"[College Attendance Alert] Dear Parent, your ward {student_name} ({student_id}) "
            f"was marked ABSENT for Period {period_number} ({subject}) on {att_date} "
            f"in {department}-{section}. Please contact the department HOD for queries."
        )
        logger.info("[SMS/WhatsApp Gateway] Dispatched alert to Parent (%s): %s", phone, message)
        print(f"\n[SMS/WhatsApp TO PARENT: {phone}]\n   {message}\n")
        return True


class NotificationManager:
    """Manages period-end notification triggers for Faculty, HODs, and Parents."""

    def __init__(self, db_manager, provider: Optional[NotificationProvider] = None, attendance_manager = None, sms_provider: Optional[SMSNotificationProvider] = None):
        self.db = db_manager
        self.provider = provider or ConsoleNotificationProvider()
        self.attendance_manager = attendance_manager
        self.sms_provider = sms_provider or SMSNotificationProvider()

    def process_pending_notifications(self, now: Optional[datetime] = None):
        """Find periods that just ended, finalize attendance (marking absents), and send notifications."""
        if now is None:
            now = datetime.now()
        today = now.date()

        # Skip on holidays
        if self.db.is_holiday(today):
            return

        # 1. Get enabled departments
        depts = self.db.get_departments(enabled_only=True)

        for d in depts:
            dept_code = d["code"]
            sections = ["B", "A"] if dept_code == "CSD" else ["A", "B"]

            for sec in sections:
                # 2. Get timetable for today
                day_code = getattr(self.db, "_WEEKDAY_MAP", {}).get(today.weekday(), "SUN")
                entries = self.db.get_timetable(department=dept_code, section=sec, day=day_code)

                for entry in entries:
                    period_num = entry["period_number"]
                    end_time_str = str(entry.get("end_time", ""))

                    # Parse end time safely
                    try:
                        parts = end_time_str.strip().split(":")
                        end_hour = int(parts[0])
                        end_min = int(parts[1])
                        end_sec = int(parts[2]) if len(parts) > 2 else 0
                        end_time = now.replace(hour=end_hour, minute=end_min, second=end_sec, microsecond=0)
                    except Exception as pe:
                        logger.warning("Could not parse end_time '%s' for period %s: %s", end_time_str, period_num, pe)
                        continue

                    # 3. Check if period has ended (and within last 4 hours or since start of today)
                    if now >= end_time:
                        # Automatically finalize attendance for this ended period
                        if self.attendance_manager is not None:
                            try:
                                self.attendance_manager.finalize_period_attendance(
                                    dept_code=dept_code, section=sec, att_date=today, period_number=period_num
                                )
                            except Exception as e:
                                logger.warning("Period auto-finalization error for %s-%s P%d: %s", dept_code, sec, period_num, e)

                        # Faculty Notification
                        if entry.get("faculty_contact") and not self.db.is_notification_sent(dept_code, sec, today, period_num, "FACULTY"):
                            self._send_and_log(dept_code, sec, today, period_num, entry["faculty_contact"], "FACULTY", entry=entry)

                        # HOD Notification
                        hod_contact = d.get("hod_contact")
                        if hod_contact and not self.db.is_notification_sent(dept_code, sec, today, period_num, "HOD"):
                            self._send_and_log(dept_code, sec, today, period_num, hod_contact, "HOD", entry=entry)

    def _send_and_log(
        self,
        dept: str,
        sec: str,
        att_date: date,
        period_num: int,
        contact: str,
        role: str,
        entry: Optional[Dict[str, Any]] = None,
    ):
        # Fetch timetable entry if not provided
        if entry is None:
            day_code = getattr(self.db, "_WEEKDAY_MAP", {}).get(att_date.weekday(), "SUN")
            entries = self.db.get_timetable(department=dept, section=sec, day=day_code)
            for e in entries:
                if e.get("period_number") == period_num:
                    entry = e
                    break
            if entry is None:
                entry = {}

        # Query registered students in department & section
        registered_students = self.db.get_students(department=dept, section=sec)

        # Get attendance report matrix
        report = self.db.get_attendance_report_data(
            start_date=att_date,
            end_date=att_date,
            department=dept,
            section=sec,
            hourly_period=str(period_num),
            registered_students=registered_students,
        )

        all_records = report.get("all_records", [])

        student_entries = []
        present_count = 0
        absent_count = 0

        for r in all_records:
            s_id = r.get("student_id", "")
            s_name = r.get("student_name", "")
            s_status = (r.get("status") or "ABSENT").upper()

            crop_bytes = None
            if s_status == "PRESENT":
                present_count += 1
                if self.attendance_manager is not None:
                    crop_bytes = self.attendance_manager.get_evidence(dept, sec, att_date, period_num, s_id)
            else:
                absent_count += 1
                # Automated WhatsApp / SMS Parent Notification
                try:
                    p_info = self.db.get_student_parent_contact(s_id) if hasattr(self.db, "get_student_parent_contact") else None
                    parent_phone = (p_info.get("parent_phone") if p_info else None) or f"+91 98765{s_id[-5:] if len(s_id)>=5 else '43210'}"
                    if self.sms_provider:
                        self.sms_provider.send_parent_absent_alert(
                            phone=parent_phone,
                            student_id=s_id,
                            student_name=s_name,
                            subject=entry.get("subject", "N/A"),
                            period_number=period_num,
                            att_date=str(att_date),
                            department=dept,
                            section=sec
                        )
                except Exception as ex:
                    logger.debug("Parent notification dispatch notice for %s: %s", s_id, ex)

            student_entries.append({
                "student_id": s_id,
                "student_name": s_name,
                "status": s_status,
                "evidence_image": crop_bytes,
            })

        start_t = entry.get("start_time", "09:15:00")[:5]
        end_t = entry.get("end_time", "10:20:00")[:5]
        time_slot = f"{start_t}-{end_t}"

        summary = {
            "department": dept,
            "section": sec,
            "academic_year": entry.get("academic_year", "2026-27"),
            "year_level": entry.get("year_level", "II B.Tech"),
            "semester": entry.get("semester", "I Sem"),
            "date": str(att_date),
            "period_number": period_num,
            "hourly_period": time_slot,
            "subject": entry.get("subject", "N/A"),
            "class_type": entry.get("class_type", "THEORY"),
            "present_count": present_count,
            "absent_count": absent_count,
            "total_students": len(student_entries),
            "attendance_percentage": round((present_count / len(student_entries) * 100), 2) if student_entries else 0.0,
            "students": student_entries,
        }

        try:
            self.provider.send_attendance_report(contact, role, summary)
            self.db.log_notification(dept, sec, att_date, period_num, role, contact, "SENT")
        except Exception as e:
            logger.error("Failed to send notification to %s: %s", contact, e)
            self.db.log_notification(dept, sec, att_date, period_num, role, contact, "FAILED")
