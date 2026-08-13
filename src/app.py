"""
app.py - Centralized Web Dashboard Application for Department HODs and Admins.
"""

import io
import csv
import os
import sys
import time
import calendar
import logging
from datetime import date, datetime, timedelta
from typing import Generator, List

import cv2
import numpy as np
from flask import (
    Flask, render_template, Response, jsonify,
    request, session, redirect, url_for,
)

from config import Config
from camera import CameraStream
from camera_manager import CameraManager, generate_offline_frame
from face_engine import FaceEngine
from face_engine_manager import FaceEngineManager
from database import DatabaseManager
from attendance import AttendanceManager, AttendanceStatus
from system_state import system_state_manager
from report_exporter import generate_attendance_csv, generate_attendance_excel, generate_attendance_pdf
from auth import (
    authenticate_user, login_user, logout_user,
    get_current_user, is_admin, login_required,
)
import face_recognition
from notifications import NotificationManager, EmailNotificationProvider
from scheduler import BackgroundScheduler

# ── Logging Configuration ─────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dashboard_app")

# ── Colors & Font Settings ─────────────────────────────────────
COLOR_RECOGNIZED = (0, 200, 0)
COLOR_UNKNOWN = (0, 0, 220)
COLOR_NEWLY_MARKED = (0, 255, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_FPS = (0, 255, 255)
COLOR_STATUS_BAR = (40, 40, 40)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_NAME = 0.55
FONT_SCALE_FPS = 0.5
FONT_SCALE_INFO = 0.45
FONT_SCALE_ATTENDANCE = 0.5
FONT_THICKNESS = 1
BOX_THICKNESS = 2

# ── Flask Application Setup ───────────────────────────────────
template_dir = os.path.join(os.path.dirname(__file__), "templates")
app = Flask(__name__, template_folder=template_dir)

config = Config()
app.secret_key = config.SECRET_KEY


@app.after_request
def apply_cors_headers(response):
    origins = getattr(config, "CORS_ORIGINS", "*")
    response.headers["Access-Control-Allow-Origin"] = origins
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    return response

face_engine_manager: FaceEngineManager = None
db_manager: DatabaseManager = None
attendance_manager: AttendanceManager = None
camera_manager: CameraManager = None
notification_manager: NotificationManager = None
scheduler: BackgroundScheduler = None
app_initialized = False


def initialize_components():
    global face_engine_manager, db_manager, attendance_manager, camera_manager, notification_manager, scheduler, app_initialized
    if app_initialized: return

    if db_manager is None:
        db_manager = DatabaseManager(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            database=config.MYSQL_DATABASE,
        )
        db_manager.connect()

    if attendance_manager is None:
        attendance_manager = AttendanceManager(db_manager=db_manager)

    if face_engine_manager is None:
        face_engine_manager = FaceEngineManager(
            base_dir=config.STUDENTS_BASE_DIR,
            recognition_threshold=config.RECOGNITION_THRESHOLD,
        )

    if camera_manager is None:
        cams_map = Config.get_department_cameras_map(db_manager=db_manager)
        camera_manager = CameraManager(db_manager=db_manager, department_cameras=cams_map)

    if notification_manager is None:
        provider = None
        if config.SMTP_HOST and config.SMTP_USERNAME:
            provider = EmailNotificationProvider(
                host=config.SMTP_HOST, port=config.SMTP_PORT,
                username=config.SMTP_USERNAME, password=config.SMTP_PASSWORD,
                sender=config.SMTP_FROM
            )
        notification_manager = NotificationManager(db_manager=db_manager, provider=provider, attendance_manager=attendance_manager)

    if scheduler is None:
        scheduler = BackgroundScheduler(notification_manager=notification_manager, interval_seconds=60)
        scheduler.start()

    app_initialized = True


def draw_dashboard_overlay(frame, results_with_status, fps=0.0, registered_count=0, today_count=0, dept_code="", hourly_period=""):
    h, w = frame.shape[:2]
    if system_state_manager.is_exam_mode_enabled():
        cv2.rectangle(frame, (0, 0), (w, 36), (0, 140, 255), cv2.FILLED)
        cv2.putText(frame, "EXAM MODE ACTIVE", (15, 24), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    for result, att_status in results_with_status:
        top, right, bottom, left = result.face_location
        color = COLOR_RECOGNIZED if (result.is_recognized and result.student_id != "Unknown") else COLOR_UNKNOWN
        cv2.rectangle(frame, (left, top), (right, bottom), color, BOX_THICKNESS)
        label = f"{result.student_id} - {result.student_name}" if color == COLOR_RECOGNIZED else "UNKNOWN"
        cv2.putText(frame, label, (left, top - 10), FONT, FONT_SCALE_NAME, color, FONT_THICKNESS, cv2.LINE_AA)
        if att_status in [AttendanceStatus.NEWLY_MARKED, AttendanceStatus.ALREADY_PRESENT]:
            cv2.putText(frame, "PRESENT", (left, bottom + 20), FONT, FONT_SCALE_ATTENDANCE, color, FONT_THICKNESS + 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 0), (w, 30), COLOR_STATUS_BAR, cv2.FILLED)
    status_text = f"FPS: {fps:.1f} | Dept: {dept_code} | Period: {hourly_period} | Reg: {registered_count} | Present: {today_count}"
    cv2.putText(frame, status_text, (10, 20), FONT, FONT_SCALE_FPS, COLOR_TEXT, FONT_THICKNESS, cv2.LINE_AA)
    return frame


def generate_mjpeg_frames(dept_code, section="B", cam_name="Default"):
    initialize_components()
    engine = face_engine_manager.get_engine(dept_code)
    camera = camera_manager.get_camera(dept_code, section, cam_name=cam_name)
    fps_start = time.time(); f_count = 0; last_results = []
    while True:
        success, frame = camera.read_frame() if camera else (False, None)
        if not success or frame is None:
            time.sleep(0.1); continue
        f_count += 1
        current_period = db_manager.get_current_hourly_period()
        if not system_state_manager.is_exam_mode_enabled() and (f_count % config.FRAME_SKIP == 0 or not last_results):
            rec_results = engine.recognize_faces(frame, resize_factor=Config.FACE_RESIZE_FACTOR)
            last_results = [(r, attendance_manager.mark_present(r.student_id, r.student_name, dept_code, current_period) if (r.is_recognized and r.student_id != "Unknown") else AttendanceStatus.SKIPPED_UNKNOWN) for r in rec_results]
        if time.time() - fps_start >= 1.0:
            fps = f_count / (time.time() - fps_start); f_count = 0; fps_start = time.time()
        annotated = draw_dashboard_overlay(frame.copy(), last_results, 0.0, engine.get_registered_count(), attendance_manager.get_today_count(dept_code, current_period), dept_code, current_period)
        yield _encode_frame_as_mjpeg(annotated)


def _encode_frame_as_mjpeg(frame):
    ret, jpeg = cv2.imencode('.jpg', frame)
    return (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n') if ret else b''


def get_iou(boxA, boxB):
    tA, rA, bA, lA = boxA; tB, rB, bB, lB = boxB
    yA = max(tA, tB); xA = max(lA, lB); yB = min(bA, bB); xB = min(rA, rB)
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (rA - lA) * (bA - tA); areaB = (rB - lB) * (bB - tB)
    union = float(areaA + areaB - inter)
    return inter / union if union > 0 else 0


@app.route("/api/students/enroll/validate-frame", methods=["POST"])
@login_required
def api_enroll_validate_frame():
    """Validate a single camera frame for enrollment."""
    initialize_components()
    if "photo" not in request.files: return jsonify({"success": False, "error": "No image provided"}), 400
    img_bytes = request.files["photo"].read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None: return jsonify({"success": False, "error": "Invalid format"}), 400

    h, w = img.shape[:2]; rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Detection
    raw_locs = face_recognition.face_locations(rgb_img, number_of_times_to_upsample=0)

    # 2. Filtering
    final_faces = []
    sorted_locs = sorted(raw_locs, key=lambda x: (x[2]-x[0])*(x[1]-x[3]), reverse=True)

    for f in sorted_locs:
        # Ignore small background faces (< 12% height)
        if (f[2]-f[0]) < (h * 0.12): continue

        # Deep verification
        if not face_recognition.face_encodings(rgb_img, [f]): continue

        is_dup = False
        for existing in final_faces:
            # Merge if they overlap significantly (>30%)
            # This handles duplicate boxes on the same physical face.
            if get_iou(f, existing) > 0.3:
                is_dup = True; break
        if not is_dup:
            final_faces.append(f)

    if len(final_faces) == 0:
        return jsonify({"success": False, "error": "No face detected"}), 400
    if len(final_faces) > 1:
        return jsonify({"success": False, "error": f"Multiple faces ({len(final_faces)}) detected. Ensure only 1 person is visible."}), 400

    # 3. Blur Check
    if cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 5:
        return jsonify({"success": False, "error": "Image too blurry. Hold steady."}), 400

    t, r, b, l = final_faces[0]
    return jsonify({"success": True, "face_location": {"top": int(t), "right": int(r), "bottom": int(b), "left": int(l)}})


@app.route("/api/students", methods=["GET"])
@login_required
def api_get_students():
    initialize_components()
    user = get_current_user()
    dept = request.args.get("dept", user["department_code"]).strip().upper() if user["role"] == "admin" else user["department_code"]
    students = face_engine_manager.get_student_details(dept, db_manager)
    return jsonify({"success": True, "students": students, "count": len(students)})


@app.route("/api/students/add", methods=["POST"])
@login_required
def api_add_student():
    """Add a new student and their photos with strict validation."""
    initialize_components()
    user = get_current_user()
    dept = request.form.get("department", "").strip().upper()
    if user["role"] != "admin" and dept != user["department_code"]:
        return jsonify({"success": False, "error": "Unauthorized"}), 403

    s_id = request.form.get("student_id", "").strip(); s_name = request.form.get("student_name", "").strip()
    if not all([dept, s_id, s_name]): return jsonify({"success": False, "error": "Missing fields"}), 400

    photo_files = request.files.getlist("photo")
    if not photo_files or not any(photo_files):
        return jsonify({"success": False, "error": "No photos provided."}), 400

    # Validate each photo before database entry
    for i, f in enumerate(photo_files):
        if f:
            fname = f.filename if (f.filename and f.filename != "blob") else f"sample_{i+1}.jpg"
            success, err = face_engine_manager.add_student_photo(dept, s_id, s_name, f, fname)
            if not success:
                return jsonify({"success": False, "error": f"Photo {i+1}: {err}"}), 400

    db_manager.add_student({"student_id": s_id, "name": s_name, "department": dept, "year_level": request.form.get("year_level"), "section": request.form.get("section", "B"), "academic_year": request.form.get("academic_year"), "semester": request.form.get("semester"), "is_active": True})
    return jsonify({"success": True, "message": "Registered successfully."})


@app.route("/api/students/delete", methods=["POST"])
@login_required
def api_delete_student():
    initialize_components()
    data = request.json or request.form
    dept = data.get("department", "").upper(); s_id = str(data.get("student_id", ""))
    if not is_admin() and dept != get_current_user()["department_code"]: return jsonify({"success": False}), 403
    face_engine_manager.delete_student(dept, s_id); db_manager.delete_student_record(s_id)
    return jsonify({"success": True})


@app.route("/api/attendance/history")
@login_required
def api_attendance_history():
    initialize_components()
    user = get_current_user(); dept = request.args.get("dept", user["department_code"]).upper()
    if user["role"] != "admin" and dept != user["department_code"] and dept != "ALL": return jsonify({"success": False}), 403
    recs = db_manager.get_attendance_history(None, None, request.args.get("search"), dept if dept != "ALL" else None)
    return jsonify({"records": [{"attendance_date": str(r["attendance_date"]), "attendance_time": _format_time(r["attendance_time"]), "student_id": r["student_id"], "student_name": r["student_name"], "department": r["department"], "hourly_period": r["hourly_period"], "status": "Present"} for r in recs], "count": len(recs)})


@app.route("/api/system/status")
@login_required
def api_system_status():
    initialize_components()
    return jsonify(system_state_manager.get_system_status(db_manager, camera_manager, face_engine_manager))


@app.route("/api/system/config-status")
@login_required
def api_config_status():
    """Return production configuration safety summary with secrets masked."""
    initialize_components()
    is_valid, warnings, summary = Config.validate_production_config(db_manager=db_manager)
    user = get_current_user()
    if not is_admin():
        return jsonify({
            "success": True,
            "role": user["role"],
            "department_code": user["department_code"],
            "is_valid": is_valid,
            "warnings_count": len(warnings),
            "operational": {
                "department_code": user["department_code"],
                "is_valid": is_valid,
            }
        })
    return jsonify({
        "success": True,
        "role": user["role"],
        "is_valid": is_valid,
        "production_ready": is_valid,
        "storage_backend": config.STORAGE_BACKEND,
        "warnings": warnings,
        "config": summary,
        "summary": summary,
    })


@app.route("/api/system/exam-mode/<action>", methods=["POST"])
@login_required
def api_toggle_exam_mode(action):
    if not is_admin(): return jsonify({"success": False}), 403
    return jsonify({"success": system_state_manager.enable_exam_mode()[0] if action == "enable" else system_state_manager.disable_exam_mode()[0]})


@app.route("/api/reports/attendance")
@login_required
def api_reports_general():
    initialize_components()
    user = get_current_user(); dept = (request.args.get("dept") or request.args.get("department") or user["department_code"]).upper()
    if user["role"] != "admin" and dept != user["department_code"]: return jsonify({"success": False}), 403
    data = db_manager.get_attendance_report_data(None, None, dept, request.args.get("section", "B"), None, None, None, face_engine_manager.get_student_details(dept, db_manager))
    return jsonify({"success": True, **data})


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if session.get("logged_in"): return redirect(url_for("dashboard"))
    if request.method == "POST":
        u = authenticate_user(request.form.get("username", "").strip(), request.form.get("password", ""))
        if u: login_user(u); return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid credentials.")
    return render_template("login.html", error=None)

@app.route("/logout")
def logout(): logout_user(); return redirect(url_for("login_page"))

@app.route("/")
@login_required
def dashboard():
    initialize_components(); u = get_current_user()
    return render_template("dashboard.html", department_name=u["department_name"], department_code=u["department_code"], username=u["username"], is_admin=(u["role"]=="admin"), departments=Config.get_active_department_codes(db_manager))

@app.route("/video_feed")
@login_required
def video_feed():
    initialize_components()
    return Response(generate_mjpeg_frames(request.args.get("dept", get_current_user()["department_code"]).upper(), request.args.get("section", "B")), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/health", methods=["GET"])
def health_check():
    """Health status endpoint for cloud platform load balancers and orchestrators."""
    db_ok = False
    if db_manager and hasattr(db_manager, "_connection") and db_manager._connection:
        try:
            db_ok = db_manager._connection.is_connected()
        except Exception:
            db_ok = False

    return jsonify({
        "status": "ok",
        "environment": config.ENVIRONMENT,
        "database_connected": db_ok,
    }), 200


@app.route("/api/evidence/<int:attendance_id>/photo")
@login_required
def api_get_evidence_photo(attendance_id: int):
    """Protected endpoint to serve face recognition evidence image."""
    initialize_components()
    user = get_current_user()

    # Query attendance record
    rec = db_manager.get_attendance_by_id(attendance_id)
    if not rec:
        return jsonify({"error": "Attendance record not found"}), 404

    dept = rec.get("department", "CSD").upper()
    if not is_admin() and user["department_code"] != dept:
        return jsonify({"error": "Unauthorized access to department evidence"}), 403

    att_date = rec.get("attendance_date")
    if isinstance(att_date, str):
        att_date = datetime.strptime(att_date, "%Y-%m-%d").date()

    period_num = rec.get("period_number") or 1
    sec = rec.get("section") or "B"
    s_id = rec.get("student_id", "")

    evidence_bytes = attendance_manager.get_evidence(
        dept=dept, sec=sec, att_date=att_date, period_num=period_num, student_id=s_id
    )

    if not evidence_bytes:
        return jsonify({"error": "Recognition image unavailable"}), 404

    return Response(evidence_bytes, mimetype="image/jpeg")


@app.route("/api/students/<student_id>/photo")
@login_required
def api_get_student_photo(student_id: str):
    """Protected endpoint to serve student reference enrollment photo."""
    initialize_components()
    user = get_current_user()
    s_id = student_id.strip().upper()

    student = db_manager.get_student_by_id(s_id)
    if not student:
        return jsonify({"error": "Student not found"}), 404

    dept = student.get("department", "CSD").upper()
    if not is_admin() and user["department_code"] != dept:
        return jsonify({"error": "Unauthorized"}), 403

    # Resolve student reference photo path
    photo_path = face_engine_manager.get_student_photo_path(dept, s_id)
    if not photo_path or not os.path.isfile(photo_path):
        return jsonify({"error": "Student photo unavailable"}), 404

    try:
        with open(photo_path, "rb") as f:
            data = f.read()
        return Response(data, mimetype="image/jpeg")
    except Exception as e:
        logger.error("Error reading student photo [%s]: %s", s_id, e)
        return jsonify({"error": "Could not read photo"}), 500


@app.route("/api/attendance/finalize", methods=["POST"])
@login_required
def api_finalize_attendance():
    """Manually or automatically finalize period attendance to mark absent students."""
    initialize_components()
    user = get_current_user()
    data = request.json or request.form or {}

    dept = (data.get("department") or user["department_code"]).strip().upper()
    if not is_admin() and dept != user["department_code"]:
        return jsonify({"success": False, "error": "Unauthorized"}), 403

    sec = (data.get("section") or "B").strip().upper()
    p_num = int(data.get("period_number", 1))

    d_str = data.get("date")
    att_date = datetime.strptime(d_str, "%Y-%m-%d").date() if d_str else date.today()

    p_cnt, a_cnt, t_cnt = attendance_manager.finalize_period_attendance(
        dept_code=dept, section=sec, att_date=att_date, period_number=p_num
    )

    return jsonify({
        "success": True,
        "department": dept,
        "section": sec,
        "period_number": p_num,
        "date": str(att_date),
        "present_count": p_cnt,
        "absent_count": a_cnt,
        "total_students": t_cnt,
    })


def _format_time(t):
    if hasattr(t, "total_seconds"):
        s = int(t.total_seconds()); return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"
    return str(t)


if __name__ == "__main__":
    initialize_components()
    is_prod = config.ENVIRONMENT == "production"
    app.run(host="0.0.0.0", port=config.PORT, debug=not is_prod, threaded=True)

