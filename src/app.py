"""
app.py - Centralized Web Dashboard Application for Department HODs and Admins.
"""

import io
import csv
import os
import sys

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import calendar
import logging
import threading
from datetime import date, datetime, timedelta
from typing import Generator, List, Dict, Tuple, Optional

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
try:
    import face_recognition
except ImportError:
    face_recognition = None
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

# Security & Session Hardening
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,       # Prevents JavaScript from reading session cookies (Anti-XSS)
    SESSION_COOKIE_SAMESITE="Lax",      # Prevents Cross-Site Request Forgery (CSRF)
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
)


@app.after_request
def apply_security_headers(response):
    origins = getattr(config, "CORS_ORIGINS", "*")
    response.headers["Access-Control-Allow-Origin"] = origins
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"

    # OWASP Top 10 Security Hardening Headers
    response.headers["X-Frame-Options"] = "SAMEORIGIN"           # Anti-Clickjacking
    response.headers["X-Content-Type-Options"] = "nosniff"        # Anti-MIME Sniffing
    response.headers["X-XSS-Protection"] = "1; mode=block"        # Cross-Site Scripting Filter
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
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
        is_rec = result.is_recognized and result.student_id != "Unknown"
        conf_tier = getattr(result, "confidence_tier", "HIGH")
        confirmed = getattr(result, "confirmed_by_temporal", True)

        if conf_tier == "SPOOF_REJECTED" or not getattr(result, "is_live", True):
            color = (0, 100, 255)  # Orange-Red for anti-spoof rejection
            label = "SPOOF REJECTED (PHOTO/SCREEN)"
        elif is_rec and confirmed:
            color = COLOR_RECOGNIZED
            angle_str = f" [{result.matched_angle}]" if getattr(result, "matched_angle", None) else ""
            label = f"{result.student_id} - {result.student_name}{angle_str}"
        elif is_rec and not confirmed:
            color = (0, 215, 255)  # Amber for pending temporal confirmation
            label = f"{result.student_id} - {result.student_name} (CONFIRMING...)"
        else:
            color = COLOR_UNKNOWN
            label = "UNKNOWN"

        cv2.rectangle(frame, (left, top), (right, bottom), color, BOX_THICKNESS)
        cv2.putText(frame, label, (left, top - 10), FONT, FONT_SCALE_NAME, color, FONT_THICKNESS, cv2.LINE_AA)
        if att_status in [AttendanceStatus.NEWLY_MARKED, AttendanceStatus.ALREADY_PRESENT]:
            cv2.putText(frame, "PRESENT", (left, bottom + 20), FONT, FONT_SCALE_ATTENDANCE, color, FONT_THICKNESS + 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 0), (w, 30), COLOR_STATUS_BAR, cv2.FILLED)
    status_text = f"FPS: {fps:.1f} | Dept: {dept_code} | Period: {hourly_period} | Reg: {registered_count} | Present: {today_count}"
    cv2.putText(frame, status_text, (10, 20), FONT, FONT_SCALE_FPS, COLOR_TEXT, FONT_THICKNESS, cv2.LINE_AA)
    return frame


class CameraWorker:
    """
    Central background capture and face recognition pipeline for a single camera source.
    - Captures frames from CameraStream in ONE background thread.
    - Runs InsightFace ArcFace recognition on FRAME_SKIP cadence.
    - Draws bounding boxes and real-time dashboard header.
    - Compresses to adaptive JPEG (quality=60, saving 70% bandwidth for low networks).
    - Publishes to in-memory broadcast buffer shared across all web clients without CPU multiplication.
    """
    def __init__(self, dept_code: str, section: str, cam_name: str):
        self.dept_code = dept_code
        self.section = section
        self.cam_name = cam_name
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest_chunk: bytes = b""
        self._frame_seq: int = 0
        self._active_clients: int = 0
        self._last_request_time: float = time.time()
        self.last_results = []
        self.last_frame = None

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=f"CamWorker-{self.dept_code}-{self.section}-{self.cam_name}"
            )
            self._thread.start()

    def stop(self):
        self._running = False

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self._running

    def _worker_loop(self):
        logger.info("[StreamBroadcaster] Starting worker thread for %s-%s [%s]", self.dept_code, self.section, self.cam_name)
        camera = camera_manager.get_camera(self.dept_code, self.section, cam_name=self.cam_name)
        
        # Load FaceEngine in background so camera preview starts streaming instantly without 20s freeze
        engine = None
        def _load_engine_async():
            nonlocal engine
            try:
                engine = face_engine_manager.get_engine(self.dept_code)
                logger.info("[StreamBroadcaster] FaceEngine ready for %s", self.dept_code)
            except Exception as ex:
                logger.error("[StreamBroadcaster] Failed to load FaceEngine: %s", ex)
        
        threading.Thread(target=_load_engine_async, daemon=True, name=f"EngineLoader-{self.dept_code}").start()

        fps_start = time.time()
        f_count = 0
        current_fps = 0.0
        last_results = []
        last_meta_time = 0.0
        current_period = ""
        registered_count = 0
        today_count = 0

        while self._running:
            try:
                # If no clients have requested frames in over 3 minutes, pause worker
                if self._active_clients <= 0 and (time.time() - self._last_request_time > 180):
                    time.sleep(1.0)
                    continue

                if camera is None or not camera.is_connected:
                    if camera:
                        camera.connect()
                    time.sleep(0.5)
                    if not camera or not camera.is_connected:
                        offline = generate_offline_frame(dept_code=f"{self.dept_code}-{self.section}")
                        ret, jpeg = cv2.imencode('.jpg', offline, [cv2.IMWRITE_JPEG_QUALITY, 55])
                        if ret:
                            chunk = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
                            with self._cond:
                                self._latest_chunk = chunk
                                self._frame_seq += 1
                                self._cond.notify_all()
                        time.sleep(1.0)
                        camera = camera_manager.get_camera(self.dept_code, self.section, cam_name=self.cam_name)
                        continue

                success, frame = camera.read_frame()
                if not success or frame is None:
                    time.sleep(0.04)
                    continue

                f_count += 1
                mean_brightness = float(np.mean(frame))

                now_ts = time.time()
                if now_ts - last_meta_time >= 1.0:
                    try:
                        current_period = db_manager.get_current_hourly_period()
                        if engine is not None:
                            registered_count = engine.get_registered_count()
                        today_count = attendance_manager.get_today_count(self.dept_code, current_period)
                        last_meta_time = now_ts
                    except Exception:
                        pass

                # Face Recognition: run when frame has sufficient lighting (> 3.0) and on FRAME_SKIP cadence
                if engine is not None and mean_brightness >= 3.0 and not system_state_manager.is_exam_mode_enabled() and (f_count % config.FRAME_SKIP == 0):
                    try:
                        rec_results = engine.recognize_faces(frame, resize_factor=Config.FACE_RESIZE_FACTOR)
                        last_results = []
                        for r in rec_results:
                            if r.is_recognized and r.student_id != "Unknown" and getattr(r, "confirmed_by_temporal", True):
                                st = attendance_manager.mark_present(
                                    student_id=r.student_id, student_name=r.student_name,
                                    dept_code=self.dept_code, hourly_period=current_period,
                                    section=self.section, frame=frame, face_location=r.face_location, distance=r.distance
                                )
                                today_count = attendance_manager.get_today_count(self.dept_code, current_period)
                            else:
                                st = AttendanceStatus.SKIPPED_UNKNOWN
                            last_results.append((r, st))
                    except Exception as e:
                        logger.error("[StreamBroadcaster] Recognition error: %s", e)
                elif mean_brightness < 3.0:
                    last_results = []

                self.last_results = last_results
                self.last_frame = frame

                if time.time() - fps_start >= 1.0:
                    current_fps = f_count / (time.time() - fps_start)
                    f_count = 0
                    fps_start = time.time()

                annotated = draw_dashboard_overlay(
                    frame.copy(), last_results, current_fps,
                    registered_count, today_count, self.dept_code, current_period
                )

                # If the camera lens / shutter is dark, display a helpful live indicator
                if mean_brightness < 3.0:
                    h, w = annotated.shape[:2]
                    dark_hint = "CAMERA ACTIVE (LOW LIGHT / SHUTTER CLOSED)"
                    font_s = 0.55
                    (tw, th), _ = cv2.getTextSize(dark_hint, cv2.FONT_HERSHEY_SIMPLEX, font_s, 1)
                    tx = (w - tw) // 2
                    ty = h // 2
                    cv2.rectangle(annotated, (tx - 10, ty - th - 8), (tx + tw + 10, ty + 8), (20, 20, 30), -1)
                    cv2.rectangle(annotated, (tx - 10, ty - th - 8), (tx + tw + 10, ty + 8), (0, 180, 240), 1)
                    cv2.putText(annotated, dark_hint, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_s, (0, 220, 255), 1, cv2.LINE_AA)

                # Highly optimized adaptive JPEG encoding for low network bandwidth (quality 60 + optimize)
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
                ret, jpeg = cv2.imencode('.jpg', annotated, encode_params)
                if ret:
                    chunk = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
                    with self._cond:
                        self._latest_chunk = chunk
                        self._frame_seq += 1
                        self._cond.notify_all()

                time.sleep(0.04)

            except Exception as e:
                logger.error("[StreamBroadcaster] Worker loop error: %s", e)
                time.sleep(0.5)

    def generate_client_stream(self):
        self._last_request_time = time.time()
        last_sent_seq = -1

        # Send immediate "Connecting..." frame
        connecting = generate_offline_frame(dept_code=f"{self.dept_code} (CONNECTING...)")
        ret, jpeg = cv2.imencode('.jpg', connecting, [cv2.IMWRITE_JPEG_QUALITY, 55])
        if ret:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'

        with self._lock:
            self._active_clients += 1

        try:
            while self._running:
                self._last_request_time = time.time()
                chunk_to_send = None
                with self._cond:
                    if self._frame_seq == last_sent_seq:
                        self._cond.wait(timeout=0.2)
                    if self._frame_seq != last_sent_seq and self._latest_chunk:
                        chunk_to_send = self._latest_chunk
                        last_sent_seq = self._frame_seq

                if chunk_to_send:
                    yield chunk_to_send
                    # Pace client at ~15 FPS (~65ms) to guarantee smooth video without overloading weak WiFi/3G/4G
                    time.sleep(0.065)
                else:
                    time.sleep(0.02)
        finally:
            with self._lock:
                self._active_clients = max(0, self._active_clients - 1)


class StreamBroadcaster:
    def __init__(self):
        self._lock = threading.Lock()
        self._workers: Dict[Tuple[str, str, str], CameraWorker] = {}

    def get_stream(self, dept_code: str, section: str = "B", cam_name: str = "Default"):
        dept = (dept_code or "CSD").strip().upper()
        sec = (section or "B").strip().upper()
        cam = (cam_name or "Default").strip()
        key = (dept, sec, cam)

        with self._lock:
            if key not in self._workers or not self._workers[key].is_alive():
                worker = CameraWorker(dept, sec, cam)
                worker.start()
                self._workers[key] = worker
            worker = self._workers[key]

        return worker.generate_client_stream()

stream_broadcaster = StreamBroadcaster()


def generate_mjpeg_frames(dept_code, section="B", cam_name="Default"):
    initialize_components()
    return stream_broadcaster.get_stream(dept_code, section=section, cam_name=cam_name)


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

    # 1. Detection & Filtering
    final_faces = []
    
    if face_recognition is not None:
        raw_locs = face_recognition.face_locations(rgb_img, number_of_times_to_upsample=0)
        sorted_locs = sorted(raw_locs, key=lambda x: (x[2]-x[0])*(x[1]-x[3]), reverse=True)

        for f in sorted_locs:
            if (f[2]-f[0]) < (h * 0.12): continue
            if not face_recognition.face_encodings(rgb_img, [f]): continue

            is_dup = False
            for existing in final_faces:
                if get_iou(f, existing) > 0.3:
                    is_dup = True; break
            if not is_dup:
                final_faces.append(f)
    else:
        from face_engine import get_insightface_app
        face_app = get_insightface_app()
        if face_app:
            faces = face_app.get(img)
            for face in faces:
                l, t, r, b = face.bbox.astype(int)
                if (b - t) < (h * 0.12): continue
                f = (t, r, b, l)
                is_dup = False
                for existing in final_faces:
                    if get_iou(f, existing) > 0.3:
                        is_dup = True; break
                if not is_dup:
                    final_faces.append(f)
        else:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h_box) in faces:
                if h_box < (h * 0.12): continue
                f = (y, x+w, y+h_box, x)
                is_dup = False
                for existing in final_faces:
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


@app.route("/api/attendance/today", methods=["GET"])
@app.route("/api/attendance/live", methods=["GET"])
@login_required
def api_get_today_attendance():
    initialize_components()
    user = get_current_user()
    dept = (request.args.get("dept") or user["department_code"]).strip().upper()
    if user["role"] != "admin" and dept != user["department_code"] and dept != "ALL":
        return jsonify({"success": False, "error": "Unauthorized"}), 403

    target_dept = None if dept == "ALL" else dept
    today_records = db_manager.get_today_attendance(dept_code=target_dept)
    formatted = []
    for r in today_records:
        formatted.append({
            "id": r.get("id"),
            "attendance_date": str(r.get("attendance_date")),
            "attendance_time": _format_time(r.get("attendance_time")),
            "student_id": r.get("student_id"),
            "student_name": r.get("student_name"),
            "department": r.get("department"),
            "section": r.get("section", "B"),
            "period_number": r.get("period_number", 1),
            "hourly_period": r.get("hourly_period", "P1"),
            "subject": r.get("subject", "--"),
            "status": r.get("status", "Present")
        })

    curr_period = db_manager.get_current_hourly_period()
    return jsonify({
        "success": True,
        "department_code": dept,
        "current_period": curr_period,
        "count": len(formatted),
        "attendance": formatted
    })


@app.route("/api/health", methods=["GET"])
def api_health():
    """Unauthenticated healthcheck endpoint for monitors and load balancers."""
    return jsonify({"status": "healthy", "service": "classroom_attendance", "success": True})


@app.route("/api/system/status")
@login_required
def api_system_status():
    initialize_components()
    return jsonify(system_state_manager.get_system_status(db_manager, camera_manager, face_engine_manager))


@app.route("/api/attendance/history")
@login_required
def api_attendance_history():
    initialize_components()
    user = get_current_user(); dept = request.args.get("dept", user["department_code"]).upper()
    if user["role"] != "admin" and dept != user["department_code"] and dept != "ALL": return jsonify({"success": False}), 403
    recs = db_manager.get_attendance_history(None, None, request.args.get("search"), dept if dept != "ALL" else None)
    return jsonify({"records": [{"attendance_date": str(r["attendance_date"]), "attendance_time": _format_time(r["attendance_time"]), "student_id": r["student_id"], "student_name": r["student_name"], "department": r["department"], "hourly_period": r["hourly_period"], "status": r.get("status", "Present")} for r in recs], "count": len(recs)})


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


def _get_user_dept():
    user = get_current_user()
    if user and user.get("department_code"):
        return user["department_code"]
    return session.get("department_code", "CSD")


@app.route("/api/faculty", methods=["GET", "POST", "DELETE"])
@app.route("/api/faculty/<faculty_id>", methods=["DELETE"])
@login_required
def api_faculty(faculty_id=None):
    initialize_components()
    user_dept = _get_user_dept()
    dept = request.args.get("department", user_dept)
    if request.method == "GET":
        facs = db_manager.get_faculty(department=dept) if db_manager else []
        return jsonify({"faculty": facs, "count": len(facs)})
    elif request.method == "POST":
        data = request.get_json() or {}
        ok, msg = db_manager.add_faculty(
            data.get("faculty_id"), data.get("name"),
            data.get("department", user_dept), data.get("phone"), data.get("email")
        ) if db_manager else (False, "DB Error")
        return jsonify({"success": ok, "message": msg}), (201 if ok else 400)
    elif request.method == "DELETE":
        data = request.get_json() or {} if request.is_json else {}
        fid = faculty_id or data.get("id") or data.get("faculty_id")
        ok, msg = db_manager.delete_faculty(fid) if db_manager else (False, "DB Error")
        return jsonify({"success": ok, "message": msg})


@app.route("/api/cameras", methods=["GET", "POST"])
@app.route("/api/cameras/registry", methods=["GET"])
@login_required
def api_cameras_registry():
    initialize_components()
    user_dept = _get_user_dept()
    if request.method == "POST":
        data = request.get_json() or {}
        ok, msg = db_manager.add_camera(data) if db_manager else (False, "DB Error")
        return jsonify({"success": ok, "message": msg}), (201 if ok else 400)
    dept = request.args.get("department", user_dept)
    sec = request.args.get("section")
    active = request.args.get("active_only") == "true"
    cams = db_manager.get_cameras(department=dept, section=sec, active_only=active) if db_manager else []
    return jsonify({"cameras": cams, "count": len(cams)})


@app.route("/api/cameras/add", methods=["POST"])
@login_required
def api_cameras_add():
    initialize_components()
    data = request.get_json() or {}
    ok, msg = db_manager.add_camera(data) if db_manager else (False, "DB Error")
    return jsonify({"success": ok, "message": msg}), (201 if ok else 400)


@app.route("/api/cameras/delete", methods=["POST", "DELETE"])
@login_required
def api_cameras_delete():
    initialize_components()
    data = request.get_json() or {}
    cid = data.get("id") or data.get("camera_id")
    ok, msg = db_manager.delete_camera(cid) if db_manager else (False, "DB Error")
    return jsonify({"success": ok, "message": msg})


@app.route("/api/system/exam-mode/<action>", methods=["POST"])
@login_required
def api_toggle_exam_mode(action):
    if not is_admin(): return jsonify({"success": False}), 403
    return jsonify({"success": system_state_manager.enable_exam_mode()[0] if action == "enable" else system_state_manager.disable_exam_mode()[0]})


def _extract_report_filters():
    user = get_current_user()
    raw_dept = (request.args.get("dept") or request.args.get("department") or user["department_code"]).upper()
    dept = "CSD" if raw_dept == "ALL" and not is_admin() else raw_dept
    sec = request.args.get("section", "B").strip().upper()
    start_d = request.args.get("start_date") or request.args.get("date")
    end_d = request.args.get("end_date") or request.args.get("date")
    ay = request.args.get("academic_year")
    yl = request.args.get("year_level")
    sm = request.args.get("semester")
    search = request.args.get("search")
    status = request.args.get("status", "ALL")
    p_num = request.args.get("hourly_period", "ALL")
    from_t = request.args.get("from_time")
    to_t = request.args.get("to_time")
    try: page = int(request.args.get("page", 1))
    except Exception: page = 1
    try: per_page = int(request.args.get("per_page", 50))
    except Exception: per_page = 50

    return dept, sec, start_d, end_d, ay, yl, sm, search, status, p_num, from_t, to_t, page, per_page


@app.route("/api/reports/attendance", methods=["GET"])
@app.route("/api/reports/attendance/<mode>", methods=["GET"])
@login_required
def api_reports_general(mode=None):
    initialize_components()
    user = get_current_user()
    dept, sec, start_d, end_d, ay, yl, sm, search, status, p_num, from_t, to_t, page, per_page = _extract_report_filters()
    if not is_admin() and dept != user["department_code"]:
        return jsonify({"success": False, "error": "Unauthorized"}), 403

    target_dept = dept if dept != "ALL" else "CSD"
    students = face_engine_manager.get_student_details(target_dept, db_manager)
    data = db_manager.get_attendance_report_data(
        start_date=start_d, end_date=end_d, department=target_dept, section=sec,
        academic_year=ay, year_level=yl, semester=sm, registered_students=students,
        search=search, status=status, hourly_period=p_num, from_time=from_t, to_time=to_t,
        page=page, per_page=per_page
    )
    return jsonify({"success": True, **data})


@app.route("/api/reports/attendance/export/<export_format>", methods=["GET"])
@login_required
def api_export_attendance_report(export_format):
    initialize_components()
    user = get_current_user()
    dept, sec, start_d, end_d, ay, yl, sm, search, status, p_num, from_t, to_t, _, _ = _extract_report_filters()
    if not is_admin() and dept != user["department_code"]:
        return jsonify({"success": False, "error": "Unauthorized"}), 403

    target_dept = dept if dept != "ALL" else "CSD"
    students = face_engine_manager.get_student_details(target_dept, db_manager)
    data = db_manager.get_attendance_report_data(
        start_date=start_d, end_date=end_d, department=target_dept, section=sec,
        academic_year=ay, year_level=yl, semester=sm, registered_students=students,
        search=search, status=status, hourly_period=p_num, from_time=from_t, to_time=to_t,
        page=1, per_page=100000
    )

    fmt = export_format.lower().strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"attendance_report_{target_dept}_{sec}_{timestamp}"

    if fmt == "csv":
        csv_content = generate_attendance_csv(data)
        return Response(csv_content, mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename={filename_base}.csv"})
    elif fmt in ["excel", "xlsx"]:
        excel_bytes = generate_attendance_excel(data)
        return Response(excel_bytes, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename={filename_base}.xlsx"})
    elif fmt == "pdf":
        pdf_bytes = generate_attendance_pdf(data)
        return Response(pdf_bytes, mimetype="application/pdf", headers={"Content-Disposition": f"attachment; filename={filename_base}.pdf"})
    else:
        return jsonify({"success": False, "error": "Unsupported export format"}), 400


@app.route("/api/attendance/export", methods=["GET"])
@login_required
def api_export_attendance_history():
    initialize_components()
    user = get_current_user()
    raw_dept = (request.args.get("dept") or request.args.get("department") or user["department_code"]).upper()
    if not is_admin() and raw_dept != user["department_code"]:
        return jsonify({"success": False, "error": "Unauthorized"}), 403

    start_d = request.args.get("start_date")
    end_d = request.args.get("end_date")
    search = request.args.get("search")
    records = db_manager.get_attendance_history(start_date=start_d, end_date=end_d, search=search, dept=(raw_dept if raw_dept != "ALL" else None))

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Attendance Date", "Attendance Time", "Period", "Student ID", "Student Name", "Department", "Section", "Status"])
    for r in records:
        writer.writerow([
            r.get("attendance_date", ""),
            r.get("attendance_time", ""),
            r.get("hourly_period", ""),
            r.get("student_id", ""),
            r.get("student_name", ""),
            r.get("department", ""),
            r.get("section", ""),
            r.get("status", "")
        ])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Response(output.getvalue(), mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename=attendance_history_{raw_dept}_{timestamp}.csv"})


@app.route("/api/holidays", methods=["GET", "POST"])
@login_required
def api_holidays():
    initialize_components()
    if request.method == "POST":
        data = request.json or request.form or {}
        h_date = data.get("date") or data.get("holiday_date")
        desc = data.get("description", "Holiday")
        if not h_date:
            return jsonify({"success": False, "error": "Date required"}), 400
        ok, msg = db_manager.add_holiday(h_date, desc)
        return jsonify({"success": ok, "message": msg})
    else:
        holidays = db_manager.get_holidays()
        return jsonify({"success": True, "holidays": holidays})

@app.route("/api/holidays/<holiday_id>", methods=["DELETE"])
@login_required
def api_delete_holiday(holiday_id):
    initialize_components()
    ok = db_manager.delete_holiday(holiday_id)
    return jsonify({"success": ok, "message": "Holiday deleted" if ok else "Delete failed"})

@app.route("/api/semesters", methods=["GET", "POST"])
@login_required
def api_semesters():
    initialize_components()
    if request.method == "POST":
        data = request.json or request.form or {}
        ay = data.get("academic_year", "2026-27")
        yl = data.get("year_level", "II")
        sm = data.get("semester", "I")
        sd = data.get("start_date")
        ed = data.get("end_date")
        ok, msg = db_manager.add_semester(ay, yl, sm, sd, ed, True)
        return jsonify({"success": ok, "message": msg})
    else:
        sems = db_manager.get_semesters()
        return jsonify({"success": True, "semesters": sems})

@app.route("/api/departments", methods=["GET", "POST"])
@app.route("/api/departments/<dept_code>/cameras", methods=["POST"])
@login_required
def api_departments(dept_code=None):
    initialize_components()
    if request.method == "POST":
        data = request.json or request.form or {}
        if dept_code:
            cam = data.get("camera_source", "0")
            ok, msg = db_manager.update_department(dept_code, camera_source=cam)
            return jsonify({"success": ok, "message": msg})
        code = (data.get("code") or "").strip().upper()
        name = (data.get("name") or code).strip()
        cam = (data.get("camera_source") or "0").strip()
        if not code:
            return jsonify({"success": False, "error": "Department code required"}), 400
        ok, msg = db_manager.add_department(code, name, cam, True)
        return jsonify({"success": ok, "message": msg})
    else:
        depts = db_manager.get_departments()
        return jsonify({"success": True, "departments": depts})

@app.route("/api/timetable/current", methods=["GET"])
@login_required
def api_timetable_current():
    initialize_components()
    dept = request.args.get("department", "CSD").strip().upper()
    sec = request.args.get("section", "B").strip().upper()
    curr_slot = db_manager.get_current_timetable_slot(dept, sec)

    if not curr_slot or curr_slot.get("status") == "NO_CLASS":
        return jsonify({
            "success": True,
            "status": "OFF_PERIOD",
            "message": "No active class currently scheduled",
            "period_number": 0,
            "subject": "Free Period",
            "faculty": "N/A"
        })

    if curr_slot.get("status") == "BEFORE_CLASS":
        # Get first scheduled class of today
        weekday_map = {0: "MON", 1: "TUE", 2: "WED", 3: "THU", 4: "FRI", 5: "SAT", 6: "SUN"}
        day_code = weekday_map.get(datetime.now().weekday(), "FRI")
        tt_today = db_manager.get_timetable(dept, sec, day=day_code)
        first_entry = tt_today[0] if tt_today else {}

        return jsonify({
            "success": True,
            "status": "BEFORE_CLASS",
            "message": f"First class starts at {first_entry.get('start_time', '09:15')[:5]}",
            "period_number": first_entry.get("period_number", 1),
            "subject": first_entry.get("subject", "Class Starts 09:15"),
            "faculty": first_entry.get("faculty_name", "N/A"),
            "class_type": first_entry.get("class_type", "THEORY"),
            "start_time": first_entry.get("start_time", "09:15:00"),
            "end_time": first_entry.get("end_time", "10:20:00")
        })

    if curr_slot.get("status") == "LUNCH":
        return jsonify({
            "success": True,
            "status": "LUNCH",
            "message": "Lunch Break (13:00-13:40)",
            "period_number": 0,
            "subject": "Lunch Break",
            "faculty": "N/A",
            "class_type": "BREAK",
            "start_time": "13:00:00",
            "end_time": "13:40:00"
        })

    if curr_slot.get("status") == "AFTER_CLASS":
        return jsonify({
            "success": True,
            "status": "AFTER_CLASS",
            "message": "College hours ended for today (Classes end at 16:10)",
            "period_number": 0,
            "subject": "College Hours Ended (after 4:10 PM)",
            "faculty": "N/A",
            "class_type": "OFF",
            "start_time": "16:10:00",
            "end_time": "23:59:59"
        })

    return jsonify({
        "success": True,
        "status": "ACTIVE",
        "period_number": curr_slot.get("period_number", 1),
        "subject": curr_slot.get("subject", "N/A"),
        "faculty": curr_slot.get("faculty_name", "N/A"),
        "class_type": curr_slot.get("class_type", "THEORY"),
        "start_time": str(curr_slot.get("start_time")),
        "end_time": str(curr_slot.get("end_time"))
    })

@app.route("/api/timetable/schedule", methods=["GET"])
@login_required
def api_timetable_schedule():
    initialize_components()
    dept = request.args.get("department", "CSD").strip().upper()
    sec = request.args.get("section", "B").strip().upper()
    day = request.args.get("day")
    if not day:
        weekday_map = {0: "MON", 1: "TUE", 2: "WED", 3: "THU", 4: "FRI", 5: "SAT", 6: "SUN"}
        day = weekday_map.get(datetime.now().weekday(), "FRI")
    elif day.strip().upper() == "ALL":
        day = None
    else:
        day = day.strip().upper()

    tt = db_manager.get_timetable(dept, sec, day=day)
    return jsonify({"success": True, "day": day or "ALL", "timetable": tt})

@app.route("/api/timetable/entries", methods=["POST"])
@app.route("/api/timetable/entries/<int:entry_id>", methods=["DELETE"])
@login_required
def api_timetable_entries(entry_id=None):
    initialize_components()
    if request.method == "DELETE":
        ok = db_manager.delete_timetable_entry(entry_id)
        return jsonify({"success": ok, "message": "Entry deleted" if ok else "Delete failed"})
    data = request.json or request.form or {}
    ok, msg = db_manager.add_timetable_entry(data)
    return jsonify({"success": ok, "message": msg})


# ── Anti-Brute-Force Rate Limiting & Account Protection ─────────────
_failed_login_attempts: Dict[str, List[float]] = {}
_ip_lockouts: Dict[str, float] = {}
_MAX_FAILED_LOGINS = 5
_LOCKOUT_WINDOW_SECONDS = 300   # 5 minutes window
_LOCKOUT_DURATION_SECONDS = 600 # 10 minutes lockout


def _get_request_client_ip() -> str:
    """Safely extract client IP behind Cloudflare, Nginx, or Direct."""
    return (
        request.headers.get("CF-Connecting-IP")
        or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or request.remote_addr
        or "127.0.0.1"
    )


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if session.get("logged_in"): return redirect(url_for("dashboard"))
    client_ip = _get_request_client_ip()
    now = time.time()

    # 1. Check if IP is currently under lockout
    if client_ip in _ip_lockouts:
        if now < _ip_lockouts[client_ip]:
            remaining_mins = int((_ip_lockouts[client_ip] - now) / 60) + 1
            return render_template(
                "login.html",
                error=f"Security Lockout: Too many failed login attempts. IP temporarily locked for {remaining_mins} minute(s)."
            ), 429
        else:
            del _ip_lockouts[client_ip]
            _failed_login_attempts.pop(client_ip, None)

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        u = authenticate_user(username, password)
        if u:
            # Login successful: reset failed count
            _failed_login_attempts.pop(client_ip, None)
            login_user(u)
            return redirect(url_for("dashboard"))

        # Login failed: record attempt timestamp
        attempts = [t for t in _failed_login_attempts.get(client_ip, []) if now - t < _LOCKOUT_WINDOW_SECONDS]
        attempts.append(now)
        _failed_login_attempts[client_ip] = attempts

        if len(attempts) >= _MAX_FAILED_LOGINS:
            _ip_lockouts[client_ip] = now + _LOCKOUT_DURATION_SECONDS
            logger.warning("SECURITY ALERT: Brute-force attack mitigated. IP %s locked out for 10 minutes.", client_ip)
            return render_template(
                "login.html",
                error="Security Alert: 5 consecutive failed attempts detected. Access from your IP is locked for 10 minutes."
            ), 429

        remaining = _MAX_FAILED_LOGINS - len(attempts)
        return render_template(
            "login.html",
            error=f"Invalid credentials. ({remaining} attempt(s) remaining before security lockout)"
        )

    return render_template("login.html", error=None)

@app.route("/logout")
def logout(): logout_user(); return redirect(url_for("login_page"))

@app.route("/")
@app.route("/dashboard")
@login_required
def dashboard():
    initialize_components(); u = get_current_user()
    return render_template("dashboard.html", department_name=u["department_name"], department_code=u["department_code"], username=u["username"], is_admin=(u["role"]=="admin"), departments=Config.get_active_department_codes(db_manager))

@app.route("/video_feed")
@login_required
def video_feed():
    initialize_components()
    dept = request.args.get("dept", get_current_user()["department_code"]).upper()
    sec = request.args.get("section", "B")
    cam = request.args.get("cam", "Default")
    return Response(
        generate_mjpeg_frames(dept, section=sec, cam_name=cam),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

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
def api_get_evidence_photo(attendance_id: int):
    """Endpoint to serve face recognition evidence image."""
    initialize_components()
    user = get_current_user()

    # Query attendance record
    rec = db_manager.get_attendance_by_id(attendance_id)
    if not rec:
        return jsonify({"error": "Attendance record not found"}), 404

    dept = rec.get("department", "CSD").upper()
    if user and not is_admin() and user.get("department_code") != dept:
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


@app.route("/student")
def student_portal_page():
    """Public, mobile-friendly student self-service portal."""
    return render_template("student_portal.html")


@app.route("/api/student/<student_id>/attendance-summary")
def api_student_attendance_summary(student_id: str):
    """Public read-only endpoint returning attendance metrics and history for student."""
    initialize_components()
    s_id = (student_id or "").strip().upper()
    if not s_id:
        return jsonify({"success": False, "error": "Invalid Student ID"}), 400

    summary = db_manager.get_student_attendance_summary(s_id)
    if not summary:
        return jsonify({"success": False, "error": f"No attendance data found for student {s_id}."}), 404

    return jsonify(summary)


@app.route("/api/analytics/occupancy")
@login_required
def api_analytics_occupancy():
    """Real-time classroom occupancy & 2D spatial heatmap analytics."""
    initialize_components()
    dept = request.args.get("dept", "CSD").strip().upper()
    sec = request.args.get("section", "B").strip().upper()
    cam = request.args.get("cam", "Default").strip()

    key = (dept, sec, cam)
    worker = stream_broadcaster._workers.get(key)
    in_room_count = 0
    spatial = {
        "front_row": 0,    # y > 60% (closer to front CCTV camera)
        "middle_row": 0,   # 30% < y <= 60%
        "back_row": 0,     # y <= 30% (farther back benches)
        "left_wing": 0,    # x < 40%
        "center_wing": 0,  # 40% <= x <= 60%
        "right_wing": 0,   # x > 60%
    }
    detected_students = []

    if worker and worker.last_results:
        in_room_count = len(worker.last_results)
        frame_h, frame_w = 480, 640
        if worker.last_frame is not None:
            frame_h, frame_w = worker.last_frame.shape[:2]

        for res, _ in worker.last_results:
            top, right, bottom, left = res.face_location
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0

            x_pct = center_x / frame_w
            y_pct = center_y / frame_h

            # Spatial mapping
            if y_pct > 0.60:
                spatial["front_row"] += 1
            elif y_pct > 0.30:
                spatial["middle_row"] += 1
            else:
                spatial["back_row"] += 1

            if x_pct < 0.40:
                spatial["left_wing"] += 1
            elif x_pct <= 0.60:
                spatial["center_wing"] += 1
            else:
                spatial["right_wing"] += 1

            detected_students.append({
                "student_id": res.student_id,
                "name": res.student_name,
                "is_recognized": res.is_recognized,
                "confidence": res.confidence_tier,
                "is_live": getattr(res, "is_live", True),
                "grid_x": round(x_pct * 100, 1),
                "grid_y": round(y_pct * 100, 1)
            })

    # Registered student count
    engine = face_engine_manager.get_engine(dept)
    reg_count = engine.get_registered_count() if engine else 16
    occupancy_pct = round((in_room_count / reg_count * 100), 1) if reg_count > 0 else 0.0

    return jsonify({
        "success": True,
        "department": dept,
        "section": sec,
        "in_room_count": in_room_count,
        "registered_count": reg_count,
        "occupancy_percentage": min(100.0, occupancy_pct),
        "spatial": spatial,
        "detected_students": detected_students,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })


def _format_time(t):
    if hasattr(t, "total_seconds"):
        s = int(t.total_seconds()); return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"
    return str(t)


if __name__ == "__main__":
    initialize_components()
    is_prod = config.ENVIRONMENT == "production"
    app.run(host="0.0.0.0", port=config.PORT, debug=not is_prod, threaded=True)

