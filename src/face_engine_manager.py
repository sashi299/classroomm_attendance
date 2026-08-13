"""
face_engine_manager.py - Manages FaceEngine instances per department and photo registry.
"""

import os
import logging
import shutil
import cv2
import numpy as np
import face_recognition
from typing import Dict, List, Optional, Tuple, Union

from face_engine import FaceEngine
from database import DatabaseManager
from config import Config

logger = logging.getLogger(__name__)

def get_iou(boxA, boxB):
    tA, rA, bA, lA = boxA; tB, rB, bB, lB = boxB
    yA = max(tA, tB); xA = max(lA, lB); yB = min(bA, bB); xB = min(rA, rB)
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (rA - lA) * (bA - tA); areaB = (rB - lB) * (bB - tB)
    union = float(areaA + areaB - inter)
    return inter / union if union > 0 else 0

class FaceEngineManager:
    """Orchestrates multiple FaceEngine instances and student photo management."""

    def __init__(self, base_dir: str, recognition_threshold: float = 0.5):
        self.base_dir = base_dir
        self.recognition_threshold = recognition_threshold
        self.engines: Dict[str, FaceEngine] = {}

    def get_engine(self, department_code: str) -> FaceEngine:
        if department_code not in self.engines:
            dept_dir = os.path.join(self.base_dir, department_code)
            os.makedirs(dept_dir, exist_ok=True)
            logger.info("Initializing FaceEngine for department [%s] from %s", department_code, dept_dir)
            engine = FaceEngine(known_students_dir=dept_dir, recognition_threshold=self.recognition_threshold)
            engine.load_registered_students()
            self.engines[department_code] = engine
        return self.engines[department_code]

    def reload_engine(self, department_code: str):
        if department_code in self.engines:
            logger.info("Reloading FaceEngine for department [%s]", department_code)
            self.engines[department_code].load_registered_students()

    def add_student_photo(self, department_code: str, student_id: str, student_name: str, file_obj, filename: str) -> Tuple[bool, str]:
        """Validate photo for exactly one face and save to department registry."""
        try:
            if hasattr(file_obj, "read"):
                file_bytes = file_obj.read()
            elif isinstance(file_obj, str) and os.path.isfile(file_obj):
                with open(file_obj, "rb") as f: file_bytes = f.read()
            else: return False, "Invalid file."

            if not file_bytes: return False, "Empty file."
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None: return False, "Invalid image format."

            h, w = img.shape[:2]; rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            raw_locs = face_recognition.face_locations(rgb_img, number_of_times_to_upsample=0)

            final_faces = []; final_encs = []
            sorted_locs = sorted(raw_locs, key=lambda x: (x[2]-x[0])*(x[1]-x[3]), reverse=True)
            for f in sorted_locs:
                if (f[2]-f[0]) < (h * 0.12): continue # Ignore small noise
                encs = face_recognition.face_encodings(rgb_img, [f])
                if not encs: continue

                is_dup = False
                for existing in final_faces:
                    if get_iou(f, existing) > 0.3: is_dup = True; break
                if not is_dup:
                    final_faces.append(f); final_encs.append(encs[0])

            if len(final_faces) == 0: return False, "No face detected. Center your face."
            if len(final_faces) > 1: return False, f"Multiple faces ({len(final_faces)}) detected."

            # Save
            dest_dir = os.path.join(self.base_dir, department_code, f"{student_id}_{student_name}")
            os.makedirs(dest_dir, exist_ok=True)
            with open(os.path.join(dest_dir, filename), "wb") as f: f.write(file_bytes)

            self.reload_engine(department_code)
            return True, "Success"
        except Exception as e:
            logger.error("Error adding student photo: %s", e)
            return False, str(e)

    def delete_student(self, department_code: str, student_id: str):
        dept_dir = os.path.join(self.base_dir, department_code)
        if not os.path.exists(dept_dir): return
        for item in os.listdir(dept_dir):
            if item.startswith(f"{student_id}_"):
                shutil.rmtree(os.path.join(dept_dir, item))
                logger.info("Deleted student folder: %s", item)
        self.reload_engine(department_code)

    def get_student_details(self, department_code: str, db_manager: DatabaseManager) -> List[dict]:
        dept_code = (department_code or "CSD").strip().upper()
        if dept_code == "ALL":
            merged = []
            active_depts = getattr(Config, "DEFAULT_DEPARTMENT_CODES", ["CSD", "CSM", "CSE", "CSC", "MECH", "CIVIL", "EEE", "ECE"])
            for d in active_depts:
                merged.extend(self.get_student_details(d, db_manager))
            return merged

        engine = self.get_engine(dept_code)
        registered_info = engine.get_registered_students_info() if engine else []
        db_students = db_manager.get_students(department=dept_code) if db_manager else []
        db_map = {str(s["student_id"]): s for s in db_students}

        merged = []
        seen_sids = set()

        for info in registered_info:
            sid = str(info["student_id"])
            seen_sids.add(sid)
            student_data = db_map.get(sid, {"year_level": "N/A", "section": "N/A", "academic_year": "N/A", "semester": "N/A", "is_active": True})
            is_act = student_data.get("is_active", True)
            merged.append({
                "student_id": sid, "student_name": info["student_name"],
                "photo_count": info["photo_count"], "source_file": info.get("source_file", "-"),
                "year_level": student_data.get("year_level", "N/A"), "section": student_data.get("section", "B"),
                "academic_year": student_data.get("academic_year", "N/A"), "semester": student_data.get("semester", "N/A"),
                "status": "Active" if is_act else "Inactive", "department_code": dept_code, "department": dept_code
            })

        for s in db_students:
            sid = str(s["student_id"])
            if sid not in seen_sids:
                is_act = s.get("is_active", True)
                merged.append({
                    "student_id": sid, "student_name": s["name"],
                    "photo_count": 0, "source_file": "-",
                    "year_level": s.get("year_level", "N/A"), "section": s.get("section", "B"),
                    "academic_year": s.get("academic_year", "N/A"), "semester": s.get("semester", "N/A"),
                    "status": "Active" if is_act else "Inactive", "department_code": dept_code, "department": dept_code
                })

        return merged

    def get_student_photo_path(self, department_code: str, student_id: str) -> Optional[str]:
        """Resolve the absolute filesystem path for a student's primary photo."""
        dept_dir = os.path.join(self.base_dir, department_code)
        if not os.path.exists(dept_dir):
            return None

        # Look for directory {ID}_{NAME}
        for item in os.listdir(dept_dir):
            if item.startswith(f"{student_id}_"):
                full_item_path = os.path.join(dept_dir, item)
                if os.path.isdir(full_item_path):
                    # Pick first image in subfolder
                    imgs = [f for f in os.listdir(full_item_path) if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png"]]
                    if imgs:
                        return os.path.join(full_item_path, sorted(imgs)[0])
                elif os.path.isfile(full_item_path):
                    return full_item_path
        return None
