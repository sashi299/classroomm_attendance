"""
face_engine.py - Face detection and multi-photo recognition engine.

Loads registered student photos from the students/{DEPT_CODE}/ directory,
supports both single-photo files and multi-photo subdirectories,
generates 128-dimensional face encodings for each valid student image,
and performs vectorized face distance matching against all reference encodings.

Student photo naming & directory conventions:
    1. Single file: {STUDENT_ID}_{STUDENT_NAME}.jpg
       Example: 25a51a4470_Sashi.jpg
    2. Multi-photo subdirectory: {STUDENT_ID}_{STUDENT_NAME}/
       Example: 25a51a4470_Sashi/front.jpg, 25a51a4470_Sashi/left.jpg
    3. Multi-photo file prefix: {STUDENT_ID}_{STUDENT_NAME}_front.jpg
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import face_recognition

# Configure module-level logger
logger = logging.getLogger(__name__)

# Supported image extensions for student photos
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class StudentProfile:
    """Stores a registered student's identity and multiple face encodings."""
    student_id: str
    student_name: str
    encodings: List[np.ndarray] = field(default_factory=list)
    source_files: List[str] = field(default_factory=list)


@dataclass
class RecognitionResult:
    """Result of recognizing a single face in a frame."""
    student_id: str          # "Unknown" if no match
    student_name: str        # "Unknown" if no match
    face_location: Tuple[int, int, int, int]  # (top, right, bottom, left)
    distance: float          # Best face distance (lower = better match)
    is_recognized: bool      # True if matched within threshold


# Cascades for side faces, 45° angle faces, and turned heads
_CASCADES_DIR = os.path.join(os.path.dirname(__file__), "cascades")
_PROFILE_CASCADE = None
_FRONTAL_CASCADE = None
_ALT_CASCADE = None

try:
    p_path = os.path.join(_CASCADES_DIR, "haarcascade_profileface.xml")
    f_path = os.path.join(_CASCADES_DIR, "haarcascade_frontalface_default.xml")
    a_path = os.path.join(_CASCADES_DIR, "haarcascade_frontalface_alt2.xml")

    if os.path.isfile(p_path):
        _PROFILE_CASCADE = cv2.CascadeClassifier(p_path)
    if os.path.isfile(f_path):
        _FRONTAL_CASCADE = cv2.CascadeClassifier(f_path)
    if os.path.isfile(a_path):
        _ALT_CASCADE = cv2.CascadeClassifier(a_path)
except Exception as e:
    logger.warning("Error loading cascades: %s", e)


def _iou_boxes(boxA, boxB):
    tA, rA, bA, lA = boxA
    tB, rB, bB, lB = boxB
    yA = max(tA, tB); xA = max(lA, lB); yB = min(bA, bB); xB = min(rA, rB)
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (rA - lA) * (bA - tA); areaB = (rB - lB) * (bB - tB)
    union = float(areaA + areaB - inter)
    return inter / union if union > 0 else 0


class FaceEngine:
    """
    Face detection and recognition engine supporting multiple reference encodings per student.

    Loads student photos, generates face encodings, and matches
    detected faces in video frames against registered student profiles.
    """

    def __init__(self, known_students_dir: str, recognition_threshold: float = 0.55):
        self.known_students_dir = known_students_dir
        self.recognition_threshold = recognition_threshold
        # Mapping: student_id -> StudentProfile
        self.registered_profiles: Dict[str, StudentProfile] = {}
        # Case-insensitive identity key map: UPPER(student_id) -> primary student_id
        self._profile_key_map: Dict[str, str] = {}

        logger.info("FaceEngine initialized.")
        logger.info("  Known students directory: %s", self.known_students_dir)
        logger.info("  Recognition threshold: %.2f", self.recognition_threshold)

    @property
    def registered_students(self) -> List[StudentProfile]:
        """Return list of all registered StudentProfile objects."""
        return list(self.registered_profiles.values())

    def load_registered_students(self) -> int:
        """
        Load student photos from directory (supporting files & subdirectories),
        detect faces, generate encodings, and group encodings strictly by student ID.

        Returns:
            Number of registered student profiles.
        """
        self.registered_profiles.clear()
        self._profile_key_map.clear()

        if not os.path.isdir(self.known_students_dir):
            logger.error("Known students directory does not exist: %s", self.known_students_dir)
            return 0

        entries = os.listdir(self.known_students_dir)
        if not entries:
            logger.warning("No files or subdirectories found in %s", self.known_students_dir)
            return 0

        logger.info("Scanning student directory: %s", self.known_students_dir)

        loaded_photos_count = 0
        skipped_count = 0

        for item in sorted(entries):
            full_path = os.path.join(self.known_students_dir, item)

            if os.path.isdir(full_path):
                # Subdirectory structure: {STUDENT_ID}_{STUDENT_NAME}/
                dir_name = item
                if "_" not in dir_name:
                    logger.warning("SKIP subfolder [%s]: Name does not follow {ID}_{NAME} convention.", dir_name)
                    skipped_count += 1
                    continue

                student_id, student_name = dir_name.split("_", 1)
                student_id = student_id.strip()
                student_name = student_name.strip()

                if not student_id or not student_name:
                    skipped_count += 1
                    continue

                sub_files = [
                    f for f in os.listdir(full_path)
                    if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
                ]

                for sf in sorted(sub_files):
                    photo_path = os.path.join(full_path, sf)
                    rel_name = os.path.join(item, sf)
                    enc = self._process_single_image(photo_path, rel_name)
                    if enc is not None:
                        self._register_encoding(student_id, student_name, enc, rel_name)
                        loaded_photos_count += 1
                    else:
                        skipped_count += 1

            elif os.path.isfile(full_path):
                ext = os.path.splitext(item)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                name_part = item
                while os.path.splitext(name_part)[1].lower() in SUPPORTED_EXTENSIONS:
                    name_part = os.path.splitext(name_part)[0]

                if "_" not in name_part:
                    logger.warning("SKIP file [%s]: Filename does not follow {ID}_{NAME} convention.", item)
                    skipped_count += 1
                    continue

                parts = name_part.split("_")
                student_id = parts[0].strip()
                student_name = "_".join(parts[1:]).strip()

                if not student_id or not student_name:
                    skipped_count += 1
                    continue

                enc = self._process_single_image(full_path, item)
                if enc is not None:
                    self._register_encoding(student_id, student_name, enc, item)
                    loaded_photos_count += 1
                else:
                    skipped_count += 1

        logger.info(
            "Student photos loaded cleanly: %d photo encodings registered across %d unique students (%d skipped).",
            loaded_photos_count, len(self.registered_profiles), skipped_count,
        )
        return len(self.registered_profiles)

    def _process_single_image(self, filepath: str, label_name: str) -> Optional[np.ndarray]:
        """Load an image, verify exactly ONE face, and generate facial encoding."""
        try:
            image = face_recognition.load_image_file(filepath)
            if image is None:
                logger.warning("SKIP [%s]: Image loaded as None.", label_name)
                return None

            # Use Upsample=0 for consistent behavior with camera enrollment.
            # Large reference photos don't need upsampling, and it reduces background noise.
            face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0)
            if len(face_locations) == 0:
                logger.warning("SKIP [%s]: No face detected in image.", label_name)
                return None

            # Consistent filtering: only count faces that can be encoded
            valid_encodings = face_recognition.face_encodings(image, face_locations)
            if len(valid_encodings) == 0:
                logger.warning("SKIP [%s]: Could not generate face encoding (likely noise).", label_name)
                return None

            if len(valid_encodings) > 1:
                logger.warning(
                    "SKIP [%s]: %d distinct faces detected. Each student photo must contain exactly one face.",
                    label_name, len(valid_encodings),
                )
                return None

            return valid_encodings[0]

        except Exception as e:
            logger.warning("SKIP [%s]: Error processing image: %s", label_name, e)
            return None

    def _register_encoding(self, student_id: str, student_name: str, encoding: np.ndarray, source_file: str):
        """Add an encoding to a student profile (creating profile if not exists, merging by student_id)."""
        key_id = (student_id or "").strip()
        if not key_id:
            return

        norm_key = key_id.upper()

        if norm_key not in self._profile_key_map:
            profile = StudentProfile(
                student_id=key_id,
                student_name=student_name,
                encodings=[encoding],
                source_files=[source_file],
            )
            self.registered_profiles[key_id] = profile
            self._profile_key_map[norm_key] = key_id
            logger.info("  REGISTERED PROFILE: [%s] %s (file: %s)", key_id, student_name, source_file)
        else:
            actual_key = self._profile_key_map[norm_key]
            profile = self.registered_profiles[actual_key]
            profile.encodings.append(encoding)
            profile.source_files.append(source_file)
            logger.info("  ADDED ENCODING: [%s] %s (file: %s, total encodings: %d)",
                        actual_key, student_name, source_file, len(profile.encodings))

    def recognize_faces(
        self,
        bgr_frame: np.ndarray,
        resize_factor: float = 1.0,
    ) -> List[RecognitionResult]:
        """
        Detect and recognize faces in an OpenCV BGR frame against all student encodings.

        Returns:
            List of RecognitionResult objects.
        """
        results: List[RecognitionResult] = []

        if bgr_frame is None or bgr_frame.size == 0:
            return results

        if 0.0 < resize_factor < 1.0:
            small_bgr = cv2.resize(bgr_frame, (0, 0), fx=resize_factor, fy=resize_factor)
            rgb_frame = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
            scale_multiplier = 1.0 / resize_factor
        else:
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            scale_multiplier = 1.0

        # Contrast normalization for Night Vision / IR cameras
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        norm_gray = clahe.apply(gray)
        rgb_frame_enhanced = cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2RGB)
        # 1. Primary dlib face locations for encoding generation
        dlib_face_locations = face_recognition.face_locations(rgb_frame_enhanced, number_of_times_to_upsample=1, model="hog")
        if not dlib_face_locations:
            dlib_face_locations = face_recognition.face_locations(rgb_frame_enhanced, number_of_times_to_upsample=0, model="hog")

        # 2. Side profile face detection fallback for turned heads
        extra_profile_boxes = []
        if _PROFILE_CASCADE is not None and not _PROFILE_CASCADE.empty():
            p_faces_left = _PROFILE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))
            for (x, y, bw, bh) in p_faces_left:
                box = (y, x + bw, y + bh, x)
                if not any(_iou_boxes(box, b) > 0.3 for b in dlib_face_locations):
                    extra_profile_boxes.append(box)

            flipped_gray = cv2.flip(gray, 1)
            p_faces_right = _PROFILE_CASCADE.detectMultiScale(flipped_gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))
            h_g, w_g = gray.shape[:2]
            for (x, y, bw, bh) in p_faces_right:
                orig_x = w_g - (x + bw)
                box = (y, orig_x + bw, y + bh, orig_x)
                if not any(_iou_boxes(box, b) > 0.3 for b in dlib_face_locations) and not any(_iou_boxes(box, b) > 0.3 for b in extra_profile_boxes):
                    extra_profile_boxes.append(box)

        if not dlib_face_locations and not extra_profile_boxes:
            return results

        # Scale coordinates if resized
        if scale_multiplier != 1.0:
            face_locations = [
                (
                    int(round(top * scale_multiplier)),
                    int(round(right * scale_multiplier)),
                    int(round(bottom * scale_multiplier)),
                    int(round(left * scale_multiplier)),
                )
                for top, right, bottom, left in dlib_face_locations
            ]
            profile_locations = [
                (
                    int(round(top * scale_multiplier)),
                    int(round(right * scale_multiplier)),
                    int(round(bottom * scale_multiplier)),
                    int(round(left * scale_multiplier)),
                )
                for top, right, bottom, left in extra_profile_boxes
            ]
        else:
            face_locations = dlib_face_locations
            profile_locations = extra_profile_boxes

        # Generate face encodings strictly on dlib locations
        face_encodings = face_recognition.face_encodings(rgb_frame_enhanced, dlib_face_locations) if dlib_face_locations else []

        if not self.registered_profiles:
            for loc in face_locations + profile_locations:
                results.append(RecognitionResult(
                    student_id="Unknown",
                    student_name="Unknown",
                    face_location=loc,
                    distance=1.0,
                    is_recognized=False,
                ))
            return results

        # Check if frame is B&W / IR Night Vision (low color saturation across channels)
        b_ch, g_ch, r_ch = cv2.split(rgb_frame)
        is_ir_night = bool(np.std(b_ch.astype(np.float32) - g_ch.astype(np.float32)) < 10.0)
        effective_threshold = max(self.recognition_threshold, 0.58) if is_ir_night else self.recognition_threshold

        # Process dlib faces for student profile matching
        for face_encoding, face_location in zip(face_encodings, face_locations):
            best_student_id = "Unknown"
            best_student_name = "Unknown"
            overall_min_distance = 1.0

            # Compare against every registered student profile's encodings
            for student_id, profile in self.registered_profiles.items():
                if not profile.encodings:
                    continue

                distances = face_recognition.face_distance(profile.encodings, face_encoding)
                min_dist_for_profile = float(np.min(distances))

                if min_dist_for_profile < overall_min_distance:
                    overall_min_distance = min_dist_for_profile
                    best_student_id = profile.student_id
                    best_student_name = profile.student_name

            if overall_min_distance <= effective_threshold and best_student_id != "Unknown":
                results.append(RecognitionResult(
                    student_id=best_student_id,
                    student_name=best_student_name,
                    face_location=face_location,
                    distance=overall_min_distance,
                    is_recognized=True,
                ))
                logger.debug("MATCH: %s (%s) distance=%.4f", best_student_name, best_student_id, overall_min_distance)
            else:
                results.append(RecognitionResult(
                    student_id="Unknown",
                    student_name="Unknown",
                    face_location=face_location,
                    distance=overall_min_distance,
                    is_recognized=False,
                ))
                logger.debug("NO MATCH: best distance=%.4f exceeds threshold=%.2f", overall_min_distance, effective_threshold)

        # Add profile cascade faces as UNKNOWN boxes
        for loc in profile_locations:
            results.append(RecognitionResult(
                student_id="Unknown",
                student_name="Unknown",
                face_location=loc,
                distance=1.0,
                is_recognized=False,
            ))

        return results

    def get_registered_count(self) -> int:
        """Return number of unique registered student profiles."""
        return len(self.registered_profiles)

    def get_registered_students_info(self) -> List[dict]:
        """Return list of registered student info dicts with photo source details."""
        info_list = []
        for profile in self.registered_profiles.values():
            info_list.append({
                "student_id": profile.student_id,
                "student_name": profile.student_name,
                "photo_count": len(profile.encodings),
                "source_files": profile.source_files,
                "source_file": profile.source_files[0] if profile.source_files else "",
            })
        return info_list
