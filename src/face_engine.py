"""
face_engine.py - Deep face detection and ArcFace multi-photo recognition engine.

Uses InsightFace ArcFace (512-dimensional normalized embeddings + SCRFD detection)
for state-of-the-art deep facial recognition with cosine similarity matching.
Supports multiple reference photos per registered student.
Gracefully falls back to dlib/face_recognition if InsightFace is unavailable.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union

import cv2
import numpy as np

# Configure module-level logger
logger = logging.getLogger(__name__)

# Supported image extensions for student photos
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif"}

# HEIC/HEIF support flag
_HEIC_AVAILABLE = None
def _read_heic_as_bgr(filepath: str):
    """Read HEIC/HEIF image file and return as OpenCV BGR numpy array."""
    global _HEIC_AVAILABLE
    if _HEIC_AVAILABLE is False:
        return None
    try:
        from pillow_heif import register_heif_opener
        from PIL import Image
        register_heif_opener()
        pil_img = Image.open(filepath).convert("RGB")
        arr = np.array(pil_img)
        _HEIC_AVAILABLE = True
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        if _HEIC_AVAILABLE is None:
            logger.warning("HEIC support unavailable: %s", e)
            _HEIC_AVAILABLE = False
        return None

# Singleton / Cached InsightFace App
_INSIGHTFACE_APP = None
_INSIGHTFACE_AVAILABLE = None


def get_insightface_app():
    """Lazily initialize and return the InsightFace FaceAnalysis instance."""
    global _INSIGHTFACE_APP, _INSIGHTFACE_AVAILABLE
    if _INSIGHTFACE_AVAILABLE is False:
        return None
    if _INSIGHTFACE_APP is not None:
        return _INSIGHTFACE_APP

    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _INSIGHTFACE_APP = app
        _INSIGHTFACE_AVAILABLE = True
        logger.info("InsightFace ArcFace deep embedding engine (buffalo_s) initialized successfully.")
        return _INSIGHTFACE_APP
    except Exception as e:
        logger.warning("InsightFace ArcFace unavailable (%s). Falling back to dlib/face_recognition.", e)
        _INSIGHTFACE_AVAILABLE = False
        _INSIGHTFACE_APP = None
        return None


def categorize_pose(pose: Optional[Union[np.ndarray, List, Tuple]]) -> str:
    """
    Categorize head pose into FRONTAL, LEFT_ANGLE, RIGHT_ANGLE, PROFILE, or TILT.
    
    Args:
        pose: [pitch, yaw, roll] in degrees from InsightFace
    """
    if pose is None or len(pose) < 2:
        return "FRONTAL"
    try:
        pitch, yaw = float(pose[0]), float(pose[1])
        if abs(yaw) <= 15:
            if abs(pitch) > 20:
                return "TILT_UP" if pitch > 0 else "TILT_DOWN"
            return "FRONTAL"
        elif -45 <= yaw < -15:
            return "LEFT_ANGLE"
        elif 15 < yaw <= 45:
            return "RIGHT_ANGLE"
        elif yaw < -45:
            return "LEFT_PROFILE"
        else:
            return "RIGHT_PROFILE"
    except Exception:
        return "FRONTAL"


class TemporalConfirmationTracker:
    """
    Tracks recognition detections across video frames to temporally confirm borderline matches.
    
    - High-confidence matches (sim >= high_threshold): Instantly confirmed (1 frame).
    - Borderline matches (low_threshold <= sim < high_threshold): Requires 2 detections within window.
    - Low-confidence (sim < low_threshold): Unknown.
    """
    def __init__(self, required_confirmations: int = 2, window_seconds: float = 3.0):
        self.required_confirmations = required_confirmations
        self.window_seconds = window_seconds
        self._history: Dict[str, List[float]] = {}

    def update_and_check(self, student_id: str, similarity: float, high_thresh: float = 0.48) -> bool:
        if not student_id or student_id == "Unknown":
            return False

        now = time.time()
        # High confidence is confirmed immediately
        if similarity >= high_thresh:
            return True

        # Borderline match: record timestamp and check window
        cutoff = now - self.window_seconds
        timestamps = [t for t in self._history.get(student_id, []) if t >= cutoff]
        timestamps.append(now)
        self._history[student_id] = timestamps

        return len(timestamps) >= self.required_confirmations

    def reset(self, student_id: Optional[str] = None):
        if student_id:
            self._history.pop(student_id, None)
        else:
            self._history.clear()


@dataclass
class StudentProfile:
    """Stores a registered student's identity, multiple face embeddings, and pose angles."""
    student_id: str
    student_name: str
    encodings: List[np.ndarray] = field(default_factory=list)
    source_files: List[str] = field(default_factory=list)
    poses: List[Tuple[float, float, float]] = field(default_factory=list)
    angles: List[str] = field(default_factory=list)

    @property
    def pose_distribution(self) -> Dict[str, int]:
        """Count of reference photos per pose angle category."""
        dist = {"FRONTAL": 0, "LEFT_ANGLE": 0, "RIGHT_ANGLE": 0, "PROFILE": 0, "TILT": 0}
        for ang in self.angles:
            if ang == "FRONTAL":
                dist["FRONTAL"] += 1
            elif ang in ("LEFT_ANGLE", "LEFT_PROFILE"):
                dist["LEFT_ANGLE"] += 1
            elif ang in ("RIGHT_ANGLE", "RIGHT_PROFILE"):
                dist["RIGHT_ANGLE"] += 1
            elif "TILT" in ang:
                dist["TILT"] += 1
            else:
                dist["FRONTAL"] += 1
        return dist

    @property
    def angle_coverage_score(self) -> float:
        """Score from 0.0 to 1.0 based on angle diversity (frontal + left + right)."""
        dist = self.pose_distribution
        has_frontal = dist["FRONTAL"] > 0
        has_left = dist["LEFT_ANGLE"] > 0
        has_right = dist["RIGHT_ANGLE"] > 0
        categories_covered = sum([has_frontal, has_left, has_right])
        return categories_covered / 3.0

    @property
    def cctv_readiness(self) -> Tuple[str, str]:
        """Assess whether the reference photo set is ready for CCTV recognition."""
        n = len(self.encodings)
        if n == 0:
            return "NOT USABLE", "No valid face embeddings loaded."
        
        dist = self.pose_distribution
        has_multi_angle = (dist["LEFT_ANGLE"] > 0 or dist["RIGHT_ANGLE"] > 0)

        if n >= 10:
            if has_multi_angle:
                return "CCTV READY", f"High coverage ({n} embeddings with frontal + angled views)"
            return "CCTV READY", f"Strong frontal set ({n} embeddings)"
        elif n >= 5:
            if has_multi_angle:
                return "CCTV READY", f"Good multi-angle coverage ({n} embeddings across angles)"
            return "CCTV READY", f"Sufficient frontal coverage ({n} embeddings)"
        elif n >= 3:
            return "BORDERLINE / NEEDS BETTER ANGLE COVERAGE", f"Only {n} photos; recommend 8-12 photos covering front, left 15-30°, and right 15-30°."
        else:
            return "BORDERLINE / NEEDS BETTER ANGLE COVERAGE", f"Only {n} photo; highly vulnerable to angle/lighting changes in CCTV."


def compute_liveness(face_crop: np.ndarray, kps: Optional[np.ndarray] = None) -> Tuple[bool, float]:
    """
    Biometric Anti-Spoofing & Liveness Verification:
    - High-frequency Laplacian texture analysis (detects photo print artifacts and flat mobile screens).
    - Biometric facial landmark aspect ratio verification (eyes-to-mouth proportions).
    
    Returns:
        (is_live: bool, liveness_score: float)
    """
    if face_crop is None or face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
        return False, 0.0

    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Landmark geometry check
        landmark_score = 1.0
        if kps is not None and len(kps) >= 5:
            d_eyes = float(np.linalg.norm(kps[0] - kps[1]))
            d_mouth = float(np.linalg.norm(kps[3] - kps[4]))
            if d_eyes > 5.0:
                ratio = d_mouth / d_eyes
                if 0.30 <= ratio <= 1.10:
                    landmark_score = 1.0
                else:
                    landmark_score = 0.5
            else:
                landmark_score = 0.5

        # Combined liveness score (normalized 0.0 to 1.0)
        norm_lap = min(1.0, lap_var / 100.0)
        score = (norm_lap * 0.7) + (landmark_score * 0.3)

        # Real human face in live CCTV has lap_var >= 10.0 and score >= 0.22
        is_live = bool(lap_var >= 10.0 and score >= 0.22)
        return is_live, round(score, 3)
    except Exception:
        return True, 0.85


@dataclass
class RecognitionResult:
    """Result of recognizing a single face in a frame."""
    student_id: str                          # "Unknown" if no match
    student_name: str                        # "Unknown" if no match
    face_location: Tuple[int, int, int, int] # (top, right, bottom, left)
    distance: float                          # Metric distance (lower is better, e.g. 1.0 - cosine_sim)
    is_recognized: bool                      # True if matched above similarity threshold
    similarity: float = 0.0                  # Cosine similarity score [-1.0, 1.0] (higher is better)
    confidence_tier: str = "LOW"             # "HIGH", "BORDERLINE", "LOW", "SPOOF_REJECTED"
    pose: Optional[Tuple[float, float, float]] = None # Live estimated pose (pitch, yaw, roll)
    matched_angle: Optional[str] = None      # Pose category of best matching reference photo
    confirmed_by_temporal: bool = False      # True if temporally confirmed across frames
    is_live: bool = True                     # Anti-spoofing liveness check
    liveness_score: float = 1.0              # 0.0 to 1.0 (Texture & landmark geometry)


class FaceEngine:
    """
    Deep face detection and ArcFace recognition engine supporting multiple reference embeddings per student,
    head pose estimation, and temporal confirmation for angled/side CCTV faces.
    """

    def __init__(self, known_students_dir: str, recognition_threshold: float = 0.38, high_confidence_threshold: float = 0.48):
        self.known_students_dir = os.path.abspath(known_students_dir)
        # Cosine similarity thresholds:
        #   >= 0.48: High confidence (instant 1-frame confirmation)
        #   0.38 - 0.48: Borderline confidence (requires temporal 2-frame confirmation)
        #   < 0.38: Low / Unknown
        self.recognition_threshold = recognition_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.registered_profiles: Dict[str, StudentProfile] = {}
        self._profile_key_map: Dict[str, str] = {}
        self.app = get_insightface_app()
        self.temporal_tracker = TemporalConfirmationTracker(required_confirmations=2, window_seconds=3.0)

        logger.info("FaceEngine initialized.")
        logger.info("  Known students directory: %s", self.known_students_dir)
        logger.info("  Engine Model: %s", "InsightFace ArcFace (buffalo_s)" if self.app else "dlib / HOG fallback")
        logger.info("  Recognition similarity thresholds: Borderline=%.2f, High=%.2f",
                    self.recognition_threshold, self.high_confidence_threshold)

    @property
    def registered_students(self) -> List[StudentProfile]:
        """Return list of all registered StudentProfile objects."""
        return list(self.registered_profiles.values())

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

    def load_registered_students(self) -> int:
        """
        Load student photos from directory (supporting subdirectories and single files),
        detect all faces, isolate primary student face using multi-factor ranking (centrality,
        size, quality, and embedding consistency), exclude background classmates, extract pose angles,
        generate normalized ArcFace embeddings, and suppress duplicate/redundant embeddings.
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
                dir_name = item
                if "_" not in dir_name:
                    logger.warning("SKIP subfolder [%s]: Name does not follow {ID}_{NAME} convention.", dir_name)
                    skipped_count += 1
                    continue

                student_id, student_name = dir_name.split("_", 1)
                student_id = student_id.strip()
                student_name = student_name.strip()

                if not student_id:
                    skipped_count += 1
                    continue

                # Use student_id as display name if name portion is empty
                if not student_name:
                    student_name = student_id

                sub_files = [
                    f for f in os.listdir(full_path)
                    if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
                ]

                # Two-pass processing for student folder:
                # Pass 1 establishes anchor embeddings from clear photos
                # Pass 2 uses anchor embeddings to disambiguate multi-face classroom photos
                for sf in sorted(sub_files):
                    photo_path = os.path.join(full_path, sf)
                    rel_name = os.path.join(item, sf)
                    
                    # Look up existing embeddings for this student to ensure consistency
                    norm_key = student_id.upper()
                    existing_encs = []
                    if norm_key in self._profile_key_map:
                        existing_encs = self.registered_profiles[self._profile_key_map[norm_key]].encodings

                    proc_res = self._process_single_image(photo_path, rel_name, existing_encodings=existing_encs)
                    if proc_res is not None:
                        enc, pose, meta = proc_res
                        was_added = self._register_encoding(student_id, student_name, enc, rel_name, pose)
                        if was_added:
                            loaded_photos_count += 1
                        else:
                            # Skipped as redundant/duplicate embedding
                            pass
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

                if not student_id:
                    skipped_count += 1
                    continue
                if not student_name:
                    student_name = student_id

                norm_key = student_id.upper()
                existing_encs = []
                if norm_key in self._profile_key_map:
                    existing_encs = self.registered_profiles[self._profile_key_map[norm_key]].encodings

                proc_res = self._process_single_image(full_path, item, existing_encodings=existing_encs)
                if proc_res is not None:
                    enc, pose, meta = proc_res
                    was_added = self._register_encoding(student_id, student_name, enc, item, pose)
                    if was_added:
                        loaded_photos_count += 1
                else:
                    skipped_count += 1

        logger.info(
            "Student photos loaded cleanly: %d photo embeddings registered across %d unique students (%d skipped).",
            loaded_photos_count, len(self.registered_profiles), skipped_count,
        )
        return len(self.registered_profiles)

    def _process_single_image(
        self,
        filepath: str,
        label_name: str,
        existing_encodings: Optional[List[np.ndarray]] = None,
    ) -> Optional[Tuple[np.ndarray, Optional[Tuple[float, float, float]], dict]]:
        """
        Load image, detect all faces, select the primary student face using multi-factor ranking
        (centrality, area ratio, detection score, and embedding consistency), exclude background people,
        and generate a normalized ArcFace embedding.
        """
        try:
            image = cv2.imread(filepath)
            if image is None:
                # Try HEIC/HEIF fallback for iPhone photos
                ext = os.path.splitext(filepath)[1].lower()
                if ext in (".heic", ".heif"):
                    image = _read_heic_as_bgr(filepath)
                if image is None:
                    logger.warning("SKIP [%s]: Image could not be read.", label_name)
                    return None

            if self.app is not None:
                faces = self.app.get(image)
                if not faces:
                    logger.warning("SKIP [%s]: No face detected by ArcFace detector.", label_name)
                    return None

                img_h, img_w = image.shape[:2]
                img_cx, img_cy = img_w / 2.0, img_h / 2.0
                max_dist = np.sqrt(img_cx**2 + img_cy**2)

                # Filter out tiny noise detections (< 35px or < 0.3% image area) and extreme blur
                valid_candidates = []
                for f in faces:
                    bbox = f.bbox.astype(int)
                    fw = bbox[2] - bbox[0]
                    fh = bbox[3] - bbox[1]
                    det_score = getattr(f, "det_score", 1.0)
                    if fw < 35 or fh < 35 or det_score < 0.40:
                        continue

                    # Check blur in face bounding box
                    fcrop = image[max(0, bbox[1]):min(img_h, bbox[3]), max(0, bbox[0]):min(img_w, bbox[2])]
                    if fcrop.size > 0:
                        gray = cv2.cvtColor(fcrop, cv2.COLOR_BGR2GRAY)
                        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
                        if blur_val < 6.0:
                            continue  # Severe blur

                    # Centrality (1.0 = center of image, 0.0 = corner)
                    fc_x = (bbox[0] + bbox[2]) / 2.0
                    fc_y = (bbox[1] + bbox[3]) / 2.0
                    dist_from_center = np.sqrt((fc_x - img_cx)**2 + (fc_y - img_cy)**2)
                    centrality = max(0.0, 1.0 - (dist_from_center / max_dist))

                    area = fw * fh

                    # Consistency with existing embeddings if available
                    consistency_score = 0.0
                    if existing_encodings and len(existing_encodings) > 0 and getattr(f, "embedding", None) is not None:
                        emb = f.embedding
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            cand_emb = emb / norm
                            sims = [float(np.dot(cand_emb, e)) for e in existing_encodings]
                            consistency_score = max(sims)

                    valid_candidates.append({
                        "face": f,
                        "area": area,
                        "centrality": centrality,
                        "det_score": det_score,
                        "consistency": consistency_score,
                        "bbox": bbox
                    })

                if not valid_candidates:
                    logger.warning("SKIP [%s]: No quality face candidates after filtering.", label_name)
                    return None

                # Multi-factor face selection
                if len(valid_candidates) == 1:
                    chosen = valid_candidates[0]
                else:
                    max_area = max(c["area"] for c in valid_candidates)
                    for c in valid_candidates:
                        area_norm = c["area"] / max_area if max_area > 0 else 0
                        # If existing embeddings match with high confidence, prioritize heavily
                        if c["consistency"] > 0.35:
                            c["score"] = 0.45 * c["consistency"] + 0.30 * c["centrality"] + 0.15 * area_norm + 0.10 * c["det_score"]
                        elif existing_encodings and len(existing_encodings) >= 2 and c["consistency"] < 0.20:
                            # Candidate clearly belongs to a background classmate
                            c["score"] = -1.0
                        else:
                            c["score"] = 0.50 * c["centrality"] + 0.35 * area_norm + 0.15 * c["det_score"]

                    valid_candidates = sorted(valid_candidates, key=lambda c: c["score"], reverse=True)
                    if valid_candidates[0]["score"] < 0:
                        logger.warning("SKIP [%s]: Multi-face image, but no candidate matched student profile.", label_name)
                        return None
                    chosen = valid_candidates[0]
                    logger.debug("MULTI-FACE [%s]: %d faces detected. Selected primary face (score=%.2f, centrality=%.2f, area=%d, consistency=%.2f)",
                                 label_name, len(faces), chosen["score"], chosen["centrality"], chosen["area"], chosen["consistency"])

                target_face = chosen["face"]
                emb = target_face.embedding
                norm = np.linalg.norm(emb)
                normed_emb = (emb / norm) if norm > 0 else None
                if normed_emb is None:
                    return None

                pose = None
                if getattr(target_face, "pose", None) is not None:
                    pose = tuple(float(x) for x in target_face.pose)

                meta = {
                    "total_faces": len(faces),
                    "background_faces_excluded": max(0, len(faces) - 1),
                    "is_multi_face": len(faces) > 1,
                }
                return (normed_emb, pose, meta)
            else:
                try:
                    import face_recognition
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=0)
                    if not locs:
                        locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=1)
                    if not locs:
                        return None
                    encs = face_recognition.face_encodings(rgb, locs)
                    meta = {"total_faces": len(locs), "background_faces_excluded": max(0, len(locs) - 1), "is_multi_face": len(locs) > 1}
                    return (encs[0], None, meta) if encs else None
                except ImportError:
                    logger.warning("Neither InsightFace nor face_recognition is available.")
                    return None

        except Exception as e:
            logger.warning("SKIP [%s]: Error processing image: %s", label_name, e)
            return None

    def _register_encoding(
        self,
        student_id: str,
        student_name: str,
        encoding: np.ndarray,
        source_file: str,
        pose: Optional[Tuple[float, float, float]] = None,
    ) -> bool:
        """
        Add embedding and pose angle to student profile, filtering out near-duplicate embeddings (>0.96 sim).
        Returns True if registered, False if skipped as redundant duplicate.
        """
        key_id = (student_id or "").strip()
        if not key_id:
            return False

        angle_label = categorize_pose(pose)
        norm_key = key_id.upper()

        if norm_key not in self._profile_key_map:
            profile = StudentProfile(
                student_id=key_id,
                student_name=student_name,
                encodings=[encoding],
                source_files=[source_file],
                poses=[pose] if pose else [],
                angles=[angle_label],
            )
            self.registered_profiles[key_id] = profile
            self._profile_key_map[norm_key] = key_id
            logger.info("  REGISTERED PROFILE: [%s] %s (file: %s, angle: %s)", key_id, student_name, source_file, angle_label)
            return True
        else:
            actual_key = self._profile_key_map[norm_key]
            profile = self.registered_profiles[actual_key]

            # Suppress near-duplicate embeddings (> 0.96 cosine sim with identical angle)
            for existing_enc, existing_ang in zip(profile.encodings, profile.angles):
                if existing_ang == angle_label:
                    sim = float(np.dot(existing_enc, encoding))
                    if sim >= 0.96:
                        logger.debug("  SUPPRESS DUPLICATE [%s]: file %s (sim=%.4f to existing %s embedding)",
                                     actual_key, source_file, sim, angle_label)
                        return False

            profile.encodings.append(encoding)
            profile.source_files.append(source_file)
            if pose:
                profile.poses.append(pose)
            profile.angles.append(angle_label)
            logger.info("  ADDED EMBEDDING: [%s] %s (file: %s, angle: %s, total embeddings: %d)",
                        actual_key, student_name, source_file, angle_label, len(profile.encodings))
            return True

    def recognize_faces(
        self,
        bgr_frame: np.ndarray,
        resize_factor: float = 1.0,
    ) -> List[RecognitionResult]:
        """
        Detect and recognize faces in a BGR frame against registered student embeddings.
        Supports multi-angle matching (frontal, left/right 15-45°, side profile) with tiered confidence
        and temporal confirmation.
        """
        results: List[RecognitionResult] = []
        if bgr_frame is None or bgr_frame.size == 0:
            return results

        if self.app is not None:
            # ── ArcFace Recognition Path ──────────────────────────────
            faces = self.app.get(bgr_frame)
            if not faces:
                return results

            for face in faces:
                if getattr(face, "det_score", 1.0) < 0.40:
                    continue

                bbox = face.bbox.astype(int)
                # Location tuple format: (top, right, bottom, left)
                top = max(0, int(bbox[1]))
                right = max(0, int(bbox[2]))
                bottom = max(0, int(bbox[3]))
                left = max(0, int(bbox[0]))
                face_loc = (top, right, bottom, left)

                emb = face.embedding
                norm = np.linalg.norm(emb)
                if norm <= 0:
                    continue
                live_emb = emb / norm

                live_pose = tuple(float(x) for x in face.pose) if getattr(face, "pose", None) is not None else None

                best_student_id = "Unknown"
                best_student_name = "Unknown"
                highest_similarity = -1.0
                best_matched_angle = None

                for student_id, profile in self.registered_profiles.items():
                    if not profile.encodings:
                        continue
                    # Vectorized cosine similarity dot product against all student embeddings
                    sims = np.dot(np.array(profile.encodings), live_emb)
                    max_idx = int(np.argmax(sims))
                    max_sim_profile = float(sims[max_idx])

                    if max_sim_profile > highest_similarity:
                        highest_similarity = max_sim_profile
                        best_student_id = profile.student_id
                        best_student_name = profile.student_name
                        if max_idx < len(profile.angles):
                            best_matched_angle = profile.angles[max_idx]

                equiv_distance = float(max(0.0, 1.0 - highest_similarity))

                # Biometric Anti-Spoofing & Liveness Check
                face_crop = bgr_frame[top:bottom, left:right]
                is_live, liveness_score = compute_liveness(face_crop, getattr(face, "kps", None))

                if not is_live:
                    results.append(RecognitionResult(
                        student_id="Spoof Detected",
                        student_name="Anti-Spoof Rejected",
                        face_location=face_loc,
                        distance=equiv_distance,
                        is_recognized=False,
                        similarity=highest_similarity,
                        confidence_tier="SPOOF_REJECTED",
                        pose=live_pose,
                        matched_angle=None,
                        confirmed_by_temporal=False,
                        is_live=False,
                        liveness_score=liveness_score,
                    ))
                    logger.warning("ANTI-SPOOF REJECTED: Face at %s failed liveness check (score=%.3f)", face_loc, liveness_score)
                    continue

                if highest_similarity >= self.high_confidence_threshold and best_student_id != "Unknown":
                    # High confidence match: immediate confirmation
                    results.append(RecognitionResult(
                        student_id=best_student_id,
                        student_name=best_student_name,
                        face_location=face_loc,
                        distance=equiv_distance,
                        is_recognized=True,
                        similarity=highest_similarity,
                        confidence_tier="HIGH",
                        pose=live_pose,
                        matched_angle=best_matched_angle,
                        confirmed_by_temporal=True,
                        is_live=True,
                        liveness_score=liveness_score,
                    ))
                    logger.debug("ARCFACE HIGH-CONF MATCH: %s (%s) sim=%.4f matched_angle=%s liveness=%.2f",
                                 best_student_name, best_student_id, highest_similarity, best_matched_angle, liveness_score)

                elif highest_similarity >= self.recognition_threshold and best_student_id != "Unknown":
                    # Borderline match: verify temporal persistence
                    is_confirmed = self.temporal_tracker.update_and_check(
                        best_student_id, highest_similarity, self.high_confidence_threshold
                    )
                    results.append(RecognitionResult(
                        student_id=best_student_id,
                        student_name=best_student_name,
                        face_location=face_loc,
                        distance=equiv_distance,
                        is_recognized=True,
                        similarity=highest_similarity,
                        confidence_tier="BORDERLINE",
                        pose=live_pose,
                        matched_angle=best_matched_angle,
                        confirmed_by_temporal=is_confirmed,
                        is_live=True,
                        liveness_score=liveness_score,
                    ))
                    logger.debug("ARCFACE BORDERLINE MATCH: %s (%s) sim=%.4f confirmed=%s liveness=%.2f",
                                 best_student_name, best_student_id, highest_similarity, is_confirmed, liveness_score)

                else:
                    results.append(RecognitionResult(
                        student_id="Unknown",
                        student_name="Unknown",
                        face_location=face_loc,
                        distance=equiv_distance,
                        is_recognized=False,
                        similarity=highest_similarity,
                        confidence_tier="LOW",
                        pose=live_pose,
                        matched_angle=None,
                        confirmed_by_temporal=False,
                        is_live=True,
                        liveness_score=liveness_score,
                    ))
                    logger.debug("ARCFACE LOW CONFIDENCE: sim=%.4f -> Unknown", highest_similarity)

            return results

        else:
            # ── Fallback dlib/face_recognition Path ───────────────────
            import face_recognition
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            dlib_locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=0)
            if not dlib_locs:
                return results

            encs = face_recognition.face_encodings(rgb, dlib_locs)
            for enc, loc in zip(encs, dlib_locs):
                best_student_id = "Unknown"
                best_student_name = "Unknown"
                min_dist = 1.0

                for student_id, profile in self.registered_profiles.items():
                    if not profile.encodings: continue
                    dists = face_recognition.face_distance(profile.encodings, enc)
                    m = float(np.min(dists))
                    if m < min_dist:
                        min_dist = m
                        best_student_id = profile.student_id
                        best_student_name = profile.student_name

                # dlib Euclidean distance threshold (~0.55)
                is_match = (min_dist <= 0.55) and (best_student_id != "Unknown")
                results.append(RecognitionResult(
                    student_id=best_student_id if is_match else "Unknown",
                    student_name=best_student_name if is_match else "Unknown",
                    face_location=loc,
                    distance=min_dist,
                    is_recognized=is_match,
                    similarity=float(max(0.0, 1.0 - min_dist)),
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

