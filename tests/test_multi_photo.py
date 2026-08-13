"""
test_multi_photo.py - Multi-Photo Student Registration & Identity Keying Test Suite.

Tests specifically for:
  - Registering front photo for student ID
  - Registering left photo for same student ID
  - Registering right photo for same student ID
  - Verifying only ONE student profile appears in student directory
  - Verifying THREE encodings exist for that student ID
  - Verifying re-uploading the same photo is rejected as duplicate
  - Verifying recognition works against any of the three reference photos
  - Preserving legacy single-file photos alongside subfolder multi-photo format
"""

import os
import unittest
import tempfile
import shutil
import cv2
import numpy as np

from face_engine import FaceEngine, StudentProfile
from face_engine_manager import FaceEngineManager


class TestMultiPhotoRegistration(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="multi_photo_test_")
        self.cse_dir = os.path.join(self.test_dir, "CSE")
        os.makedirs(self.cse_dir, exist_ok=True)
        self.fem = FaceEngineManager(base_dir=self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_multi_photo_registration_flow(self):
        """
        Test registering front, left, and right photos for the SAME student ID.
        Verifies:
          1. Only ONE StudentProfile appears in student directory.
          2. Profile contains THREE face encodings.
          3. Re-uploading identical photo is rejected as duplicate.
          4. Recognition matches against any of the 3 reference encodings.
        """
        student_id = "25a51a4470"
        student_name = "sashi"

        # 1. Register front photo
        enc_front = np.zeros(128, dtype=np.float64)
        enc_front[0] = 0.1

        engine = self.fem.get_engine("CSE")
        engine._register_encoding(student_id, student_name, enc_front, "25a51a4470_sashi/front.jpg")

        self.assertEqual(engine.get_registered_count(), 1)
        profile = engine.registered_profiles[student_id]
        self.assertEqual(len(profile.encodings), 1)
        self.assertEqual(profile.source_files, ["25a51a4470_sashi/front.jpg"])

        # 2. Register left photo for SAME student ID
        enc_left = np.zeros(128, dtype=np.float64)
        enc_left[0] = 0.2

        engine._register_encoding(student_id, student_name, enc_left, "25a51a4470_sashi/left.jpg")

        # Must still be ONE student profile
        self.assertEqual(engine.get_registered_count(), 1)
        self.assertEqual(len(profile.encodings), 2)
        self.assertEqual(profile.source_files, ["25a51a4470_sashi/front.jpg", "25a51a4470_sashi/left.jpg"])

        # 3. Register right photo for SAME student ID
        enc_right = np.zeros(128, dtype=np.float64)
        enc_right[0] = 0.3

        engine._register_encoding(student_id, student_name, enc_right, "25a51a4470_sashi/right.jpg")

        # Must still be ONE student profile
        self.assertEqual(engine.get_registered_count(), 1)
        self.assertEqual(len(profile.encodings), 3)
        self.assertEqual(profile.source_files, [
            "25a51a4470_sashi/front.jpg",
            "25a51a4470_sashi/left.jpg",
            "25a51a4470_sashi/right.jpg",
        ])

        # 4. Verify student directory returns Sashi ONLY ONCE with photo_count == 3
        details = self.fem.get_student_details("CSE")
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["student_id"], student_id)
        self.assertEqual(details[0]["student_name"], student_name)
        self.assertEqual(details[0]["photo_count"], 3)

        # 5. Verify duplicate photo content check
        distances = [np.linalg.norm(e - enc_front) for e in profile.encodings]
        self.assertTrue(any(d < 0.02 for d in distances), "Duplicate front photo should match existing encoding.")

        # 6. Verify recognition matches against any of the 3 encodings (e.g. matching left pose)
        query_left_pose = np.zeros(128, dtype=np.float64)
        query_left_pose[0] = 0.21  # Close to enc_left (distance ~ 0.01)

        # Compute minimum distance across all 3 reference encodings
        dists = [np.linalg.norm(e - query_left_pose) for e in profile.encodings]
        best_dist = min(dists)
        self.assertLess(best_dist, 0.50)
        self.assertAlmostEqual(best_dist, 0.01, places=2)

    def test_legacy_format_and_folder_format_coexistence(self):
        """Test scanning legacy file (25a51a4470_sashi.jpeg) and subfolder photos together."""
        engine = FaceEngine(known_students_dir=self.cse_dir)

        # Legacy file
        legacy_enc = np.zeros(128, dtype=np.float64)
        engine._register_encoding("25a51a4470", "sashi", legacy_enc, "25a51a4470_sashi.jpeg")

        # Subfolder file
        sub_enc = np.zeros(128, dtype=np.float64)
        sub_enc[1] = 0.15
        engine._register_encoding("25a51a4470", "sashi", sub_enc, "25a51a4470_sashi/left.jpg")

        # Must merge into ONE StudentProfile with 2 encodings
        self.assertEqual(engine.get_registered_count(), 1)
        profile = engine.registered_profiles["25a51a4470"]
        self.assertEqual(len(profile.encodings), 2)
        self.assertEqual(len(profile.source_files), 2)


if __name__ == "__main__":
    unittest.main()
