import io
import os
import sys
import unittest
import numpy as np
import cv2
from unittest.mock import patch

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import app

class TestMultifaceFinal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.app.testing = True
        app.initialize_components()
        cls.client = app.app.test_client()

        base = os.path.join(os.path.dirname(__file__), "..", "students")
        cls.real_img_path = os.path.join(base, "EEE", "99A01_Test EEE Student", "valid_student.jpg")

        with open(cls.real_img_path, "rb") as f:
            cls.real_img_bytes = f.read()

    def setUp(self):
        self.client.post("/login", data={"username": "admin", "password": "admin@2026"})

    def test_overlap_merging(self):
        """Test that two highly overlapping boxes on 1 face are merged."""
        # Note: valid_student.jpg is 1600h x 1200w
        box1 = (400, 800, 900, 300)
        box2 = (410, 810, 910, 310)

        with patch('face_recognition.face_locations') as mock_locs, \
             patch('face_recognition.face_encodings') as mock_encs:

            mock_locs.return_value = [box1, box2]
            dummy_enc = np.random.rand(128)
            mock_encs.side_effect = [[dummy_enc], [dummy_enc]]

            data = {"photo": (io.BytesIO(self.real_img_bytes), "test.jpg")}
            res = self.client.post("/api/students/enroll/validate-frame", data=data, content_type='multipart/form-data')

            json_res = res.get_json()
            self.assertEqual(res.status_code, 200, f"Expected merged success, got {json_res}")

    def test_noise_filtering(self):
        """Test that a box with no recognizable landmarks is filtered out."""
        real_face = (400, 800, 900, 300)
        noise_box = (100, 200, 200, 100) # passes size check (100px)

        with patch('face_recognition.face_locations') as mock_locs, \
             patch('face_recognition.face_encodings') as mock_encs:

            mock_locs.return_value = [real_face, noise_box]
            mock_encs.side_effect = [[np.random.rand(128)], []] # Noise fails encoding

            data = {"photo": (io.BytesIO(self.real_img_bytes), "test.jpg")}
            res = self.client.post("/api/students/enroll/validate-frame", data=data, content_type='multipart/form-data')

            self.assertEqual(res.status_code, 200, f"Expected noise filtered success, got {res.get_json()}")

    def test_genuine_multiple_faces(self):
        """Test that two separate, recognizable people are rejected."""
        face1 = (400, 400, 900, 100)
        face2 = (400, 1100, 900, 800)

        with patch('face_recognition.face_locations') as mock_locs, \
             patch('face_recognition.face_encodings') as mock_encs:

            mock_locs.return_value = [face1, face2]
            mock_encs.side_effect = [[np.random.rand(128)], [np.random.rand(128)]]

            data = {"photo": (io.BytesIO(self.real_img_bytes), "test.jpg")}
            res = self.client.post("/api/students/enroll/validate-frame", data=data, content_type='multipart/form-data')

            self.assertEqual(res.status_code, 400)
            self.assertIn("Multiple faces", res.get_json()["error"])

if __name__ == "__main__":
    unittest.main()
