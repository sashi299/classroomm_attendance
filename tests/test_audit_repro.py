import io
import os
import sys
import unittest
import numpy as np
import cv2
import logging

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import app

# Set up logging to capture the audit logs
log_capture = io.StringIO()
ch = logging.StreamHandler(log_capture)
ch.setLevel(logging.INFO)
logging.getLogger("dashboard_app").addHandler(ch)

class TestAuditRepro(unittest.TestCase):
    def setUp(self):
        app.app.testing = True
        app.initialize_components()
        self.client = app.app.test_client()
        self.client.post("/login", data={"username": "admin", "password": "admin@2026"})

    def test_repro_with_existing_images(self):
        base = os.path.join(os.path.dirname(__file__), "..", "students")
        images = [
            os.path.join(base, "EEE", "99A01_Test EEE Student", "valid_student.jpg"),
            os.path.join(base, "CSD", "99A03_Double Face Test", "double_face.jpg")
        ]

        for img_path in images:
            if not os.path.exists(img_path): continue

            print(f"\n--- TESTING IMAGE: {os.path.basename(img_path)} ---")
            with open(img_path, "rb") as f:
                data = {"photo": (io.BytesIO(f.read()), "enroll_frame.jpg")}
                res = self.client.post("/api/students/enroll/validate-frame",
                                       data=data, content_type='multipart/form-data')

                print(f"Status: {res.status_code}")
                print(f"Response: {res.get_json()}")
                print("Logs:")
                print(log_capture.getvalue())
                log_capture.truncate(0)
                log_capture.seek(0)

if __name__ == "__main__":
    unittest.main()
