import io
import os
import sys
import requests

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

def test_api_with_image(image_path):
    url = "http://localhost:5000/api/students/enroll/validate-frame"

    # Need a session/login because of @login_required
    session = requests.Session()
    # Assuming the app is running
    try:
        # Try to login first
        login_res = session.post("http://localhost:5000/login", data={"username": "admin", "password": "admin@2026"})

        with open(image_path, "rb") as f:
            files = {"photo": ("enroll_frame.jpg", f, "image/jpeg")}
            res = session.post(url, files=files)
            print(f"Status: {res.status_code}")
            print(f"Response: {res.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    img = "students/EEE/99A01_Test EEE Student/valid_student.jpg"
    if os.path.exists(img):
        test_api_with_image(img)
    else:
        print(f"File not found: {img}")
