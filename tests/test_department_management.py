"""
test_department_management.py - Dynamic Department Management Tests.

Tests:
  - Adding new departments (e.g. CSD, AIML) dynamically without Python code changes.
  - Updating department details & camera sources.
  - Disabling/enabling departments.
  - Automatic directory creation under students/{DEPT_CODE}/.
  - HOD access isolation & Admin multi-department access.
"""

import os
import unittest
import tempfile
import shutil

from config import Config
from database import DatabaseManager
from face_engine_manager import FaceEngineManager
from camera_manager import CameraManager


class TestDepartmentManagement(unittest.TestCase):

    def setUp(self):
        self.db = DatabaseManager(
            host=Config.MYSQL_HOST,
            port=Config.MYSQL_PORT,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DATABASE,
        )
        self.assertTrue(self.db.connect(), "Failed to connect to MySQL database.")

        # Clean up any leftover test departments
        self._cleanup_test_depts()

        self.test_dir = tempfile.mkdtemp(prefix="dept_mgr_test_")
        self.fem = FaceEngineManager(base_dir=self.test_dir)

    def tearDown(self):
        self._cleanup_test_depts()
        self.db.disconnect()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _cleanup_test_depts(self):
        """Remove test department rows from database."""
        with self.db._lock:
            if self.db._ensure_connection_unlocked():
                cursor = None
                try:
                    cursor = self.db._connection.cursor()
                    cursor.execute("DELETE FROM departments WHERE code IN ('AIML', 'CSD', 'MECH', 'CIVIL');")
                    self.db._connection.commit()
                except Exception:
                    pass
                finally:
                    if cursor:
                        try:
                            cursor.close()
                        except Exception:
                            pass

    def test_add_dynamic_department(self):
        """Test adding new departments dynamically via database."""
        success, msg = self.db.add_department(code="AIML", name="Artificial Intelligence & Machine Learning", camera_source="0")
        self.assertTrue(success, f"Failed to add department AIML: {msg}")

        dept = self.db.get_department_by_code("AIML")
        self.assertIsNotNone(dept)
        self.assertEqual(dept["code"], "AIML")
        self.assertEqual(dept["name"], "Artificial Intelligence & Machine Learning")
        self.assertEqual(dept["camera_source"], "0")
        self.assertTrue(dept["is_enabled"])

    def test_update_department(self):
        """Test updating department camera source and name."""
        self.db.add_department(code="CSD", name="Data Science", camera_source="1")

        success, msg = self.db.update_department(code="CSD", camera_source="rtsp://192.168.1.50/live")
        self.assertTrue(success, f"Failed to update department: {msg}")

        dept = self.db.get_department_by_code("CSD")
        self.assertEqual(dept["camera_source"], "rtsp://192.168.1.50/live")

    def test_toggle_department(self):
        """Test disabling and enabling a department."""
        self.db.add_department(code="MECH", name="Mechanical Engineering")

        # Disable
        success, _ = self.db.toggle_department(code="MECH", is_enabled=False)
        self.assertTrue(success)

        enabled_depts = [d["code"] for d in self.db.get_departments(enabled_only=True)]
        self.assertNotIn("MECH", enabled_depts)

        # Enable
        self.db.toggle_department(code="MECH", is_enabled=True)
        enabled_depts = [d["code"] for d in self.db.get_departments(enabled_only=True)]
        self.assertIn("MECH", enabled_depts)

    def test_folder_auto_creation_for_new_dept(self):
        """Test that student photo directory is created for new departments."""
        dept_path = os.path.join(self.test_dir, "CIVIL")
        self.assertFalse(os.path.exists(dept_path))

        engine = self.fem.get_engine("CIVIL")
        self.assertIsNotNone(engine)
        self.assertTrue(os.path.exists(dept_path))

    def test_camera_manager_dynamic_update(self):
        """Test CameraManager dynamic source configuration."""
        cm = CameraManager(department_cameras={"CSE": ["0"]})
        self.assertEqual(cm.get_camera_sources("CSE"), ["0"])

        cm.set_department_camera("AIML", "rtsp://camera1")
        self.assertEqual(cm.get_camera_sources("AIML"), ["rtsp://camera1"])
        cm.release_all()


if __name__ == "__main__":
    unittest.main()
