"""
test_storage.py - Unit tests for StorageManager (local and cloud storage abstraction).
"""

import os
import sys
import unittest
import tempfile
import shutil

# Add src/ to path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from storage import StorageManager


class TestStorageManager(unittest.TestCase):
    """Test Local and S3 storage abstraction behavior."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.local_storage = StorageManager(backend="local", base_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_local_storage_save_and_get(self):
        """Verify local storage saves bytes and retrieves them cleanly."""
        data = b"RECOGNITION_FACE_CROP_BYTES"
        key = "test_dept/2026-08-13/P1/ACC_01.jpg"

        saved = self.local_storage.save_bytes(key, data)
        self.assertTrue(saved)

        retrieved = self.local_storage.get_bytes(key)
        self.assertEqual(retrieved, data)

    def test_local_storage_missing_key_returns_none(self):
        """Verify missing key returns None without error."""
        retrieved = self.local_storage.get_bytes("non_existent_key.jpg")
        self.assertIsNone(retrieved)

    def test_s3_backend_fallback_without_boto3(self):
        """Verify S3 backend falls back safely without error."""
        s3_storage = StorageManager(
            backend="s3",
            base_dir=self.temp_dir,
            s3_bucket="test-bucket",
            s3_endpoint="https://s3.example.com",
        )
        self.assertEqual(s3_storage.backend, "s3")

        data = b"S3_TEST_DATA"
        key = "test_key.jpg"
        # Saves safely via fallback if boto3 credentials unavailable
        result = s3_storage.save_bytes(key, data)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
