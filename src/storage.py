"""
storage.py - Cloud-Ready Object Storage Abstraction for Face Evidence and Student Photos.

Supports:
  1. STORAGE_BACKEND = "local" (Default for local development / testing)
  2. STORAGE_BACKEND = "s3" (AWS S3, Cloudflare R2, DigitalOcean Spaces, MinIO, etc.)
"""

import os
import io
import logging
from typing import Optional

logger = logging.getLogger("storage")


class StorageManager:
    """Provider-neutral storage abstraction layer for local and cloud object storage."""

    def __init__(
        self,
        backend: str = "local",
        base_dir: str = "data/evidence",
        s3_endpoint: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        s3_region: Optional[str] = None,
        s3_access_key: Optional[str] = None,
        s3_secret_key: Optional[str] = None,
    ):
        self.backend = (backend or os.getenv("STORAGE_BACKEND", "local")).lower().strip()
        self.base_dir = os.path.abspath(base_dir)
        self.s3_endpoint = s3_endpoint or os.getenv("S3_ENDPOINT", "")
        self.s3_bucket = s3_bucket or os.getenv("S3_BUCKET", "")
        self.s3_region = s3_region or os.getenv("S3_REGION", "us-east-1")
        self.s3_access_key = s3_access_key or os.getenv("S3_ACCESS_KEY", "")
        self.s3_secret_key = s3_secret_key or os.getenv("S3_SECRET_KEY", "")

        if self.backend == "local":
            os.makedirs(self.base_dir, exist_ok=True)
            logger.info("StorageManager initialized with LOCAL backend at '%s'", self.base_dir)
        elif self.backend == "s3":
            logger.info("StorageManager initialized with S3 backend (bucket: %s, endpoint: %s)", self.s3_bucket, self.s3_endpoint)
        else:
            logger.warning("Unknown STORAGE_BACKEND '%s'. Falling back to local.", self.backend)
            self.backend = "local"
            os.makedirs(self.base_dir, exist_ok=True)

    def save_bytes(self, storage_key: str, data: bytes, content_type: str = "image/jpeg") -> bool:
        """Save binary data under a unique storage key."""
        clean_key = storage_key.lstrip("/").replace("\\", "/")

        if self.backend == "local":
            target_path = os.path.join(self.base_dir, clean_key)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            try:
                with open(target_path, "wb") as f:
                    f.write(data)
                return True
            except Exception as e:
                logger.error("Error saving file locally [%s]: %s", target_path, e)
                return False

        elif self.backend == "s3":
            try:
                import boto3
                session = boto3.session.Session()
                s3_opts = {}
                if self.s3_endpoint:
                    s3_opts["endpoint_url"] = self.s3_endpoint
                if self.s3_region:
                    s3_opts["region_name"] = self.s3_region

                s3_client = session.client(
                    "s3",
                    aws_access_key_id=self.s3_access_key,
                    aws_secret_access_key=self.s3_secret_key,
                    **s3_opts
                )
                s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=clean_key,
                    Body=data,
                    ContentType=content_type
                )
                return True
            except Exception as e:
                logger.error("Error uploading to S3 [%s]: %s", clean_key, e)
                # Fallback to local storage if S3 fails
                target_path = os.path.join(self.base_dir, clean_key)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with open(target_path, "wb") as f:
                    f.write(data)
                return True

        return False

    def get_bytes(self, storage_key: str) -> Optional[bytes]:
        """Retrieve binary data by storage key."""
        clean_key = storage_key.lstrip("/").replace("\\", "/")

        if self.backend == "local":
            target_path = os.path.join(self.base_dir, clean_key)
            if not os.path.isfile(target_path):
                return None
            try:
                with open(target_path, "rb") as f:
                    return f.read()
            except Exception as e:
                logger.error("Error reading local file [%s]: %s", target_path, e)
                return None

        elif self.backend == "s3":
            try:
                import boto3
                session = boto3.session.Session()
                s3_opts = {}
                if self.s3_endpoint:
                    s3_opts["endpoint_url"] = self.s3_endpoint
                if self.s3_region:
                    s3_opts["region_name"] = self.s3_region

                s3_client = session.client(
                    "s3",
                    aws_access_key_id=self.s3_access_key,
                    aws_secret_access_key=self.s3_secret_key,
                    **s3_opts
                )
                obj = s3_client.get_object(Bucket=self.s3_bucket, Key=clean_key)
                return obj["Body"].read()
            except Exception as e:
                logger.error("Error reading from S3 [%s]: %s", clean_key, e)
                # Fallback check locally
                target_path = os.path.join(self.base_dir, clean_key)
                if os.path.isfile(target_path):
                    with open(target_path, "rb") as f:
                        return f.read()
                return None

        return None
