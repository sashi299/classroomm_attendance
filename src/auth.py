"""
auth.py - Authentication and department-based access control.

Provides:
  - Hardcoded user store for HOD login (CSE, EEE, ECE, Admin).
  - Password hashing with werkzeug security utilities.
  - Login/logout session helpers.
  - Department isolation decorators for Flask routes.

In production, replace the hardcoded user store with a database-backed
user table and proper credential management.
"""

import os
import functools
import logging
from typing import Optional, Dict

from werkzeug.security import generate_password_hash, check_password_hash
from flask import session, redirect, url_for, request

logger = logging.getLogger(__name__)

# ── Department User Store ──────────────────────────────────────
# Each user has: username, password_hash, role, department_code, department_name
# Passwords should be set via environment variables in production.
USERS: Dict[str, dict] = {
    "csd_hod": {
        "password_hash": generate_password_hash(os.getenv("PASS_CSD_HOD", "csd@hod2026")),
        "role": "hod",
        "department_code": "CSD",
        "department_name": "CSD - Computer Science & Design",
    },
    "csm_hod": {
        "password_hash": generate_password_hash(os.getenv("PASS_CSM_HOD", "csm@hod2026")),
        "role": "hod",
        "department_code": "CSM",
        "department_name": "CSM - CSE (AI & ML)",
    },
    "cse_hod": {
        "password_hash": generate_password_hash(os.getenv("PASS_CSE_HOD", "cse@hod2026")),
        "role": "hod",
        "department_code": "CSE",
        "department_name": "CSE - Computer Science & Engineering",
    },
    "csc_hod": {
        "password_hash": generate_password_hash(os.getenv("PASS_CSC_HOD", "csc@hod2026")),
        "role": "hod",
        "department_code": "CSC",
        "department_name": "CSC - CSE (Cyber Security)",
    },
    "mech_hod": {
        "password_hash": generate_password_hash(os.getenv("PASS_MECH_HOD", "mech@hod2026")),
        "role": "hod",
        "department_code": "MECH",
        "department_name": "MECH - Mechanical Engineering",
    },
    "civil_hod": {
        "password_hash": generate_password_hash(os.getenv("PASS_CIVIL_HOD", "civil@hod2026")),
        "role": "hod",
        "department_code": "CIVIL",
        "department_name": "CIVIL - Civil Engineering",
    },
    "eee_hod": {
        "password_hash": generate_password_hash(os.getenv("PASS_EEE_HOD", "eee@hod2026")),
        "role": "hod",
        "department_code": "EEE",
        "department_name": "EEE - Electrical & Electronics Engineering",
    },
    "ece_hod": {
        "password_hash": generate_password_hash(os.getenv("PASS_ECE_HOD", "ece@hod2026")),
        "role": "hod",
        "department_code": "ECE",
        "department_name": "ECE - Electronics & Communication Engineering",
    },
    "admin": {
        "password_hash": generate_password_hash(os.getenv("PASS_ADMIN", "admin@2026")),
        "role": "admin",
        "department_code": "ALL",
        "department_name": "All Departments",
    },
}


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """
    Validate username and password against the user store.

    Args:
        username: The login username.
        password: The plaintext password to verify.

    Returns:
        User info dict (without password_hash) if valid, None otherwise.
    """
    user = USERS.get(username)
    if user is None:
        logger.warning("Login attempt with unknown username: %s", username)
        return None

    if not check_password_hash(user["password_hash"], password):
        logger.warning("Failed login for user: %s", username)
        return None

    logger.info("Successful login: %s (%s)", username, user["department_code"])
    return {
        "username": username,
        "role": user["role"],
        "department_code": user["department_code"],
        "department_name": user["department_name"],
    }


def login_user(user_info: dict) -> None:
    """Store authenticated user info in Flask session."""
    session["logged_in"] = True
    session["username"] = user_info["username"]
    session["role"] = user_info["role"]
    session["department_code"] = user_info["department_code"]
    session["department_name"] = user_info["department_name"]


def logout_user() -> None:
    """Clear the Flask session."""
    session.clear()


def get_current_user() -> Optional[dict]:
    """
    Get the currently logged-in user from the session.

    Returns:
        Dict with username, role, department_code, department_name
        or None if not logged in.
    """
    if not session.get("logged_in"):
        return None
    return {
        "username": session.get("username"),
        "role": session.get("role"),
        "department_code": session.get("department_code"),
        "department_name": session.get("department_name"),
    }


def is_admin() -> bool:
    """Check if the current user has admin role."""
    return session.get("role") == "admin"


def login_required(f):
    """
    Flask route decorator that requires authentication.
    Redirects to /login if not authenticated.
    """
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated_function
