# Classroom Attendance System — Deployment Guide

A dynamic, cloud-configurable, AI-powered Classroom Attendance System featuring automated face recognition evidence, timetable period management, period-end absent finalization, and inline email notifications.

---

## 🛠️ GitHub & Railway Deployment

### 1. GitHub Preparation
1. Create a **PRIVATE** repository on GitHub.
2. Initialize Git and push the code:
   ```bash
   git init
   git add .
   git commit -m "Production cloud deployment preparation"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_PRIVATE_REPO.git
   git push -u origin main
   ```

### 2. Railway Cloud Deployment
1. Log in to [Railway.app](https://railway.app).
2. Click **New Project** > **Deploy from GitHub repo**.
3. Select your private `classroom_attendance` repository.
4. Add a **MySQL** database plugin to your project.
5. Railway will automatically detect the `Procfile` and `requirements.txt`.

### 3. Environment Variables (Railway)
Configure the following in the **Variables** tab of your Railway service:

| Variable Name | Description | Example |
|---------------|-------------|---------|
| `ENVIRONMENT` | Run mode | `production` |
| `SECRET_KEY` | Flask session secret | `random-string-here` |
| `DATABASE_URL` | MySQL connection | `${{MySQL.MYSQL_URL}}` (automatic) |
| `SMTP_HOST` | Email server | `smtp.gmail.com` |
| `SMTP_PORT` | Email port | `587` |
| `SMTP_USERNAME`| Email user | `user@example.com` |
| `SMTP_PASSWORD`| App password | `your-app-password` |
| `CAMERA_CSD` | CCTV Source | `rtsp://user:pass@ip:554/stream` |
| `STORAGE_BACKEND` | Storage type | `s3` (for cloud) or `local` |
| `S3_BUCKET` | Cloud bucket name | `my-attendance-evidence` |
| `S3_ENDPOINT` | Cloud endpoint | `https://<accountid>.r2.cloudflarestorage.com` |
| `S3_ACCESS_KEY` | Cloud access key | `...` |
| `S3_SECRET_KEY` | Cloud secret key | `...` |

---

## 📹 Architecture & CCTV Integration

- **Cloud Server**: Runs the Flask dashboard, handles API requests, and stores metadata in MySQL.
- **Biometric Evidence**: In production, `STORAGE_BACKEND=s3` is recommended to ensure student photos and recognition crops persist across container restarts.
- **Campus CCTV**: If cameras are on a local network, a **local agent** (on-premise) can be deployed to run recognition and forward events to this cloud server's API.

---

## 🗄️ Database & Attendance Logic

- **Automatic Migrations**: Database tables are created and migrated automatically on startup.
- **Timetable Enforcement**: Attendance is only recorded during active periods defined in the `timetable` table.
- **Auto-Absent**: Use the `/api/attendance/finalize` endpoint (via dashboard) to mark students absent who were not recognized during a period.
- **Data Safety**: `.env`, `students/`, and `data/evidence/` are excluded from Git to prevent leaking sensitive biometric data or credentials.

---

## 🚀 Local Development

1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill values.
3. Run: `python src/app.py`

---

## 📹 Security Notes

- **Credential Masking**: RTSP and DB passwords are never exposed in logs or API responses.
- **Protected Endpoints**: Evidence photos and student registry are protected by session authentication.
- **Department Isolation**: HOD users can only access data for their own department.
