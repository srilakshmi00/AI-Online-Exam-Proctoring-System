"""
Module 5: Alert Engine & Event Logger
========================================
Project : Online Exam Proctoring System Based on AI
Authors : Ramkumar R, Sri Lakshmi T, Subiksha V
College : Shree Venkateshwara Hi-Tech Engineering College

Receives violation events from all other modules,
logs them to MySQL, saves screenshot evidence,
and displays real-time alerts to the instructor.
"""

import cv2
import os
import pickle
import mysql.connector
from datetime import datetime


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "password",    
    "database": "proctoring_db"
}

# Folder where violation screenshots are saved
SCREENSHOT_DIR = "violation_screenshots"


# ─────────────────────────────────────────────
#  VIOLATION TYPE → SEVERITY MAPPING
# ─────────────────────────────────────────────

SEVERITY = {
    "no_face":          "HIGH",
    "multiple_faces":   "HIGH",
    "mismatch":         "HIGH",
    "phone_detected":   "HIGH",
    "extra_person":     "HIGH",
    "book_detected":    "MEDIUM",
    "eyes_closed":      "MEDIUM",
    "looking_away":     "MEDIUM",
    "speech_detected":  "MEDIUM",
    "suspicious_speech":"HIGH",
}


# ═══════════════════════════════════════════════
#  HELPER — DB CONNECTION
# ═══════════════════════════════════════════════

def _get_conn():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as e:
        print(f"[DB ERROR] {e}")
        return None


# ═══════════════════════════════════════════════
#  PART 1 — SAVE SCREENSHOT
# ═══════════════════════════════════════════════

def save_screenshot(frame, candidate_id: str, event_type: str) -> str:
    """
    Saves a violation screenshot to disk.
    Returns the file path, or empty string on failure.
    """
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{candidate_id}_{event_type}_{ts}.jpg"
    path     = os.path.join(SCREENSHOT_DIR, filename)
    try:
        cv2.imwrite(path, frame)
        return path
    except Exception as e:
        print(f"[SCREENSHOT] Failed to save: {e}")
        return ""


# ═══════════════════════════════════════════════
#  PART 2 — LOG EVENT TO DATABASE
# ═══════════════════════════════════════════════

def log_event(candidate_id: str, session_id: str,
              event_type: str, description: str,
              screenshot_path: str = "") -> bool:
    """
    Inserts a violation event into the event_logs table.

    Args:
        candidate_id    : Student ID
        session_id      : Current exam session ID
        event_type      : e.g. "phone_detected", "no_face"
        description     : Human-readable detail string
        screenshot_path : Path to saved screenshot (optional)
    """
    conn = _get_conn()
    if conn is None:
        return False

    severity = SEVERITY.get(event_type, "LOW")

    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO event_logs
                (candidate_id, session_id, event_type, description,
                 severity, screenshot_path, logged_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (candidate_id, session_id, event_type, description,
              severity, screenshot_path, datetime.now()))
        conn.commit()
        return True

    except mysql.connector.Error as e:
        print(f"[DB ERROR] log_event: {e}")
        return False

    finally:
        cursor.close()
        conn.close()


# ═══════════════════════════════════════════════
#  PART 3 — UPDATE SESSION VIOLATION COUNT
# ═══════════════════════════════════════════════

def increment_violation_count(session_id: str) -> int:
    """
    Increments the violation counter for a session.
    Returns the new total count.
    """
    conn = _get_conn()
    if conn is None:
        return 0

    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE exam_sessions
            SET violation_count = violation_count + 1
            WHERE session_id = %s
        """, (session_id,))
        conn.commit()

        cursor.execute("""
            SELECT violation_count FROM exam_sessions
            WHERE session_id = %s
        """, (session_id,))
        row = cursor.fetchone()
        return row[0] if row else 0

    except mysql.connector.Error as e:
        print(f"[DB ERROR] increment_violation: {e}")
        return 0

    finally:
        cursor.close()
        conn.close()


# ═══════════════════════════════════════════════
#  PART 4 — MAIN ALERT HANDLER
#  This is the single function all modules call
# ═══════════════════════════════════════════════

class AlertEngine:
    """
    Central alert handler.
    All 4 detection modules call handle_violation() when they detect something.

    Usage:
        engine = AlertEngine(candidate_id="1234", session_id="S001")
        engine.handle_violation(event_type="phone_detected",
                                description="Phone seen at 94% confidence",
                                frame=frame)
    """

    def __init__(self, candidate_id: str, session_id: str,
                 on_alert=None):
        """
        candidate_id : student being monitored
        session_id   : current exam session
        on_alert     : optional callback(event_type, description, count)
                       used by the dashboard to update its UI
        """
        self.candidate_id    = candidate_id
        self.session_id      = session_id
        self.on_alert        = on_alert
        self.violation_count = 0
        self._alert_log      = []   # in-memory log for dashboard

    def handle_violation(self, event_type: str, description: str,
                         frame=None):
        """
        Called by any detection module when a violation is found.
        - Saves a screenshot
        - Logs to DB
        - Increments violation count
        - Fires on_alert callback for the dashboard
        """
        ts = datetime.now().strftime("%H:%M:%S")
        severity = SEVERITY.get(event_type, "LOW")

        # Print to terminal
        print(f"[{ts}] [{severity}] {event_type}: {description}")

        # Save screenshot if frame provided
        screenshot_path = ""
        if frame is not None:
            screenshot_path = save_screenshot(
                frame, self.candidate_id, event_type
            )

        # Log to database
        log_event(
            candidate_id    = self.candidate_id,
            session_id      = self.session_id,
            event_type      = event_type,
            description     = description,
            screenshot_path = screenshot_path
        )

        # Update violation count
        self.violation_count += 1
        increment_violation_count(self.session_id)

        # Store in memory for dashboard
        entry = {
            "time":       ts,
            "event_type": event_type,
            "description": description,
            "severity":   severity,
            "screenshot": screenshot_path
        }
        self._alert_log.append(entry)

        # Fire dashboard callback
        if self.on_alert:
            self.on_alert(event_type, description, self.violation_count)

    def get_log(self) -> list:
        """Returns the in-memory alert log for the dashboard."""
        return self._alert_log.copy()

    def get_violation_count(self) -> int:
        return self.violation_count


# ═══════════════════════════════════════════════
#  PART 5 — FETCH LOGS (for dashboard / report)
# ═══════════════════════════════════════════════

def fetch_session_logs(session_id: str) -> list:
    """
    Retrieves all event logs for a given session from MySQL.
    Used by the instructor dashboard to display history.
    """
    conn = _get_conn()
    if conn is None:
        return []

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT event_type, description, severity,
                   screenshot_path, logged_at
            FROM event_logs
            WHERE session_id = %s
            ORDER BY logged_at ASC
        """, (session_id,))
        return cursor.fetchall()

    except mysql.connector.Error as e:
        print(f"[DB ERROR] fetch_logs: {e}")
        return []

    finally:
        cursor.close()
        conn.close()


def fetch_candidate_summary(candidate_id: str) -> dict:
    """
    Returns a summary of all violations for a candidate
    across all sessions — used in the results report.
    """
    conn = _get_conn()
    if conn is None:
        return {}

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT event_type, COUNT(*) as count
            FROM event_logs
            WHERE candidate_id = %s
            GROUP BY event_type
            ORDER BY count DESC
        """, (candidate_id,))
        rows = cursor.fetchall()
        return {r["event_type"]: r["count"] for r in rows}

    except mysql.connector.Error as e:
        print(f"[DB ERROR] fetch_summary: {e}")
        return {}

    finally:
        cursor.close()
        conn.close()
