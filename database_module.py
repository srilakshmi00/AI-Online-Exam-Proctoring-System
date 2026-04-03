"""
Module 6: Database Schema & Setup
====================================
Project : Online Exam Proctoring System Based on AI
Authors : Ramkumar R, Sri Lakshmi T, Subiksha V
College : Shree Venkateshwara Hi-Tech Engineering College

Run this file ONCE to create all required MySQL tables.
Also contains helper functions used by other modules.

USAGE:
    python database_module.py
"""

import mysql.connector
import pickle
from datetime import datetime


# ─────────────────────────────────────────────
#  CONFIG  — update password
# ─────────────────────────────────────────────

DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "password",   
    "database": "proctoring_db"
}


# ═══════════════════════════════════════════════
#  PART 1 — CREATE DATABASE & TABLES
# ═══════════════════════════════════════════════

CREATE_DB_SQL = "CREATE DATABASE IF NOT EXISTS proctoring_db;"

TABLE_SQLS = [

    # ── Candidates ──────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS candidates (
        candidate_id   VARCHAR(20)  PRIMARY KEY,
        name           VARCHAR(100) NOT NULL,
        face_embedding BLOB         NOT NULL,
        enrolled_at    DATETIME     DEFAULT CURRENT_TIMESTAMP
    );
    """,

    # ── Exam Sessions ────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS exam_sessions (
        session_id      VARCHAR(30)  PRIMARY KEY,
        candidate_id    VARCHAR(20)  NOT NULL,
        exam_name       VARCHAR(100) NOT NULL,
        started_at      DATETIME     DEFAULT CURRENT_TIMESTAMP,
        ended_at        DATETIME     DEFAULT NULL,
        violation_count INT          DEFAULT 0,
        status          ENUM('active','completed','flagged')
                        DEFAULT 'active',
        FOREIGN KEY (candidate_id)
            REFERENCES candidates(candidate_id)
            ON DELETE CASCADE
    );
    """,

    # ── Event Logs ───────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS event_logs (
        log_id          INT AUTO_INCREMENT PRIMARY KEY,
        candidate_id    VARCHAR(20)  NOT NULL,
        session_id      VARCHAR(30)  NOT NULL,
        event_type      VARCHAR(50)  NOT NULL,
        description     TEXT,
        severity        ENUM('LOW','MEDIUM','HIGH') DEFAULT 'MEDIUM',
        screenshot_path VARCHAR(255) DEFAULT '',
        logged_at       DATETIME     DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_session  (session_id),
        INDEX idx_candidate(candidate_id)
    );
    """,

    # ── Results ──────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS results (
        result_id       INT AUTO_INCREMENT PRIMARY KEY,
        candidate_id    VARCHAR(20)  NOT NULL,
        session_id      VARCHAR(30)  NOT NULL,
        total_violations INT         DEFAULT 0,
        high_severity    INT         DEFAULT 0,
        medium_severity  INT         DEFAULT 0,
        verdict          ENUM('PASS','FLAGGED','FAIL') DEFAULT 'PASS',
        generated_at     DATETIME    DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (candidate_id)
            REFERENCES candidates(candidate_id)
            ON DELETE CASCADE
    );
    """
]


def setup_database():
    """
    Creates the proctoring_db database and all required tables.
    Safe to run multiple times — uses IF NOT EXISTS.
    """
    # Connect without specifying a database first to create it
    try:
        conn = mysql.connector.connect(
            host     = DB_CONFIG["host"],
            user     = DB_CONFIG["user"],
            password = DB_CONFIG["password"]
        )
        cursor = conn.cursor()
        cursor.execute(CREATE_DB_SQL)
        print("[DB] Database 'proctoring_db' ready.")
        cursor.close()
        conn.close()
    except mysql.connector.Error as e:
        print(f"[DB ERROR] Cannot connect: {e}")
        print("[HINT] Check your password in DB_CONFIG.")
        return False

    # Now connect to the database and create tables
    try:
        conn   = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        for sql in TABLE_SQLS:
            cursor.execute(sql)
            conn.commit()

        print("[DB] All tables created successfully:")
        print("       candidates, exam_sessions, event_logs, results")
        return True

    except mysql.connector.Error as e:
        print(f"[DB ERROR] Table creation failed: {e}")
        return False

    finally:
        cursor.close()
        conn.close()


# ═══════════════════════════════════════════════
#  PART 2 — SESSION HELPERS
# ═══════════════════════════════════════════════

def _get_conn():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as e:
        print(f"[DB ERROR] {e}")
        return None


def create_session(candidate_id: str, exam_name: str) -> str:
    """
    Creates a new exam session row and returns the session_id.
    session_id format: <candidate_id>_<YYYYMMDD_HHMMSS>
    """
    session_id = f"{candidate_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    conn = _get_conn()
    if conn is None:
        return session_id   # return it anyway for offline use

    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO exam_sessions (session_id, candidate_id, exam_name)
            VALUES (%s, %s, %s)
        """, (session_id, candidate_id, exam_name))
        conn.commit()
        print(f"[DB] Session created: {session_id}")
        return session_id

    except mysql.connector.Error as e:
        print(f"[DB ERROR] create_session: {e}")
        return session_id

    finally:
        cursor.close()
        conn.close()


def end_session(session_id: str):
    """Marks a session as completed and records the end time."""
    conn = _get_conn()
    if conn is None:
        return

    try:
        cursor = conn.cursor()

        # Count violations to decide verdict
        cursor.execute("""
            SELECT violation_count FROM exam_sessions
            WHERE session_id = %s
        """, (session_id,))
        row   = cursor.fetchone()
        count = row[0] if row else 0
        status = "flagged" if count >= 5 else "completed"

        cursor.execute("""
            UPDATE exam_sessions
            SET ended_at = %s, status = %s
            WHERE session_id = %s
        """, (datetime.now(), status, session_id))
        conn.commit()
        print(f"[DB] Session {session_id} ended — status: {status}")

    except mysql.connector.Error as e:
        print(f"[DB ERROR] end_session: {e}")

    finally:
        cursor.close()
        conn.close()


def generate_result(candidate_id: str, session_id: str):
    """
    Tallies all violations for the session and writes
    a summary record to the results table.
    Verdict: PASS (<3 violations), FLAGGED (3–9), FAIL (10+)
    """
    conn = _get_conn()
    if conn is None:
        return

    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(severity = 'HIGH')   as high,
                SUM(severity = 'MEDIUM') as medium
            FROM event_logs
            WHERE session_id = %s
        """, (session_id,))
        row    = cursor.fetchone()
        total  = row[0] or 0
        high   = row[1] or 0
        medium = row[2] or 0

        if total < 3:
            verdict = "PASS"
        elif total < 10:
            verdict = "FLAGGED"
        else:
            verdict = "FAIL"

        cursor.execute("""
            INSERT INTO results
                (candidate_id, session_id, total_violations,
                 high_severity, medium_severity, verdict)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                total_violations = VALUES(total_violations),
                high_severity    = VALUES(high_severity),
                medium_severity  = VALUES(medium_severity),
                verdict          = VALUES(verdict),
                generated_at     = NOW()
        """, (candidate_id, session_id, total, high, medium, verdict))
        conn.commit()

        print(f"\n[RESULT] Candidate: {candidate_id}")
        print(f"         Total violations : {total}")
        print(f"         High severity    : {high}")
        print(f"         Verdict          : {verdict}")

    except mysql.connector.Error as e:
        print(f"[DB ERROR] generate_result: {e}")

    finally:
        cursor.close()
        conn.close()


def get_all_candidates() -> list:
    """Returns list of all enrolled candidates for the dashboard dropdown."""
    conn = _get_conn()
    if conn is None:
        return []

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT candidate_id, name, enrolled_at FROM candidates"
        )
        return cursor.fetchall()

    except mysql.connector.Error as e:
        print(f"[DB ERROR] get_all_candidates: {e}")
        return []

    finally:
        cursor.close()
        conn.close()


def get_session_result(session_id: str) -> dict:
    """Fetches the result record for a given session."""
    conn = _get_conn()
    if conn is None:
        return {}

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM results WHERE session_id = %s", (session_id,)
        )
        row = cursor.fetchone()
        return row if row else {}

    except mysql.connector.Error as e:
        print(f"[DB ERROR] get_session_result: {e}")
        return {}

    finally:
        cursor.close()
        conn.close()


# ═══════════════════════════════════════════════
#  RUN SETUP
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("  Database Setup — Proctoring System")
    print("=" * 50)
    ok = setup_database()
    if ok:
        print("\n[OK] Database is ready. You can now run other modules.")
    else:
        print("\n[FAILED] Fix the DB error above and try again.")
