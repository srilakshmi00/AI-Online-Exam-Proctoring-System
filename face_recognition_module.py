"""
Module 1: Face Detection & Recognition
=======================================
Project : Online Exam Proctoring System Based on AI
Authors : Ramkumar R, Sri Lakshmi T, Subiksha V
College : Shree Venkateshwara Hi-Tech Engineering College

Libraries used:
    pip install face-recognition dlib opencv-python numpy mysql-connector-python
"""

import cv2
import face_recognition
import numpy as np
import pickle
import os
import time
import mysql.connector
from datetime import datetime


# ─────────────────────────────────────────────
#  DATABASE CONFIG  (update with your details)
# ─────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "password",
    "database": "proctoring_db"
}

# Distance threshold: below = same person, above = alert
FACE_MATCH_THRESHOLD = 0.6

# How many seconds between each verification check during exam
VERIFICATION_INTERVAL = 3


# ═══════════════════════════════════════════════
#  PART 1 — ENROLLMENT
#  Run this ONCE before the exam to register
#  the student's face and save it to the DB.
# ═══════════════════════════════════════════════

def enroll_candidate(candidate_id: str, candidate_name: str):
    """
    Opens the webcam, captures the student's face,
    generates a 128-D embedding, and saves it to MySQL.

    Args:
        candidate_id   : Unique student ID (e.g. "732521104040")
        candidate_name : Student's full name
    """
    print(f"\n[ENROLLMENT] Starting for: {candidate_name} ({candidate_id})")
    print("  Look directly at the camera. Press SPACE to capture, Q to quit.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return False

    embedding = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        # Show live feed with instructions
        display = frame.copy()
        cv2.putText(display, "ENROLLMENT MODE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        cv2.putText(display, "Press SPACE to capture | Q to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Detect face in real time and draw box
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow("Enrollment - Proctoring System", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[INFO] Enrollment cancelled.")
            break

        if key == ord(' '):
            # Capture embedding on SPACE press
            if len(face_locations) == 0:
                print("[WARNING] No face detected. Please adjust your position.")
                continue
            if len(face_locations) > 1:
                print("[WARNING] Multiple faces detected. Please ensure only you are in frame.")
                continue

            encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if encodings:
                embedding = encodings[0]
                print(f"[INFO] Face captured successfully! Embedding shape: {embedding.shape}")
                cv2.putText(display, "CAPTURED!", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.imshow("Enrollment - Proctoring System", display)
                cv2.waitKey(800)
                break

    cap.release()
    cv2.destroyAllWindows()

    if embedding is None:
        print("[ERROR] Enrollment failed — no embedding generated.")
        return False

    # Save embedding to MySQL as binary (pickle)
    return _save_embedding_to_db(candidate_id, candidate_name, embedding)


def _save_embedding_to_db(candidate_id: str, candidate_name: str, embedding: np.ndarray) -> bool:
    """Serialises the 128-D numpy array and stores it in the candidates table."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        embedding_blob = pickle.dumps(embedding)

        cursor.execute("""
            INSERT INTO candidates (candidate_id, name, face_embedding, enrolled_at)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                face_embedding = VALUES(face_embedding),
                enrolled_at    = VALUES(enrolled_at)
        """, (candidate_id, candidate_name, embedding_blob, datetime.now()))

        conn.commit()
        print(f"[DB] Embedding saved for {candidate_name} ({candidate_id})")
        return True

    except mysql.connector.Error as e:
        print(f"[DB ERROR] {e}")
        return False

    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


# ═══════════════════════════════════════════════
#  PART 2 — VERIFICATION
#  Runs continuously during the exam in a
#  background thread (called by main_monitor.py)
# ═══════════════════════════════════════════════

def load_candidate_embedding(candidate_id: str):
    """
    Fetches the stored 128-D face embedding from MySQL
    for the given candidate. Returns a numpy array or None.
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT face_embedding FROM candidates WHERE candidate_id = %s",
            (candidate_id,)
        )
        row = cursor.fetchone()
        if row:
            return pickle.loads(row[0])
        else:
            print(f"[DB] No enrollment found for candidate: {candidate_id}")
            return None

    except mysql.connector.Error as e:
        print(f"[DB ERROR] {e}")
        return None

    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


def verify_face(frame: np.ndarray, stored_embedding: np.ndarray) -> dict:
    """
    Detects and verifies the face in the given webcam frame
    against the stored enrollment embedding.

    Args:
        frame            : BGR frame from OpenCV
        stored_embedding : 128-D numpy array from DB

    Returns a dict with keys:
        status    : "match" | "no_face" | "multiple_faces" | "mismatch"
        distance  : float (Euclidean distance, lower = more similar)
        message   : Human-readable result string
        face_box  : (top, right, bottom, left) or None
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    # ── No face in frame ──
    if len(face_locations) == 0:
        return {
            "status": "no_face",
            "distance": None,
            "message": "No face detected — candidate may be absent",
            "face_box": None
        }

    # ── More than one face ──
    if len(face_locations) > 1:
        return {
            "status": "multiple_faces",
            "distance": None,
            "message": f"{len(face_locations)} faces detected — possible collusion",
            "face_box": face_locations[0]
        }

    # ── Exactly one face — compute distance ──
    live_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    if not live_encodings:
        return {
            "status": "no_face",
            "distance": None,
            "message": "Face detected but encoding failed",
            "face_box": None
        }

    live_embedding = live_encodings[0]
    distance = float(np.linalg.norm(stored_embedding - live_embedding))

    if distance < FACE_MATCH_THRESHOLD:
        return {
            "status": "match",
            "distance": round(distance, 4),
            "message": f"Identity verified (dist={distance:.4f})",
            "face_box": face_locations[0]
        }
    else:
        return {
            "status": "mismatch",
            "distance": round(distance, 4),
            "message": f"Identity mismatch! (dist={distance:.4f}) — possible impersonation",
            "face_box": face_locations[0]
        }


# ═══════════════════════════════════════════════
#  PART 3 — REAL-TIME MONITORING LOOP
#  Draws results on screen and logs violations.
# ═══════════════════════════════════════════════

def draw_result_on_frame(frame: np.ndarray, result: dict) -> np.ndarray:
    """
    Overlays verification result (bounding box + status text)
    onto the webcam frame for the instructor's live view.
    """
    display = frame.copy()
    status = result["status"]
    message = result["message"]

    # Color coding
    color_map = {
        "match":          (0, 200, 0),    # Green
        "mismatch":       (0, 0, 255),    # Red
        "no_face":        (0, 140, 255),  # Orange
        "multiple_faces": (0, 0, 200),    # Dark red
    }
    color = color_map.get(status, (200, 200, 200))

    # Draw face bounding box
    if result["face_box"]:
        top, right, bottom, left = result["face_box"]
        cv2.rectangle(display, (left, top), (right, bottom), color, 2)

    # Status banner at top
    cv2.rectangle(display, (0, 0), (display.shape[1], 40), (30, 30, 30), -1)
    cv2.putText(display, message, (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Distance badge
    if result["distance"] is not None:
        badge = f"dist: {result['distance']}"
        cv2.putText(display, badge, (display.shape[1] - 160, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return display


def run_face_verification_loop(candidate_id: str, alert_callback=None):
    """
    Main real-time verification loop.
    Runs continuously and:
      - Verifies identity every VERIFICATION_INTERVAL seconds
      - Draws status on the live frame
      - Calls alert_callback(result) on any violation

    Args:
        candidate_id   : The student being monitored
        alert_callback : Optional function to call with violation results
                         (used by the alert engine in Module 5)

    Press Q to stop the loop.
    """
    print(f"\n[MONITOR] Loading embedding for candidate: {candidate_id}")
    stored_embedding = load_candidate_embedding(candidate_id)
    if stored_embedding is None:
        print("[ERROR] Cannot start monitoring — candidate not enrolled.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    print("[MONITOR] Face verification started. Press Q to stop.\n")
    last_check_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # Run verification only every VERIFICATION_INTERVAL seconds
        if current_time - last_check_time >= VERIFICATION_INTERVAL:
            result = verify_face(frame, stored_embedding)
            last_check_time = current_time

            # Log the result
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {result['message']}")

            # Fire alert callback for violations
            if result["status"] != "match" and alert_callback:
                alert_callback(result, frame)

        # Always show the annotated frame
        display_frame = draw_result_on_frame(frame, result if 'result' in dir() else
                                             {"status": "match", "distance": None,
                                              "message": "Initialising...", "face_box": None})
        cv2.imshow("Face Verification — Proctoring System", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[MONITOR] Stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════
#  QUICK TEST — run this file directly to test
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("  Face Recognition Module — Test Mode")
    print("=" * 50)

    mode = input("\nChoose mode:\n  1. Enroll a new candidate\n  2. Run verification loop\nEnter 1 or 2: ").strip()

    if mode == "1":
        cid  = input("Enter candidate ID   : ").strip()
        name = input("Enter candidate name : ").strip()
        success = enroll_candidate(cid, name)
        if success:
            print(f"\n[OK] {name} enrolled successfully!")
        else:
            print("\n[FAILED] Enrollment unsuccessful.")

    elif mode == "2":
        cid = input("Enter candidate ID to monitor: ").strip()

        def sample_alert(result, frame):
            print(f"  *** ALERT TRIGGERED: {result['status'].upper()} ***")
            # In the real system, this calls Module 5 (alert engine)

        run_face_verification_loop(cid, alert_callback=sample_alert)

    else:
        print("Invalid choice.")
