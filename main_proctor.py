"""
Main Proctoring System — Fully Standalone
==========================================
Project : Online Exam Proctoring System Based on AI
Authors : Ramkumar R, Sri Lakshmi T, Subiksha V
College : Shree Venkateshwara Hi-Tech Engineering College

THIS FILE IS SELF-CONTAINED.
All detection logic is built in — no imports from other modules needed.
Just run:  python main_proctor.py

INSTALL (run once):
    pip install opencv-python face-recognition dlib ultralytics
    pip install numpy pillow mysql-connector-python
    pip install pyaudio speechrecognition scipy
    pip install cmake   (Windows — needed before dlib)
"""

import cv2
import numpy as np
import threading
import time
import os
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from datetime import datetime

# ── Optional imports — system still runs if these are missing ──
try:
    import face_recognition
    FACE_REC_OK = True
except ImportError:
    FACE_REC_OK = False
    print("[WARN] face_recognition not installed — face check disabled")

try:
    import dlib
    DLIB_OK = True
except ImportError:
    DLIB_OK = False
    print("[WARN] dlib not installed — gaze check disabled")

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
    print("[WARN] ultralytics not installed — object detection disabled")

try:
    import pyaudio, speech_recognition as sr
    AUDIO_OK = True
except ImportError:
    AUDIO_OK = False
    print("[WARN] pyaudio/speechrecognition not installed — audio disabled")

try:
    import mysql.connector
    DB_OK = True
except ImportError:
    DB_OK = False
    print("[WARN] mysql-connector not installed — DB logging disabled")

try:
    from scipy.spatial import distance as dist
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# ─────────────────────────────────────────────
#  CONFIG — update DB password here
# ─────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "password",  
    "database": "proctoring_db"
}

FACE_THRESHOLD   = 0.6
EAR_THRESHOLD    = 0.22
YAW_LIMIT        = 20
PITCH_LIMIT      = 15
NOISE_THRESHOLD  = 500
LANDMARK_MODEL   = "shape_predictor_68_face_landmarks.dat"

LEFT_EYE_IDX  = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))
POSE_INDICES  = [30, 8, 36, 45, 48, 54]
FACE_3D_PTS   = np.array([
    (0.0,    0.0,    0.0),
    (0.0,  -330.0, -65.0),
    (-225.0, 170.0,-135.0),
    (225.0,  170.0,-135.0),
    (-150.0,-150.0,-125.0),
    (150.0, -150.0,-125.0),
], dtype=np.float64)

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

SCREENSHOT_DIR = "violation_screenshots"


# ═══════════════════════════════════════════════
#  DATABASE HELPERS
# ═══════════════════════════════════════════════

def db_connect():
    if not DB_OK:
        return None
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[DB] {e}")
        return None


def db_setup():
    if not DB_OK:
        return
    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        cur = conn.cursor()
        cur.execute("CREATE DATABASE IF NOT EXISTS proctoring_db;")
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[DB setup] {e}")
        return

    sqls = [
        """CREATE TABLE IF NOT EXISTS candidates (
            candidate_id VARCHAR(20) PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            face_embedding BLOB NOT NULL,
            enrolled_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );""",
        """CREATE TABLE IF NOT EXISTS exam_sessions (
            session_id VARCHAR(50) PRIMARY KEY,
            candidate_id VARCHAR(20),
            exam_name VARCHAR(100),
            started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            ended_at DATETIME DEFAULT NULL,
            violation_count INT DEFAULT 0,
            status ENUM('active','completed','flagged') DEFAULT 'active'
        );""",
        """CREATE TABLE IF NOT EXISTS event_logs (
            log_id INT AUTO_INCREMENT PRIMARY KEY,
            candidate_id VARCHAR(20),
            session_id VARCHAR(50),
            event_type VARCHAR(50),
            description TEXT,
            severity ENUM('LOW','MEDIUM','HIGH') DEFAULT 'MEDIUM',
            screenshot_path VARCHAR(255) DEFAULT '',
            logged_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );""",
        """CREATE TABLE IF NOT EXISTS results (
            result_id INT AUTO_INCREMENT PRIMARY KEY,
            candidate_id VARCHAR(20),
            session_id VARCHAR(50),
            total_violations INT DEFAULT 0,
            high_severity INT DEFAULT 0,
            verdict ENUM('PASS','FLAGGED','FAIL') DEFAULT 'PASS',
            generated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );"""
    ]
    conn = db_connect()
    if not conn:
        return
    try:
        cur = conn.cursor()
        for sql in sqls:
            cur.execute(sql)
        conn.commit()
        print("[DB] Tables ready.")
    except Exception as e:
        print(f"[DB setup tables] {e}")
    finally:
        cur.close()
        conn.close()


def db_log_event(candidate_id, session_id, event_type, description, screenshot=""):
    conn = db_connect()
    if not conn:
        return
    severity = SEVERITY.get(event_type, "LOW")
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO event_logs
              (candidate_id, session_id, event_type, description, severity, screenshot_path)
            VALUES (%s,%s,%s,%s,%s,%s)
        """, (candidate_id, session_id, event_type, description, severity, screenshot))
        cur.execute("""
            UPDATE exam_sessions SET violation_count = violation_count+1
            WHERE session_id=%s
        """, (session_id,))
        conn.commit()
    except Exception as e:
        print(f"[DB log] {e}")
    finally:
        cur.close()
        conn.close()


def db_create_session(candidate_id, exam_name):
    sid = f"{candidate_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    conn = db_connect()
    if not conn:
        return sid
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO exam_sessions (session_id, candidate_id, exam_name)
            VALUES (%s,%s,%s)
        """, (sid, candidate_id, exam_name))
        conn.commit()
    except Exception as e:
        print(f"[DB session] {e}")
    finally:
        cur.close()
        conn.close()
    return sid


def db_end_session(session_id, candidate_id):
    conn = db_connect()
    if not conn:
        return "N/A", 0
    try:
        cur = conn.cursor()
        cur.execute("SELECT violation_count FROM exam_sessions WHERE session_id=%s",
                    (session_id,))
        row   = cur.fetchone()
        count = row[0] if row else 0
        status = "flagged" if count >= 5 else "completed"
        verdict = "PASS" if count < 3 else ("FLAGGED" if count < 10 else "FAIL")

        cur.execute("""
            UPDATE exam_sessions SET ended_at=%s, status=%s WHERE session_id=%s
        """, (datetime.now(), status, session_id))
        cur.execute("""
            INSERT INTO results (candidate_id, session_id, total_violations, verdict)
            VALUES (%s,%s,%s,%s)
        """, (candidate_id, session_id, count, verdict))
        conn.commit()
        return verdict, count
    except Exception as e:
        print(f"[DB end] {e}")
        return "N/A", 0
    finally:
        cur.close()
        conn.close()


def db_save_embedding(candidate_id, name, embedding):
    conn = db_connect()
    if not conn:
        return False
    try:
        cur  = conn.cursor()
        blob = pickle.dumps(embedding)
        cur.execute("""
            INSERT INTO candidates (candidate_id, name, face_embedding)
            VALUES (%s,%s,%s)
            ON DUPLICATE KEY UPDATE face_embedding=VALUES(face_embedding)
        """, (candidate_id, name, blob))
        conn.commit()
        return True
    except Exception as e:
        print(f"[DB embed] {e}")
        return False
    finally:
        cur.close()
        conn.close()


def db_load_embedding(candidate_id):
    conn = db_connect()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        cur.execute("SELECT face_embedding FROM candidates WHERE candidate_id=%s",
                    (candidate_id,))
        row = cur.fetchone()
        return pickle.loads(row[0]) if row else None
    except Exception as e:
        print(f"[DB load emb] {e}")
        return None
    finally:
        cur.close()
        conn.close()


# ═══════════════════════════════════════════════
#  DETECTION HELPERS
# ═══════════════════════════════════════════════

def _ear(eye):
    if not SCIPY_OK:
        return 0.3
    v1 = dist.euclidean(eye[1], eye[5])
    v2 = dist.euclidean(eye[2], eye[4])
    h  = dist.euclidean(eye[0], eye[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


def check_face(frame, stored_emb):
    """Returns (status, message, face_box)"""
    if not FACE_REC_OK or stored_emb is None:
        return "skip", "Face check disabled", None
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, model="hog")
    if len(locs) == 0:
        return "no_face", "No face detected", None
    if len(locs) > 1:
        return "multiple_faces", f"{len(locs)} faces in frame", locs[0]
    encs = face_recognition.face_encodings(rgb, locs)
    if not encs:
        return "no_face", "Encoding failed", None
    d = float(np.linalg.norm(stored_emb - encs[0]))
    if d < FACE_THRESHOLD:
        return "match", f"Identity OK (dist={d:.3f})", locs[0]
    return "mismatch", f"Identity MISMATCH (dist={d:.3f})", locs[0]


def check_objects(frame, yolo_model):
    """Returns list of violation dicts and annotated frame"""
    if not YOLO_OK or yolo_model is None:
        return [], frame
    results    = yolo_model(frame, verbose=False, conf=0.5)[0]
    violations = []
    person_cnt = 0
    display    = frame.copy()

    FORBIDDEN = {67: "cell phone", 73: "book", 0: "person"}
    COLORS    = {"cell phone": (0,0,255), "book": (0,140,255), "person": (255,0,0)}

    for box in results.boxes:
        cid   = int(box.cls[0])
        conf  = float(box.conf[0])
        label = yolo_model.names[cid]
        if cid == 0:
            person_cnt += 1
        if cid in FORBIDDEN:
            x1,y1,x2,y2 = [int(c) for c in box.xyxy[0].tolist()]
            color = COLORS.get(label, (200,200,200))
            cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
            cv2.putText(display, f"{label} {conf:.0%}",
                        (x1+4, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255,255,255), 1)
            if cid == 67:
                violations.append(("phone_detected", f"Phone at {conf:.0%}"))
            elif cid == 73:
                violations.append(("book_detected", f"Book at {conf:.0%}"))

    if person_cnt > 1:
        violations.append(("extra_person", f"{person_cnt} persons in frame"))

    return violations, display


def check_gaze(frame, detector, predictor):
    """Returns list of violation strings"""
    if not DLIB_OK or detector is None:
        return []
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    if not faces:
        return []

    shape  = predictor(gray, faces[0])
    coords = np.array([(shape.part(i).x, shape.part(i).y)
                       for i in range(68)], dtype=np.float32)

    # EAR
    le  = coords[LEFT_EYE_IDX]
    re  = coords[RIGHT_EYE_IDX]
    ear = (_ear(le) + _ear(re)) / 2.0

    violations = []
    if ear < EAR_THRESHOLD:
        violations.append(("eyes_closed", f"Eyes closed (EAR={ear:.3f})"))

    # Head pose
    h, w = frame.shape[:2]
    focal   = w
    cam_mtx = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float64)
    dist_c  = np.zeros((4,1))
    pts_2d  = np.array([(shape.part(i).x, shape.part(i).y)
                        for i in POSE_INDICES], dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(FACE_3D_PTS, pts_2d, cam_mtx, dist_c,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        rot,_  = cv2.Rodrigues(rvec)
        proj   = np.hstack([rot, tvec])
        _,_,_,_,_,_, euler = cv2.decomposeProjectionMatrix(proj)
        yaw, pitch = float(euler[1].item()), float(euler[0].item())
        if abs(yaw) > YAW_LIMIT:
            dir_ = "right" if yaw > 0 else "left"
            violations.append(("looking_away",
                                f"Head turned {dir_} (yaw={yaw:.1f}°)"))
        elif abs(pitch) > PITCH_LIMIT:
            violations.append(("looking_away",
                                f"Head tilted (pitch={pitch:.1f}°)"))

    return violations


# ═══════════════════════════════════════════════
#  PROCTOR ENGINE  (runs in background thread)
# ═══════════════════════════════════════════════

class ProctorEngine:
    def __init__(self, candidate_id, session_id, on_frame, on_alert):
        self.candidate_id = candidate_id
        self.session_id   = session_id
        self.on_frame     = on_frame
        self.on_alert     = on_alert

        self._stop       = threading.Event()
        self._cap        = None
        self._emb        = None
        self._yolo       = None
        self._detector   = None
        self._predictor  = None
        self._viol_count = 0

        self._t_face   = 0
        self._t_obj    = 0
        self._t_gaze   = 0
        self._t_audio  = 0

        self._last_face_status = ("match", "Initialising...", None)
        self._gaze_bad_frames  = 0

    def load(self):
        """Load all models. Raises RuntimeError if face emb missing."""
        print("[ENGINE] Loading models...")

        self._emb = db_load_embedding(self.candidate_id)
        if self._emb is None and FACE_REC_OK:
            raise RuntimeError(
                f"Candidate '{self.candidate_id}' not found in database.\n"
                "Please enroll the candidate first using the Enroll tab."
            )

        if YOLO_OK:
            try:
                self._yolo = YOLO("yolov8n.pt")
                print("[ENGINE] YOLO loaded.")
            except Exception as e:
                print(f"[ENGINE] YOLO load failed: {e}")

        if DLIB_OK and os.path.exists(LANDMARK_MODEL):
            self._detector  = dlib.get_frontal_face_detector()
            self._predictor = dlib.shape_predictor(LANDMARK_MODEL)
            print("[ENGINE] dlib loaded.")
        elif DLIB_OK:
            print(f"[ENGINE] {LANDMARK_MODEL} not found — gaze disabled.")

        print("[ENGINE] Ready.")

    def start(self):
        self._stop.clear()
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open webcam. Check that it is connected.")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._stop.set()
        time.sleep(0.3)
        if self._cap:
            self._cap.release()

    def _fire(self, event_type, description, frame):
        self._viol_count += 1
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = ""
        if frame is not None:
            path = os.path.join(SCREENSHOT_DIR,
                                f"{self.candidate_id}_{event_type}_{ts}.jpg")
            cv2.imwrite(path, frame)

        db_log_event(self.candidate_id, self.session_id,
                     event_type, description, path)
        self.on_alert(event_type, description,
                      self._viol_count, SEVERITY.get(event_type, "LOW"))

    def _loop(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            now      = time.time()
            display  = frame.copy()

            # ── Face check every 3 s ──
            if now - self._t_face >= 3:
                status, msg, box = check_face(frame, self._emb)
                self._last_face_status = (status, msg, box)
                self._t_face = now
                if status not in ("match", "skip"):
                    self._fire(status, msg, frame)
                if box:
                    top, right, bottom, left = box
                    color = (0,200,0) if status=="match" else (0,0,255)
                    cv2.rectangle(display, (left,top), (right,bottom), color, 2)

            # ── Object detection every 2 s ──
            if now - self._t_obj >= 2:
                obj_viols, display = check_objects(display, self._yolo)
                self._t_obj = now
                for etype, edesc in obj_viols:
                    self._fire(etype, edesc, frame)

            # ── Gaze check every 0.15 s ──
            if now - self._t_gaze >= 0.15:
                gaze_viols = check_gaze(frame, self._detector, self._predictor)
                self._t_gaze = now
                if gaze_viols:
                    self._gaze_bad_frames += 1
                    if self._gaze_bad_frames >= 15:
                        for etype, edesc in gaze_viols:
                            self._fire(etype, edesc, frame)
                        self._gaze_bad_frames = 0
                else:
                    self._gaze_bad_frames = 0

            # ── Audio check every 4 s ──
            if AUDIO_OK and now - self._t_audio >= 4:
                self._t_audio = now
                threading.Thread(target=self._audio_check,
                                 args=(frame.copy(),), daemon=True).start()

            # ── Status banner ──
            _, msg, _ = self._last_face_status
            status    = self._last_face_status[0]
            color     = (0,200,0) if status in ("match","skip") else (0,0,255)
            cv2.rectangle(display, (0,0), (display.shape[1], 44),
                          (20,20,35), -1)
            cv2.putText(display, msg, (10,29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(display, f"Violations: {self._viol_count}",
                        (display.shape[1]-180, 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

            self.on_frame(display)

    def _audio_check(self, frame):
        try:
            recognizer = sr.Recognizer()
            mic        = sr.Microphone(sample_rate=16000)
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = recognizer.listen(source, timeout=3,
                                          phrase_time_limit=3)
            text = recognizer.recognize_google(audio).lower()
            KEYWORDS = ["answer","question","help","tell me",
                        "solution","correct","send"]
            matched = [k for k in KEYWORDS if k in text]
            if text:
                etype = "suspicious_speech" if matched else "speech_detected"
                self._fire(etype, f"Heard: '{text}'", frame)
        except Exception:
            pass


# ═══════════════════════════════════════════════
#  TKINTER DASHBOARD
# ═══════════════════════════════════════════════

class App(tk.Tk):

    # ── Colour palette ──
    BG      = "#0f0f1a"
    PANEL   = "#1a1a2e"
    CARD    = "#16213e"
    ACCENT  = "#e94560"
    GREEN   = "#0f9b58"
    TEXT    = "#eaeaea"
    MUTED   = "#7a7a9a"
    HIGH_C  = "#ff4757"
    MED_C   = "#ffa502"
    LOW_C   = "#2ed573"

    def __init__(self):
        super().__init__()
        self.title("AI Online Exam Proctoring System")
        self.configure(bg=self.BG)
        self.geometry("1100x680")
        self.resizable(True, True)
        self.minsize(900, 600)

        self._engine     = None
        self._session_id = None
        self._running    = False
        self._frame_img  = None
        self._viol_count = 0

        db_setup()
        self._build_ui()
        self._show_idle_frame()

    # ─────────────────────────────────────────────
    #  UI CONSTRUCTION
    # ─────────────────────────────────────────────
    def _build_ui(self):
        # ── Header bar ──
        hdr = tk.Frame(self, bg=self.ACCENT, height=48)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  AI Online Exam Proctoring System",
                 font=("Helvetica", 14, "bold"),
                 bg=self.ACCENT, fg="white").pack(side="left",
                                                   pady=10, padx=8)
        self._clock_lbl = tk.Label(hdr, text="",
                                    font=("Helvetica", 11),
                                    bg=self.ACCENT, fg="white")
        self._clock_lbl.pack(side="right", padx=16)
        self._tick_clock()

        # ── Notebook tabs ──
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",        background=self.BG,  borderwidth=0)
        style.configure("TNotebook.Tab",    background=self.PANEL,
                         foreground=self.MUTED, padding=[14,6],
                         font=("Helvetica", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", self.CARD)],
                  foreground=[("selected", self.TEXT)])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        self._tab_monitor = tk.Frame(nb, bg=self.BG)
        self._tab_enroll  = tk.Frame(nb, bg=self.BG)
        nb.add(self._tab_monitor, text="  Live Monitor  ")
        nb.add(self._tab_enroll,  text="  Enroll Candidate  ")

        self._build_monitor_tab()
        self._build_enroll_tab()

    # ── Monitor tab ──────────────────────────────
    def _build_monitor_tab(self):
        tab = self._tab_monitor

        # Left column: video + controls
        left = tk.Frame(tab, bg=self.BG)
        left.pack(side="left", fill="both", expand=True, padx=12, pady=12)

        # Video canvas (640×480)
        self._canvas = tk.Canvas(left, width=640, height=480,
                                  bg="#050510", highlightthickness=1,
                                  highlightbackground=self.ACCENT)
        self._canvas.pack()

        # Controls row
        ctrl = tk.Frame(left, bg=self.BG)
        ctrl.pack(fill="x", pady=(10,0))

        # Candidate ID input
        tk.Label(ctrl, text="Candidate ID:",
                 font=("Helvetica",10), bg=self.BG,
                 fg=self.MUTED).pack(side="left")
        self._cid_var = tk.StringVar()
        tk.Entry(ctrl, textvariable=self._cid_var,
                 font=("Helvetica",11), width=14,
                 bg=self.CARD, fg=self.TEXT,
                 insertbackground=self.TEXT,
                 relief="flat").pack(side="left", padx=(4,16))

        # Exam name input
        tk.Label(ctrl, text="Exam:",
                 font=("Helvetica",10), bg=self.BG,
                 fg=self.MUTED).pack(side="left")
        self._exam_var = tk.StringVar(value="Semester Exam")
        tk.Entry(ctrl, textvariable=self._exam_var,
                 font=("Helvetica",11), width=16,
                 bg=self.CARD, fg=self.TEXT,
                 insertbackground=self.TEXT,
                 relief="flat").pack(side="left", padx=(4,16))

        self._start_btn = tk.Button(
            ctrl, text="▶  Start",
            font=("Helvetica",10,"bold"),
            bg=self.GREEN, fg="white",
            relief="flat", padx=14, pady=6,
            cursor="hand2",
            command=self._start_session)
        self._start_btn.pack(side="left", padx=(0,8))

        self._stop_btn = tk.Button(
            ctrl, text="■  Stop",
            font=("Helvetica",10,"bold"),
            bg="#555", fg="white",
            relief="flat", padx=14, pady=6,
            cursor="hand2",
            state="disabled",
            command=self._stop_session)
        self._stop_btn.pack(side="left")

        # Status label
        self._status_lbl = tk.Label(
            left, text="● Idle — enter Candidate ID and press Start",
            font=("Helvetica",10), bg=self.BG, fg=self.MUTED)
        self._status_lbl.pack(anchor="w", pady=(8,0))

        # Right column: stats + log
        right = tk.Frame(tab, bg=self.BG, width=360)
        right.pack(side="right", fill="y", padx=(0,12), pady=12)
        right.pack_propagate(False)

        # Stats cards row
        stats_row = tk.Frame(right, bg=self.BG)
        stats_row.pack(fill="x", pady=(0,10))

        self._viol_card = self._stat_card(stats_row, "Violations", "0",
                                           self.HIGH_C)
        self._viol_card.pack(side="left", fill="x", expand=True, padx=(0,6))

        self._session_card = self._stat_card(stats_row, "Session", "--:--",
                                              self.MUTED)
        self._session_card.pack(side="left", fill="x", expand=True)

        # Log panel
        tk.Label(right, text="Alert log",
                 font=("Helvetica",10,"bold"),
                 bg=self.BG, fg=self.TEXT).pack(anchor="w", pady=(4,4))

        log_outer = tk.Frame(right, bg=self.CARD,
                              highlightthickness=1,
                              highlightbackground="#2a2a4a")
        log_outer.pack(fill="both", expand=True)

        self._log = tk.Text(
            log_outer,
            font=("Courier New", 9),
            bg=self.CARD, fg=self.TEXT,
            relief="flat", state="disabled",
            wrap="word", padx=8, pady=6,
            cursor="arrow"
        )
        sb = ttk.Scrollbar(log_outer, command=self._log.yview)
        self._log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._log.pack(fill="both", expand=True)

        # Severity colour tags
        self._log.tag_config("HIGH",   foreground=self.HIGH_C)
        self._log.tag_config("MEDIUM", foreground=self.MED_C)
        self._log.tag_config("LOW",    foreground=self.LOW_C)
        self._log.tag_config("INFO",   foreground="#74b9ff")
        self._log.tag_config("TIME",   foreground=self.MUTED)

    def _stat_card(self, parent, label, value, color):
        card = tk.Frame(parent, bg=self.CARD, padx=12, pady=10)
        tk.Label(card, text=label,
                 font=("Helvetica", 9), bg=self.CARD,
                 fg=self.MUTED).pack(anchor="w")
        val_lbl = tk.Label(card, text=value,
                            font=("Helvetica", 20, "bold"),
                            bg=self.CARD, fg=color)
        val_lbl.pack(anchor="w")
        card._val_lbl = val_lbl
        return card

    # ── Enroll tab ───────────────────────────────
    def _build_enroll_tab(self):
        tab = self._tab_enroll

        wrapper = tk.Frame(tab, bg=self.BG)
        wrapper.place(relx=0.5, rely=0.5, anchor="center")

        card = tk.Frame(wrapper, bg=self.CARD, padx=40, pady=36)
        card.pack()

        tk.Label(card, text="Enroll a new candidate",
                 font=("Helvetica", 14, "bold"),
                 bg=self.CARD, fg=self.TEXT).pack(pady=(0,20))

        # Fields
        for label, var_name, default in [
            ("Candidate ID", "_enroll_id",   ""),
            ("Full name",    "_enroll_name", ""),
        ]:
            tk.Label(card, text=label,
                     font=("Helvetica",10), bg=self.CARD,
                     fg=self.MUTED).pack(anchor="w")
            v = tk.StringVar(value=default)
            setattr(self, var_name, v)
            tk.Entry(card, textvariable=v,
                     font=("Helvetica",11), width=26,
                     bg=self.PANEL, fg=self.TEXT,
                     insertbackground=self.TEXT,
                     relief="flat").pack(pady=(2,12), ipady=4)

        self._enroll_status = tk.Label(card, text="",
                                        font=("Helvetica",10),
                                        bg=self.CARD, fg=self.MUTED)
        self._enroll_status.pack(pady=(4,12))

        tk.Button(card, text="Open webcam & enroll",
                  font=("Helvetica",11,"bold"),
                  bg=self.ACCENT, fg="white",
                  relief="flat", padx=20, pady=8,
                  cursor="hand2",
                  command=self._do_enroll).pack()

    # ─────────────────────────────────────────────
    #  ENROLLMENT LOGIC
    # ─────────────────────────────────────────────
    def _do_enroll(self):
        cid  = self._enroll_id.get().strip()
        name = self._enroll_name.get().strip()
        if not cid or not name:
            self._enroll_status.config(
                text="Please fill in both fields.", fg=self.HIGH_C)
            return
        if not FACE_REC_OK:
            self._enroll_status.config(
                text="face_recognition not installed.", fg=self.HIGH_C)
            return

        self._enroll_status.config(
            text="Webcam opening — press SPACE to capture, Q to cancel.",
            fg=self.MED_C)
        self.update()

        threading.Thread(target=self._enroll_thread,
                         args=(cid, name), daemon=True).start()

    def _enroll_thread(self, cid, name):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.after(0, lambda: self._enroll_status.config(
                text="Cannot open webcam.", fg=self.HIGH_C))
            return

        embedding = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb, model="hog")
            disp = frame.copy()
            for (t,r,b,l) in locs:
                cv2.rectangle(disp, (l,t), (r,b), (0,255,0), 2)
            cv2.putText(disp, "SPACE=capture  Q=cancel",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0), 2)
            cv2.imshow("Enrollment", disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            if key == ord(' '):
                if len(locs) != 1:
                    cv2.putText(disp,
                                "Need exactly 1 face!",
                                (10,65), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0,0,255), 2)
                    cv2.imshow("Enrollment", disp)
                    cv2.waitKey(800)
                    continue
                encs = face_recognition.face_encodings(rgb, locs)
                if encs:
                    embedding = encs[0]
                    cv2.putText(disp, "CAPTURED!", (10,65),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (0,255,0), 3)
                    cv2.imshow("Enrollment", disp)
                    cv2.waitKey(1000)
                    break

        cap.release()
        cv2.destroyAllWindows()

        if embedding is not None:
            ok = db_save_embedding(cid, name, embedding)
            msg = (f"✓ {name} enrolled successfully!" if ok
                   else "DB save failed — check DB_CONFIG password.")
            color = self.LOW_C if ok else self.HIGH_C
        else:
            msg, color = "Enrollment cancelled — no face captured.", self.MUTED

        self.after(0, lambda: self._enroll_status.config(
            text=msg, fg=color))

    # ─────────────────────────────────────────────
    #  SESSION CONTROL
    # ─────────────────────────────────────────────
    def _start_session(self):
        cid  = self._cid_var.get().strip()
        exam = self._exam_var.get().strip() or "Exam"
        if not cid:
            messagebox.showwarning("Input needed", "Enter a Candidate ID.")
            return

        try:
            self._session_id = db_create_session(cid, exam)
            self._engine     = ProctorEngine(
                candidate_id = cid,
                session_id   = self._session_id,
                on_frame     = self._on_frame,
                on_alert     = self._on_alert
            )
            self._engine.load()
            self._engine.start()
        except RuntimeError as e:
            messagebox.showerror("Error", str(e))
            return

        self._running      = True
        self._session_start = time.time()
        self._viol_count   = 0
        self._start_btn.config(state="disabled", bg="#555")
        self._stop_btn.config(state="normal", bg=self.HIGH_C)
        self._status_lbl.config(text=f"● Live — {cid} | {exam}",
                                 fg=self.LOW_C)
        self._log_add("Session started.", "INFO")
        self._update_session_timer()

    def _stop_session(self):
        self._running = False
        if self._engine:
            self._engine.stop()

        verdict, count = "N/A", 0
        if self._session_id and self._engine:
            verdict, count = db_end_session(
                self._session_id, self._engine.candidate_id)

        self._start_btn.config(state="normal",  bg=self.GREEN)
        self._stop_btn.config(state="disabled", bg="#555")
        self._status_lbl.config(text="● Idle", fg=self.MUTED)
        self._show_idle_frame()
        self._log_add("Session ended.", "INFO")

        messagebox.showinfo("Session complete",
                            f"Violations : {count}\n"
                            f"Verdict    : {verdict}")

    # ─────────────────────────────────────────────
    #  CALLBACKS FROM ENGINE
    # ─────────────────────────────────────────────
    def _on_frame(self, frame):
        """Called from engine thread — schedule UI update on main thread."""
        self.after(0, lambda: self._render_frame(frame))

    def _render_frame(self, frame):
        try:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb).resize((640, 480))
            photo = ImageTk.PhotoImage(image=img)
            self._canvas.create_image(0, 0, anchor="nw", image=photo)
            self._canvas.image = photo
        except Exception as e:
            print(f"[RENDER] {e}")

    def _on_alert(self, event_type, description, count, severity):
        self.after(0, lambda: self._handle_alert_ui(
            event_type, description, count, severity))

    def _handle_alert_ui(self, event_type, description, count, severity):
        self._viol_count = count
        # Update violation counter card
        self._viol_card._val_lbl.config(text=str(count))
        # Log entry
        ts  = datetime.now().strftime("%H:%M:%S")
        msg = f"[{ts}] {event_type}: {description}"
        self._log_add(msg, severity)

    # ─────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────
    def _show_idle_frame(self):
        idle = np.zeros((480, 640, 3), dtype=np.uint8)
        idle[:] = (15, 15, 26)
        cv2.putText(idle, "Proctoring System",
                    (150, 220), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (100, 100, 160), 2)
        cv2.putText(idle, "Enter Candidate ID and press Start",
                    (80, 265), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (60, 60, 100), 1)
        self._render_frame(idle)

    def _log_add(self, message, tag="INFO"):
        self._log.config(state="normal")
        self._log.insert("end", message + "\n", tag)
        self._log.see("end")
        self._log.config(state="disabled")

    def _update_session_timer(self):
        if not self._running:
            return
        elapsed = int(time.time() - self._session_start)
        m, s    = divmod(elapsed, 60)
        self._session_card._val_lbl.config(text=f"{m:02d}:{s:02d}")
        self.after(1000, self._update_session_timer)

    def _tick_clock(self):
        self._clock_lbl.config(
            text=datetime.now().strftime("%d %b %Y  %H:%M:%S"))
        self.after(1000, self._tick_clock)

    def on_close(self):
        if self._running:
            self._stop_session()
        self.destroy()


# ═══════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
