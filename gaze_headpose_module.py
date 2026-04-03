"""
Module 3: Gaze & Head Pose Tracking
======================================
Project : Online Exam Proctoring System Based on AI
Authors : Ramkumar R, Sri Lakshmi T, Subiksha V
College : Shree Venkateshwara Hi-Tech Engineering College

Detects:
    - Eye blink rate & drowsiness  (Eye Aspect Ratio — EAR)
    - Head looking left/right/up/down  (solvePnP head pose)
    - Lip movement  (Mouth Aspect Ratio — MAR)

INSTALL:
    pip install opencv-python dlib numpy imutils
    Download dlib 68-point model:
        http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    Extract and place shape_predictor_68_face_landmarks.dat in the same folder.
"""

import cv2
import dlib
import numpy as np
import time
from datetime import datetime
from imutils import face_utils


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"

# EAR thresholds
EAR_THRESHOLD       = 0.20   # below this = eye closed
EAR_CONSEC_FRAMES   = 15     # closed for this many frames = drowsy alert

# MAR threshold (lip movement / talking)
MAR_THRESHOLD       = 0.65   # above this = mouth open (possible talking)
MAR_CONSEC_FRAMES   = 8      # open for this many frames = lip movement alert

# Head pose thresholds (degrees)
YAW_THRESHOLD       = 30     # left/right turn
PITCH_THRESHOLD     = 20     # up/down tilt
POSE_CONSEC_FRAMES  = 10     # frames before alerting

# Landmark index ranges (dlib 68-point model)
LEFT_EYE_IDX   = (42, 48)
RIGHT_EYE_IDX  = (36, 42)
MOUTH_IDX      = (48, 68)

# 3D model points for solvePnP (generic human face, mm scale)
FACE_3D_POINTS = np.array([
    (  0.0,    0.0,    0.0),   # nose tip          — point 30
    (  0.0,  -63.6,  -12.5),   # chin              — point  8
    (-43.3,   32.7,  -26.0),   # left eye corner   — point 36
    ( 43.3,   32.7,  -26.0),   # right eye corner  — point 45
    (-28.9,  -28.9,  -24.1),   # left mouth corner — point 48
    ( 28.9,  -28.9,  -24.1),   # right mouth corner— point 54
], dtype=np.float64)

FACE_2D_INDICES = [30, 8, 36, 45, 48, 54]


# ═══════════════════════════════════════════════
#  PART 1 — LOAD MODELS
# ═══════════════════════════════════════════════

def load_models():
    """
    Loads the dlib face detector and 68-point landmark predictor.
    Returns (detector, predictor).
    """
    print("[GAZE] Loading dlib models...")
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(LANDMARK_MODEL)
    print("[GAZE] Models loaded.")
    return detector, predictor


# ═══════════════════════════════════════════════
#  PART 2 — FEATURE CALCULATIONS
# ═══════════════════════════════════════════════

def _eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    """
    EAR = (vertical distances) / (2 × horizontal distance)
    A high value = open eye. Low value = closed.
    """
    # Vertical distances
    v1 = np.linalg.norm(eye_pts[1] - eye_pts[5])
    v2 = np.linalg.norm(eye_pts[2] - eye_pts[4])
    # Horizontal distance
    h  = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (v1 + v2) / (2.0 * h)


def _mouth_aspect_ratio(mouth_pts: np.ndarray) -> float:
    """
    MAR measures how open the mouth is.
    Uses the inner lip landmarks for accuracy.
    """
    v1 = np.linalg.norm(mouth_pts[2]  - mouth_pts[10])  # 51–61
    v2 = np.linalg.norm(mouth_pts[4]  - mouth_pts[8])   # 53–59
    h  = np.linalg.norm(mouth_pts[0]  - mouth_pts[6])   # 48–54
    return (v1 + v2) / (2.0 * h)


def _estimate_head_pose(landmarks: np.ndarray, frame_shape: tuple) -> dict:
    """
    Uses OpenCV solvePnP to estimate yaw, pitch, roll from
    6 facial landmark points matched to a 3D face model.

    Returns dict with yaw, pitch, roll in degrees, and success flag.
    """
    h, w = frame_shape[:2]
    focal  = w
    center = (w / 2, h / 2)
    cam_matrix = np.array([
        [focal,     0, center[0]],
        [    0, focal, center[1]],
        [    0,     0,          1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    # Extract the 6 2D reference points from landmarks
    pts_2d = np.array(
        [landmarks[i] for i in FACE_2D_INDICES],
        dtype=np.float64
    )

    success, rot_vec, trans_vec = cv2.solvePnP(
        FACE_3D_POINTS, pts_2d,
        cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return {"success": False, "yaw": 0, "pitch": 0, "roll": 0}

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)

    yaw   = angles[1] * 360   # left/right
    pitch = angles[0] * 360   # up/down
    roll  = angles[2] * 360   # tilt

    return {
        "success": True,
        "yaw":   round(yaw,   2),
        "pitch": round(pitch, 2),
        "roll":  round(roll,  2),
        "rot_vec":   rot_vec,
        "trans_vec": trans_vec,
        "cam_matrix":  cam_matrix,
        "dist_coeffs": dist_coeffs
    }


# ═══════════════════════════════════════════════
#  PART 3 — ANALYSE ONE FRAME
# ═══════════════════════════════════════════════

def analyse_frame(frame: np.ndarray,
                  detector, predictor,
                  state: dict) -> dict:
    """
    Runs EAR, MAR, and head pose analysis on a single frame.
    Maintains persistent counters via the `state` dict so
    alerts only fire after sustained violations.

    Args:
        frame     : BGR webcam frame
        detector  : dlib face detector
        predictor : dlib shape predictor
        state     : mutable dict carrying per-session counters

    Returns a result dict with violations, angles, and EAR/MAR values.
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    violations = []
    result = {
        "face_found":  False,
        "ear":         None,
        "mar":         None,
        "yaw":         None,
        "pitch":       None,
        "roll":        None,
        "violations":  violations,
        "status":      "clean",
        "landmarks":   None,
        "pose_data":   None
    }

    if len(faces) == 0:
        return result

    # Take the largest detected face
    face = max(faces, key=lambda r: r.width() * r.height())
    shape = predictor(gray, face)
    landmarks = face_utils.shape_to_np(shape)

    result["face_found"] = True
    result["landmarks"]  = landmarks

    # ── EAR — blink / drowsiness ──
    l_eye = landmarks[LEFT_EYE_IDX[0]:LEFT_EYE_IDX[1]]
    r_eye = landmarks[RIGHT_EYE_IDX[0]:RIGHT_EYE_IDX[1]]
    ear   = (_eye_aspect_ratio(l_eye) + _eye_aspect_ratio(r_eye)) / 2.0
    result["ear"] = round(ear, 3)

    if ear < EAR_THRESHOLD:
        state["ear_counter"] += 1
    else:
        state["ear_counter"] = 0

    if state["ear_counter"] >= EAR_CONSEC_FRAMES:
        violations.append({
            "type":    "drowsy",
            "message": f"Eyes closed for {state['ear_counter']} frames (EAR={ear:.3f})"
        })

    # ── MAR — lip movement / talking ──
    mouth = landmarks[MOUTH_IDX[0]:MOUTH_IDX[1]]
    mar   = _mouth_aspect_ratio(mouth)
    result["mar"] = round(mar, 3)

    if mar > MAR_THRESHOLD:
        state["mar_counter"] += 1
    else:
        state["mar_counter"] = 0

    if state["mar_counter"] >= MAR_CONSEC_FRAMES:
        violations.append({
            "type":    "lip_movement",
            "message": f"Lip movement detected for {state['mar_counter']} frames (MAR={mar:.3f})"
        })

    # ── Head Pose — looking away ──
    pose = _estimate_head_pose(landmarks, frame.shape)
    result["pose_data"] = pose

    if pose["success"]:
        yaw   = pose["yaw"]
        pitch = pose["pitch"]
        result["yaw"]   = yaw
        result["pitch"] = pitch
        result["roll"]  = pose["roll"]

        looking_away = abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD

        if looking_away:
            state["pose_counter"] += 1
        else:
            state["pose_counter"] = 0

        if state["pose_counter"] >= POSE_CONSEC_FRAMES:
            direction = _gaze_direction(yaw, pitch)
            violations.append({
                "type":    "looking_away",
                "message": f"Looking {direction} (yaw={yaw:.1f}°, pitch={pitch:.1f}°)"
            })

    result["status"] = "violation" if violations else "clean"
    return result


def _gaze_direction(yaw: float, pitch: float) -> str:
    """Returns a human-readable gaze direction string."""
    if abs(yaw) > YAW_THRESHOLD:
        return "left" if yaw < 0 else "right"
    if pitch > PITCH_THRESHOLD:
        return "up"
    if pitch < -PITCH_THRESHOLD:
        return "down"
    return "away"


def make_state() -> dict:
    """Creates a fresh state dict for a new monitoring session."""
    return {
        "ear_counter":  0,
        "mar_counter":  0,
        "pose_counter": 0
    }


# ═══════════════════════════════════════════════
#  PART 4 — DRAW OVERLAY ON FRAME
# ═══════════════════════════════════════════════

def draw_gaze_overlay(frame: np.ndarray, result: dict) -> np.ndarray:
    """
    Draws landmarks, head pose axes, and a status banner
    on the webcam frame.
    """
    display = frame.copy()

    # Draw 68 landmark dots
    if result["landmarks"] is not None:
        for (x, y) in result["landmarks"]:
            cv2.circle(display, (x, y), 1, (0, 200, 200), -1)

    # Draw head pose axis line
    pose = result.get("pose_data")
    if pose and pose.get("success") and result["landmarks"] is not None:
        nose_tip = tuple(result["landmarks"][30])
        axis_pts, _ = cv2.projectPoints(
            np.array([(0, 0, 50.0)]),
            pose["rot_vec"], pose["trans_vec"],
            pose["cam_matrix"], pose["dist_coeffs"]
        )
        axis_end = tuple(axis_pts[0].ravel().astype(int))
        cv2.arrowedLine(display, nose_tip, axis_end, (0, 255, 255), 2, tipLength=0.3)

    # Status banner
    is_clean = result["status"] == "clean"
    color    = (0, 200, 0) if is_clean else (0, 0, 255)

    if is_clean:
        msg = "Gaze OK"
    else:
        msg = " | ".join(v["message"] for v in result["violations"])

    cv2.rectangle(display, (0, 0), (display.shape[1], 45), (30, 30, 30), -1)
    cv2.putText(display, msg, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

    # EAR / MAR / Angles readout (bottom left)
    y0 = display.shape[0] - 80
    for i, line in enumerate([
        f"EAR: {result['ear']}   MAR: {result['mar']}",
        f"Yaw: {result['yaw']}°   Pitch: {result['pitch']}°   Roll: {result['roll']}°"
    ]):
        cv2.putText(display, line, (10, y0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return display


# ═══════════════════════════════════════════════
#  PART 5 — REAL-TIME MONITORING LOOP
# ═══════════════════════════════════════════════

def run_gaze_monitoring_loop(alert_callback=None):
    """
    Runs gaze & head pose monitoring from the webcam continuously.
    Analyses every frame for EAR/MAR/pose, fires alert_callback
    on sustained violations.

    Press Q to stop.
    """
    detector, predictor = load_models()
    cap   = cv2.VideoCapture(0)
    state = make_state()

    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    print("[GAZE] Monitoring started. Press Q to stop.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = analyse_frame(frame, detector, predictor, state)

        if result["status"] == "violation":
            ts = datetime.now().strftime("%H:%M:%S")
            for v in result["violations"]:
                print(f"[{ts}] VIOLATION — {v['type']}: {v['message']}")
            if alert_callback:
                alert_callback(result, frame)

        display = draw_gaze_overlay(frame, result)
        cv2.imshow("Gaze & Head Pose — Proctoring System", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[GAZE] Stopped.")
            break

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════
#  QUICK TEST
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("  Gaze & Head Pose Module — Test Mode")
    print("=" * 50)

    def sample_alert(result, frame):
        types = [v["type"] for v in result["violations"]]
        print(f"  *** ALERT: {types} ***")

    run_gaze_monitoring_loop(alert_callback=sample_alert)
