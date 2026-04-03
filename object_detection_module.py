"""
Module 2: Object Detection (YOLOv8)
=====================================
Project : Online Exam Proctoring System Based on AI
Authors : Ramkumar R, Sri Lakshmi T, Subiksha V
College : Shree Venkateshwara Hi-Tech Engineering College

Detects forbidden objects in the webcam feed:
    - Mobile phone
    - Book / notebook
    - Extra person in frame

INSTALL:
    pip install ultralytics opencv-python numpy
    (YOLOv8 weights download automatically on first run ~6 MB)
"""

import cv2
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

# 'yolov8n.pt' = nano (fastest, less accurate)
# 'yolov8s.pt' = small (slightly slower, more accurate)
MODEL_PATH = "yolov8n.pt"

# Minimum confidence to count a detection (0.0 – 1.0)
CONFIDENCE_THRESHOLD = 0.5

# Seconds between each detection pass (reduce CPU load)
DETECTION_INTERVAL = 2

# COCO class IDs we care about
FORBIDDEN_CLASSES = {
    67: "cell phone",
    73: "book",
    0: "person",     # we allow exactly 1 — more = extra person alert
}

# BGR colours for bounding boxes
BOX_COLORS = {
    "cell phone": (0,   0, 255),   # red
    "book":       (0, 140, 255),   # orange
    "person":     (255,  0,   0),  # blue
}


# ═══════════════════════════════════════════════
#  PART 1 — LOAD MODEL
# ═══════════════════════════════════════════════

def load_yolo_model(model_path: str = MODEL_PATH) -> YOLO:
    """
    Loads (and auto-downloads if needed) the YOLOv8 model.
    First run downloads ~6 MB for the nano model.
    """
    print(f"[YOLO] Loading model: {model_path}")
    model = YOLO(model_path)
    print("[YOLO] Model ready.")
    return model


# ═══════════════════════════════════════════════
#  PART 2 — DETECT OBJECTS IN ONE FRAME
# ═══════════════════════════════════════════════

def detect_objects(frame: np.ndarray, model: YOLO) -> dict:
    """
    Runs YOLOv8 on a single BGR frame.
    Returns only detections relevant to exam proctoring.

    Returns a dict:
        status       : "clean" | "violation"
        violations   : list of violation dicts
        phone_found  : bool
        book_found   : bool
        extra_person : bool
        person_count : int
        raw_boxes    : list of all detected forbidden-class boxes
    """
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]

    raw_boxes    = []
    person_count = 0

    for box in results.boxes:
        cls_id     = int(box.cls[0])
        confidence = float(box.conf[0])
        label      = model.names[cls_id]
        coords     = [int(c) for c in box.xyxy[0].tolist()]  # [x1,y1,x2,y2]

        if cls_id == 0:
            person_count += 1

        if cls_id in FORBIDDEN_CLASSES:
            raw_boxes.append({
                "label":      label,
                "confidence": round(confidence, 3),
                "coords":     coords,
                "cls_id":     cls_id
            })

    violations   = []
    phone_found  = False
    book_found   = False
    extra_person = False

    for det in raw_boxes:
        cid = det["cls_id"]

        if cid == 67:
            phone_found = True
            violations.append({
                "type":       "phone_detected",
                "label":      det["label"],
                "confidence": det["confidence"],
                "coords":     det["coords"]
            })

        elif cid == 73:
            book_found = True
            violations.append({
                "type":       "book_detected",
                "label":      det["label"],
                "confidence": det["confidence"],
                "coords":     det["coords"]
            })

    if person_count > 1:
        extra_person = True
        violations.append({
            "type":       "extra_person",
            "label":      f"{person_count} persons in frame",
            "confidence": None,
            "coords":     None
        })

    return {
        "status":       "violation" if violations else "clean",
        "violations":   violations,
        "phone_found":  phone_found,
        "book_found":   book_found,
        "extra_person": extra_person,
        "person_count": person_count,
        "raw_boxes":    raw_boxes
    }


# ═══════════════════════════════════════════════
#  PART 3 — DRAW DETECTIONS ON FRAME
# ═══════════════════════════════════════════════

def draw_detections(frame: np.ndarray, result: dict) -> np.ndarray:
    """
    Draws bounding boxes, labels, and a status banner
    onto the webcam frame for the live monitoring view.
    """
    display = frame.copy()

    for det in result["raw_boxes"]:
        x1, y1, x2, y2 = det["coords"]
        label  = det["label"]
        conf   = det["confidence"]
        color  = BOX_COLORS.get(label, (200, 200, 200))

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        badge = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(display, badge, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Top status banner
    is_clean = result["status"] == "clean"
    color    = (0, 200, 0) if is_clean else (0, 0, 255)
    msg      = "No violations detected" if is_clean else \
               "VIOLATION: " + ", ".join(v["type"] for v in result["violations"])

    cv2.rectangle(display, (0, 0), (display.shape[1], 45), (30, 30, 30), -1)
    cv2.putText(display, msg, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Person count (bottom right)
    cv2.putText(display,
                f"Persons in frame: {result['person_count']}",
                (display.shape[1] - 200, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return display


# ═══════════════════════════════════════════════
#  PART 4 — REAL-TIME MONITORING LOOP
# ═══════════════════════════════════════════════

def run_object_detection_loop(alert_callback=None):
    """
    Continuously detects forbidden objects from the webcam.
    Fires alert_callback(result, frame) on every violation.

    Press Q to stop.
    """
    model = load_yolo_model()
    cap   = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    print("[YOLO] Detection loop started. Press Q to stop.\n")

    last_check  = 0
    last_result = {
        "status": "clean", "violations": [], "phone_found": False,
        "book_found": False, "extra_person": False,
        "person_count": 0, "raw_boxes": []
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()

        if now - last_check >= DETECTION_INTERVAL:
            last_result = detect_objects(frame, model)
            last_check  = now
            ts          = datetime.now().strftime("%H:%M:%S")

            if last_result["status"] == "violation":
                for v in last_result["violations"]:
                    print(f"[{ts}] VIOLATION — {v['type']}  ({v['label']})")
                if alert_callback:
                    alert_callback(last_result, frame)
            else:
                print(f"[{ts}] Clean — no violations.")

        display = draw_detections(frame, last_result)
        cv2.imshow("Object Detection — Proctoring System", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[YOLO] Stopped.")
            break

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════
#  PART 5 — TEST ON A SAVED IMAGE (no webcam)
# ═══════════════════════════════════════════════

def test_on_image(image_path: str):
    """
    Run detection on a saved image for quick offline testing.
    Useful to verify the model works before using the webcam.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not load: {image_path}")
        return

    model  = load_yolo_model()
    result = detect_objects(frame, model)

    print("\n── Detection Results ──")
    print(f"  Status       : {result['status']}")
    print(f"  Person count : {result['person_count']}")
    print(f"  Phone found  : {result['phone_found']}")
    print(f"  Book found   : {result['book_found']}")
    print(f"  Extra person : {result['extra_person']}")
    if result["violations"]:
        print("  Violations   :")
        for v in result["violations"]:
            conf = f"{v['confidence']:.0%}" if v["confidence"] else "n/a"
            print(f"    - {v['type']}  |  {v['label']}  |  conf={conf}")

    display = draw_detections(frame, result)
    cv2.imshow("Test Result — press any key to close", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════
#  QUICK TEST
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("  Object Detection Module — Test Mode")
    print("=" * 50)
    print("\n  1. Live webcam detection")
    print("  2. Test on a saved image")
    choice = input("\nEnter 1 or 2: ").strip()

    if choice == "1":
        def sample_alert(result, frame):
            types = [v["type"] for v in result["violations"]]
            print(f"  *** ALERT: {types} ***")

        run_object_detection_loop(alert_callback=sample_alert)

    elif choice == "2":
        path = input("Image file path: ").strip()
        test_on_image(path)

    else:
        print("Invalid choice.")
