"""
Module 4: Audio Monitoring
============================
Project : Online Exam Proctoring System Based on AI
Authors : Ramkumar R, Sri Lakshmi T, Subiksha V
College : Shree Venkateshwara Hi-Tech Engineering College

Detects:
    - Candidate speaking during exam
    - Background voices / multiple speakers
    - Sustained noise above threshold

INSTALL:
    pip install pyaudio speechrecognition numpy
    Windows extra: pip install pipwin && pipwin install pyaudio
"""

import pyaudio
import speech_recognition as sr
import numpy as np
import threading
import time
from datetime import datetime


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

# RMS energy level that counts as "speech detected"
NOISE_THRESHOLD     = 500

# How many seconds of audio to sample per check
CHUNK_DURATION      = 2

# How often to run the full speech-recognition check (seconds)
SR_INTERVAL         = 5

# Suspicious keywords — any of these trigger an extra flag
SUSPICIOUS_KEYWORDS = [
    "answer", "question", "what is", "tell me", "help",
    "solution", "correct", "wrong", "pass", "send"
]

SAMPLE_RATE  = 16000
CHANNELS     = 1
FORMAT       = pyaudio.paInt16
CHUNK_SIZE   = 1024


# ═══════════════════════════════════════════════
#  PART 1 — RMS ENERGY CHECK  (fast, no API)
# ═══════════════════════════════════════════════

def measure_audio_energy(duration: float = CHUNK_DURATION) -> dict:
    """
    Records a short clip and returns the RMS energy level.
    Fast — no internet needed. Used for continuous noise monitoring.

    Returns:
        energy        : float   (RMS amplitude)
        speech_likely : bool    (energy > NOISE_THRESHOLD)
        message       : str
    """
    p      = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
                    input=True, frames_per_buffer=CHUNK_SIZE)

    frames = []
    n_chunks = int(SAMPLE_RATE / CHUNK_SIZE * duration)
    for _ in range(n_chunks):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_np = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32)
    rms      = float(np.sqrt(np.mean(audio_np ** 2)))

    speech_likely = rms > NOISE_THRESHOLD
    return {
        "energy":        round(rms, 1),
        "speech_likely": speech_likely,
        "message":       f"Speech detected (energy={rms:.0f})" if speech_likely
                         else f"Quiet (energy={rms:.0f})"
    }


# ═══════════════════════════════════════════════
#  PART 2 — SPEECH RECOGNITION  (keyword check)
# ═══════════════════════════════════════════════

def transcribe_and_check(duration: float = CHUNK_DURATION) -> dict:
    """
    Records audio and transcribes it using Google Speech Recognition.
    Checks for suspicious exam-cheating keywords.

    Requires internet. Fires less frequently (every SR_INTERVAL seconds).

    Returns:
        transcript         : str   (what was heard, or "")
        speech_detected    : bool
        suspicious_keyword : bool
        matched_keywords   : list[str]
        message            : str
    """
    recognizer = sr.Recognizer()
    mic        = sr.Microphone(sample_rate=SAMPLE_RATE)

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        try:
            audio = recognizer.listen(source, timeout=duration,
                                      phrase_time_limit=duration)
        except sr.WaitTimeoutError:
            return {
                "transcript":         "",
                "speech_detected":    False,
                "suspicious_keyword": False,
                "matched_keywords":   [],
                "message":            "No speech heard in time window"
            }

    try:
        transcript = recognizer.recognize_google(audio).lower()
    except sr.UnknownValueError:
        return {
            "transcript":         "",
            "speech_detected":    False,
            "suspicious_keyword": False,
            "matched_keywords":   [],
            "message":            "Audio unclear / not speech"
        }
    except sr.RequestError as e:
        return {
            "transcript":         "",
            "speech_detected":    False,
            "suspicious_keyword": False,
            "matched_keywords":   [],
            "message":            f"Speech API error: {e}"
        }

    matched = [kw for kw in SUSPICIOUS_KEYWORDS if kw in transcript]
    suspicious = len(matched) > 0

    return {
        "transcript":         transcript,
        "speech_detected":    True,
        "suspicious_keyword": suspicious,
        "matched_keywords":   matched,
        "message":            (
            f"SUSPICIOUS speech: '{transcript}' (keywords: {matched})"
            if suspicious else
            f"Speech heard: '{transcript}'"
        )
    }


# ═══════════════════════════════════════════════
#  PART 3 — BACKGROUND MONITORING THREAD
# ═══════════════════════════════════════════════

class AudioMonitor:
    """
    Runs two parallel checks in a background thread:
      1. Continuous RMS energy check every CHUNK_DURATION seconds
      2. Full speech recognition every SR_INTERVAL seconds

    Usage:
        monitor = AudioMonitor(alert_callback=my_fn)
        monitor.start()
        ...
        monitor.stop()
    """

    def __init__(self, alert_callback=None):
        self.alert_callback  = alert_callback
        self._stop_event     = threading.Event()
        self._thread         = None
        self.last_result     = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[AUDIO] Monitoring started.")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        print("[AUDIO] Monitoring stopped.")

    def _loop(self):
        last_sr_check = 0

        while not self._stop_event.is_set():
            ts  = datetime.now().strftime("%H:%M:%S")
            now = time.time()

            # Fast RMS check every cycle
            energy_result = measure_audio_energy(CHUNK_DURATION)
            self.last_result = energy_result

            if energy_result["speech_likely"]:
                print(f"[{ts}] {energy_result['message']}")

                # Full SR check on a slower interval
                if now - last_sr_check >= SR_INTERVAL:
                    sr_result      = transcribe_and_check(CHUNK_DURATION)
                    last_sr_check  = now
                    self.last_result = sr_result
                    print(f"[{ts}] {sr_result['message']}")

                    if (sr_result["speech_detected"] or
                            sr_result["suspicious_keyword"]):
                        if self.alert_callback:
                            self.alert_callback(sr_result)
            else:
                print(f"[{ts}] {energy_result['message']}")


# ═══════════════════════════════════════════════
#  QUICK TEST
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("  Audio Monitoring Module — Test Mode")
    print("=" * 50)
    print("\n  1. Single energy snapshot")
    print("  2. Full speech recognition test")
    print("  3. Run continuous background monitor (30 sec)")
    choice = input("\nEnter 1, 2 or 3: ").strip()

    if choice == "1":
        print("\nRecording 2 seconds...")
        result = measure_audio_energy()
        print(f"  Energy       : {result['energy']}")
        print(f"  Speech likely: {result['speech_likely']}")
        print(f"  Message      : {result['message']}")

    elif choice == "2":
        print("\nListening for 3 seconds — say something...")
        result = transcribe_and_check(3)
        print(f"  Transcript : {result['transcript']}")
        print(f"  Suspicious : {result['suspicious_keyword']}")
        print(f"  Message    : {result['message']}")

    elif choice == "3":
        def on_alert(result):
            print(f"\n  *** AUDIO ALERT: {result['message']} ***\n")

        monitor = AudioMonitor(alert_callback=on_alert)
        monitor.start()
        time.sleep(30)
        monitor.stop()

    else:
        print("Invalid choice.")
