# AI Online Exam Proctoring System

Final Year Project — Bachelor of Engineering (Computer Science)
Shree Venkateshwara Hi-Tech Engineering College, Gobichettipalayam
Anna University, Chennai — May 2025

## Team
- Ramkumar R (732521104040)
- Sri Lakshmi T (732521104046)
- Subiksha V (732521104049)

## Description
An AI-powered online exam proctoring system that monitors students
in real-time using computer vision and audio analysis to detect
suspicious behavior and maintain academic integrity.

## Features
- Face detection and identity verification (dlib + face_recognition)
- Object detection — phone, books, extra person (YOLOv8)
- Gaze and head pose tracking (OpenCV solvePnP)
- Audio monitoring and speech detection (PyAudio + SpeechRecognition)
- Real-time alert engine with violation logging
- MySQL database with Excel export
- Tkinter instructor dashboard

## Installation

pip install cmake
pip install dlib
pip install -r requirements.txt


## Run

# Step 1 — Setup database
python database_setup.py

# Step 2 — Launch system
python main_proctor.py


## Note
Download `shape_predictor_68_face_landmarks.dat` separately:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2