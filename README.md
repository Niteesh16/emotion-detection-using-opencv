# Real-Time Emotion Detection using OpenCV and DeepFace

A simple Python project that detects emotions in real-time using your computer's webcam. It uses **OpenCV** to find faces and **DeepFace** to analyze the emotion.

## Supported Emotions

Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral

## Requirements

- Python 3.7+
- A working webcam

## Installation

Install all required libraries by running:

```
pip install -r requirements.txt
```

## How to Run

```
python emotion_detector.py
```

Press **Q** to quit the window.

## How It Works

1. Opens your webcam
2. Detects faces in each frame using OpenCV
3. Crops the face and sends it to DeepFace
4. DeepFace returns the dominant emotion
5. Displays the emotion label above the face in real-time
