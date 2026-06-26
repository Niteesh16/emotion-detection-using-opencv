# Real-Time Emotion Detection using OpenCV and DeepFace

This is a simple, highly accurate Python project that detects emotions in real-time using your computer's webcam. It relies on **OpenCV** to locate faces and the **DeepFace** library to analyze the emotions.

## Features
- **Real-Time Detection:** Processes video frames continuously from your webcam.
- **High Accuracy:** Uses state-of-the-art Deep Learning models (via DeepFace) to predict emotions.
- **Supported Emotions:** Happy, Sad, Angry, Fear, Surprise, Disgust, and Neutral.

## Prerequisites
- Python 3.7+
- A working webcam

## Installation

1. **Clone this repository** (or download the files):
   ```bash
   git clone https://github.com/Niteesh16/emotion-detection-using-opencv.git
   cd emotion-detection-using-opencv
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: DeepFace might download some pre-trained model weights (around ~100MB) the very first time you run the script.*

## Usage

Run the main script:

```bash
python emotion_detector.py
```

- Your webcam will open in a new window.
- Make different facial expressions (smile, frown, act surprised, etc.).
- A green box will appear around your face, and the detected emotion will be displayed above it.
- **Press 'q'** on your keyboard to quit the application and close the window.

## How It Works
1. **Face Detection:** The `emotion_detector.py` script uses OpenCV's Haar Cascade Classifier (`haarcascade_frontalface_default.xml`) to quickly and efficiently find faces in the video frame.
2. **Emotion Analysis:** Once a face is found, the cropped face image is passed to `DeepFace.analyze()`. DeepFace uses a pre-trained Convolutional Neural Network (CNN) to predict the dominant emotion.
3. **Display:** OpenCV draws the bounding box and the emotion label onto the frame, which is then displayed on your screen.
