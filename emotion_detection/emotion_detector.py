import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import cv2
from deepface import DeepFace
import logging
import threading

logging.getLogger("deepface").setLevel(logging.ERROR)

current_emotion = "Detecting..."
emotion_probs = {}
is_analyzing = False

def analyze_emotion(frame_copy, x, y, w, h):
    global current_emotion, emotion_probs, is_analyzing
    
    # Large 40% padding for better alignment (vital for recognizing sadness)
    padding_y = int(h * 0.4)
    padding_x = int(w * 0.4)
    
    y1 = max(0, y - padding_y)
    y2 = min(frame_copy.shape[0], y + h + padding_y)
    x1 = max(0, x - padding_x)
    x2 = min(frame_copy.shape[1], x + w + padding_x)
    
    face_roi = frame_copy[y1:y2, x1:x2]
    
    try:
        # Use deepface with 'opencv' but tell it to align the face first
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, detector_backend='opencv', align=True)
        if isinstance(result, list):
            result = result[0]
            
        probs = result['emotion']
        
        # FER2013 Dataset Bias Correction
        # The AI model heavily under-represents sadness and over-represents neutral.
        # We boost the 'sad' probability slightly to make it more accurate to real life.
        if 'sad' in probs:
            probs['sad'] = probs['sad'] * 1.5 
            
        # Re-calculate the dominant emotion after our bias correction
        dominant = max(probs, key=probs.get)
        
        emotion_probs = probs
        current_emotion = dominant.capitalize()
    except Exception:
        pass
    finally:
        is_analyzing = False

def main():
    global current_emotion, emotion_probs, is_analyzing
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=6, minSize=(60, 60))

        # We only process the largest face to save processing power
        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            (x, y, w, h) = faces[0]
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Use threading so the webcam stays super smooth while AI runs in the background
            if not is_analyzing:
                is_analyzing = True
                frame_copy = frame.copy()
                threading.Thread(target=analyze_emotion, args=(frame_copy, x, y, w, h), daemon=True).start()

            cv2.putText(frame, current_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display the probabilities
            if emotion_probs:
                y_offset = 30
                sorted_emotions = sorted(emotion_probs.items(), key=lambda item: item[1], reverse=True)
                for emotion, prob in sorted_emotions:
                    text = f"{emotion.capitalize()}: {prob:.1f}%"
                    color = (0, 255, 0) if emotion.capitalize() == current_emotion else (255, 255, 255)
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (8, y_offset - text_h - 4), (12 + text_w, y_offset + 4), (0, 0, 0), -1)
                    cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 25

        cv2.imshow('Real-time Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
