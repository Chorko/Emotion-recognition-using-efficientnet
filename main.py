import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pyttsx3

# Path to the haarcascade file and saved model
face_classifier_path = 'D:/Emotion_recognition_using_efficientnet/haarcascade_frontalface_default.xml'
model_path = 'D:/Emotion_recognition_using_efficientnet/models/best_model.keras'

# Load the face classifier and emotion model
face_classifier = cv2.CascadeClassifier(face_classifier_path)
classifier = load_model(model_path)

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to detect and predict emotion
def predict_emotion(face, gray_frame):
    roi_gray = gray_frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

    if np.sum([roi_gray]) != 0:
        roi = roi_gray.astype('float32') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Predict emotion
        prediction = classifier.predict(roi)[0]
        max_index = np.argmax(prediction)
        label = emotion_labels[max_index]
        return label
    return None

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Variable to hold the last spoken emotion
last_spoken_emotion = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        label = predict_emotion((x, y, w, h), gray)
        
        if label:
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Check if the label is different from the last spoken emotion
            if label != last_spoken_emotion:
                # Convert emotion to speech using pyttsx3
                engine.say(f"You seem to be {label}")
                engine.runAndWait()
                last_spoken_emotion = label

    # Display the resulting frame
    cv2.imshow('Emotion Detector', frame)

    # Press 'q' to quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()




