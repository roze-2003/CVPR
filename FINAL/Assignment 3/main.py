import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# 1. SET WORKING DIRECTORY
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 2. LOAD LABELS (Needed to define the output layer)
with open("labels.txt", "r") as f:
    labels = f.read().splitlines()
num_classes = len(labels)

# 3. MANUALLY RECONSTRUCT THE ARCHITECTURE
# This creates the "skeleton" so Keras doesn't have to guess
print("🔄 Building model architecture...")
base_model = MobileNetV2(weights=None, include_top=False, input_shape=(150, 150, 3))

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'), # Use 128 if you used 128 in Colab
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])

# 4. LOAD THE WEIGHTS (This bypasses the '2 input tensors' error)
# Note: You can load weights from a .keras OR a .h5 file here
MODEL_PATH = 'final_synchronized_model.keras' 

try:
    model.load_weights(MODEL_PATH)
    print("✅ Success! Weights injected into the local architecture.")
except Exception as e:
    print(f"❌ Weight mismatch: {e}")
    print("TIP: If this fails, change the Dense layer from 256 to 128.")
    exit()

# 5. WEBCAM LOGIC
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (150, 150))
        
        # Proper MobileNetV2 Preprocessing
        img_array = np.expand_dims(face_resized, axis=0).astype(np.float32)
        img_preprocessed = preprocess_input(img_array)

        # Run Prediction
        prediction = model.predict(img_preprocessed, verbose=0)
        idx = np.argmax(prediction)
        confidence = prediction[0][idx]

        # UI Feedback
        if confidence > 0.80:
            label = f"ID: {labels[idx]} ({confidence*100:.1f}%)"
            color = (0, 255, 0)
        else:
            label = f"Unknown ({confidence*100:.1f}%)"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Attendance System - Final Fix', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()