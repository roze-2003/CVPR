import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('student_recognition_model.h5')
with open("labels.txt", "r") as f:
    labels = f.read().splitlines()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        roi_color = frame[y:y+h, x:x+w]
        img = cv2.resize(roi_color, (150, 150))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx]

        if confidence > 0.65:
            student_id = labels[class_idx]
            label = f"ID: {student_id} ({confidence*100:.0f}%)"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Attendance System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()