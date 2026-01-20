import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("digit_classifier.keras")
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 784)
    return img

cap = cv2.VideoCapture(0)

print("Press 'c' to capture image")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    x, y, w, h = 200, 100, 200, 200
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Draw digit inside box", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        roi = frame[y:y+h, x:x+w]
        processed = preprocess_image(roi)

        prediction = model.predict(processed)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        print(f"Predicted Digit: {digit}, Confidence: {confidence:.2f}")

        plt.imshow(processed.reshape(28,28), cmap='gray')
        plt.title(f"Prediction: {digit} Confidence: {confidence:.2f}")
        plt.axis('off')
        plt.show()

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()