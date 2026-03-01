import cv2
import numpy as np
import time
from collections import Counter
from tensorflow.keras.models import load_model

# -----------------------------
# LOAD MODEL (MobileNet)
# -----------------------------
model = load_model("mobilenet_emotion_model.keras")

# Emotion labels (same order as training)
labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# -----------------------------
# FACE DETECTOR
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# SMOOTHING BUFFER
# -----------------------------
buffer = []
last_update = time.time()
emotion_text = "Detecting..."
emotion_conf = 0

WINDOW_NAME = "Emotion Detection"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale ONLY for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:

        # ✅ USE RGB FACE FOR MOBILENET
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (96, 96))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)  # (1, 96, 96, 3)

        # -----------------------------
        # PREDICTION
        # -----------------------------
        preds = model.predict(face, verbose=0)[0]
        label = np.argmax(preds)
        conf = preds[label] * 100

        # Buffer smoothing
        if conf > 35:
            buffer.append(label)

        if len(buffer) > 20:
            buffer.pop(0)

        if time.time() - last_update > 0.6 and buffer:
            most_common = Counter(buffer).most_common(1)[0][0]
            emotion_text = labels[most_common]
            emotion_conf = conf
            last_update = time.time()

        # -----------------------------
        # DRAW UI
        # -----------------------------
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{emotion_text} ({emotion_conf:.1f}%)",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    cv2.imshow(WINDOW_NAME, frame)

    # PRESS Q TO EXIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # CLICK ❌ TO EXIT
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()