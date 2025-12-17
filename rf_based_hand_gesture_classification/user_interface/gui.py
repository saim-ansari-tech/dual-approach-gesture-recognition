import cv2
import mediapipe as mp
import numpy as np
import joblib
import sklearn


model = joblib.load("trained_model.pkl")

labels = ["Go", "Good", "Stop"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

def preprocess_landmarks(landmarks):
    coords = []
    for lm in landmarks.landmark:
        coords.append([lm.x, lm.y])

    coords = np.array(coords)
    coords -= coords[0]
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val

    return coords.flatten().reshape(1, -1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1) 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        vec = preprocess_landmarks(lm)
        pred = model.predict(vec)[0]
        gesture = labels[pred]

        cv2.putText(frame, gesture, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
