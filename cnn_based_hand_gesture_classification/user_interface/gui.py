import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

try:
    model = load_model("trained_model.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'update_wth_dataset_hand_gesture.h5' is in the current directory.")
    exit()

labels = ["Go", "Good", "Stop"]

MODEL_IMG_SIZE = 128

cap = cv2.VideoCapture(0)
cap.set(4,100)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)        
 
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        prediction_text = "No Hand Detected"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                h, w, c = image.shape

                x_min, y_min = w, h
                x_max, y_max = 0, 0
                
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
  
                buffer = 20  
                x_min = max(0, x_min - buffer)
                y_min = max(0, y_min - buffer)
                x_max = min(w, x_max + buffer)
                y_max = min(h, y_max + buffer)

                hand_crop = image[y_min:y_max, x_min:x_max]
                
                if hand_crop.size > 0:
       
                    input_img = cv2.resize(hand_crop, (MODEL_IMG_SIZE, MODEL_IMG_SIZE))
                    

                    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) 
                    
      
                    input_img = input_img.astype('float32') / 255.0
                    input_img = np.expand_dims(input_img, axis=0) # 

               
                    prediction = model.predict(input_img, verbose=0)
                    class_index = np.argmax(prediction)
                    confidence = np.max(prediction)
                    predicted_label = labels[class_index]
                    prediction_text = f"Gesture: {predicted_label} ({confidence*100:.2f}%)"

                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

        cv2.putText(image, prediction_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Real-Time Gesture Recognition', image)
          
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
