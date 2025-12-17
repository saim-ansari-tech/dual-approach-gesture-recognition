# Hand Gesture Recognition Using Random Forest (MediaPipe Landmarks
This project implements a lightweight and fast hand gesture recognition system using:
- MediaPipe Hands for hand landmark detection
- Random Forest Classifier for gesture prediction
- OpenCV for real-time webcam interface

This approach uses 2D hand landmarks instead of images, making the model extremely fast and suitable for low-power devices.

# 1. Project Overview
Instead of using pixel images (CNN-based approach), this project extracts 21 hand landmarks (x, y) using MediaPipe, normalizes them, and feeds them into a Random Forest classifier.
This reduces:
- Model size
- Computation time
- Training complexity

It enables real-time gesture recognition with high speed.

# 2. Features
- Extracts 21 hand landmarks × (x,y) = 42 numerical features
- Uses Random Forest for multi-class classification
- Real-time gesture recognition from webcam
- Extremely fast inference 
- Dataset stored in .npy arrays for direct training

# 3. Dataset Details
Dataset generated through MediaPipe landmark extraction.
**Shape:**
Train shape: (4449, 42)
Test shape : (1148, 42)
**Classes:
- Go
- Good
- Stop

**Saved files**
X_train.npy
X_test.npy
y_train.npy
y_test.npy

# 4. Model Training (Random Forest)
The Random Forest model uses:
- n_estimators = 100
- max_depth = None
- random_state = 42

# Training Script Summary
- Loads .npy feature dataset
- Trains Random Forest classifier
- Evaluates using accuracy + classification report
- Saves final model as .pkl

# 5. Training Performance
- Accuracy: ~97%
- Classification report:
  -  Precision:
      -  Go: ~97%
      -  Good: ~97%
      -  Stop: ~99%
  -  Recall:
      - Go: ~97%
      - Good: ~95%
      - Stop: 1.00
  - F1-score:
      - Go: ~97%
      - Good: ~96%
      - Stop ~99%
  
The model performs well due to normalized landmark features.

# 6. Real-Time Detection Pipeline
The webcam-based prediction performs:
1. OpenCV captures frame
2. MediaPipe detects hand & extracts 21 landmarks
3. Landmarks normalized:
    - Hand centered (subtract landmark 0)
    - Scaled between −1 to +1
4. Flatten to 42-dim vector
5. Random Forest predicts gesture
6. Display predicted label on screen

# 9. Advantages of Landmark + Random Forest Approach
- No need for GPU
- Much faster than CNN
- Works even with low-quality cameras
- Requires small dataset to achieve good accuracy
- Great for robotics and embedded systems

# Author
Muhammad Saim Ansari
Robotics & AI Student
