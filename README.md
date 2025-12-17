# Hand Gesture Recognition System
This project implements real-time hand gesture recognition using two AI techniques:

1 - Convolutional Neural Network (CNN)

2 - Random Forest Classifier (RF)

It predicts gestures based on hand landmark features extracted using MediaPipe, enabling gesture-based control for robotics, automation, and human–computer interaction.

# Features
- Real-time hand detection
- CNN model trained on gesture images
- Random Forest model trained on landmark coordinates
- Multi-class gesture classification
- High accuracy and stable prediction performance

# Techniques Used

1. CNN-Based Gesture Classification
   - Input: gesture images
   - Layers: Conv2D, MaxPooling, Flatten, Dense
   - Output: gesture label
2. Random Forest Classifier
   - Input: 21 MediaPipe keypoints → converted into 42-feature vector
   - Fast and lightweight
   - Works without heavy computation

# Project Structure
dual-approach-gesture-recognition
  |- Hand Gesture Recognition using CNN
     |- dataset:
        train
        test
     |- model:
        cnn_model.ipynb
        trained_model.h5
     |- src:
        train.py
        preprocess.py
        realtime_test.py
     |- README.md 
     |- requirements.txt  
     
  |- Hand Gesture Recognition using RF
     |- dataset:
        X_test.npy
        X_train.npy
        labels.txt
        y_test.npy
        y_train.npy
     |- model:
        rf_model.py
        trained_model.h5
     |- src:
        train.py
        preprocess.py
        realtime_test.py
     |- README.md 
     |- requirements.txt  
        


