# CNN-Based Hand Gesture Classification
A deep learning–based hand gesture recognition system built using a Convolutional Neural Network (CNN) and deployed with a real-time detection interface using MediaPipe and OpenCV.
This project classifies three gestures: 
- Go
- Good
- Stop

The model is trained on a dataset of 4800 images and supports real-time webcam-based prediction.

# 1. Project Overview
This project aims to build an efficient and lightweight hand gesture classification system using a CNN model trained on RGB hand crop images. The system uses: 
 - CNN (for image-based classification)
 - MediaPipe Hands (for hand detection and landmark tracking)
 - OpenCV (for real-time webcam interface)

The architecture is optimized to deliver high accuracy with minimal computational overhead.

# 2. Features
- Custom dataset of 4800 labeled hand gesture images
- Model trained using TensorFlow/Keras CNN
- Real-time inference with:
  - Hand detection
  - Automatic bounding box extraction
  - Hand region cropping
  - Gesture prediction
  - Confidence score
- Real-time visualization with bounding boxes & landmarks

# Dataset details
**Total images:** 4,800
**Classes: 3**
- Go - 1600 images
- Good - 1600 images
- Stop - 1600 images
**Image Type: ** RGB
**Image Size:** 128 x 128

# 3. Model Architecture (CNN)
The model includes:
- Conv2D → MaxPooling2D layers
- Increasing filters: 32 → 64 → 128 → 256
- Flatten + Fully Connected Layers
- ReLU activation
- Softmax output layer (3 classes)

**Optimizer:** Adam (lr = 0.0005)

**Loss:** categorical_crossentropy

**Batch Size:** 32

**Epochs:** 10

# 4. Training Performance
| Metric               | value    |
|----------------------|----------|
| Training accurcay    | ~99%     |
| Validation Accuracy  | ~97%     |
| Training Loss        | very low |
| Validation Loss      | Stable   |

The model generalizes well without overfitting.

# 6. Real-Time Detection Pipeline:
The camera pipeline performs:
1. Capture webcam frame
2. Apply MediaPipe Hand Detection
3. Extract bounding box
4. Crop hand region
5. Resize & preprocess to 128×128
6. Convert to RGB
7. Normalize & predict
8. Display gesture label + confidence
9. Draw bounding box & landmarks


# Author
M Saim Ansari
