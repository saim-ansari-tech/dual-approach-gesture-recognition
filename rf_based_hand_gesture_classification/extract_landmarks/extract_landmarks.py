import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm


TRAIN_PATH = "dataset/train"
TEST_PATH  = "dataset/test"
OUT_PATH   = "dataset_land_mark/land_mark_dataset"

IMG_SIZE = 128

ALLOWED_CLASSES = ['Go', 'Good', 'Stop']

os.makedirs(OUT_PATH, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

def preprocess_landmarks(landmarks):
    """
    Convert landmarks to normalized relative coordinates
    Output: (42,) vector
    """
    coords = []

    for lm in landmarks.landmark:
        coords.append([lm.x, lm.y])

    coords = np.array(coords)

  
    coords -= coords[0]

    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val

    return coords.flatten()


def build_dataset(base_path):
    X, y = [], []

    class_names = sorted(
        [c for c in os.listdir(base_path) if c in ALLOWED_CLASSES]
    )

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(base_path, class_name)

        for file in tqdm(os.listdir(class_dir), desc=f"{class_name}"):
            img_path = os.path.join(class_dir, file)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img)

            if not results.multi_hand_landmarks:
                continue

            landmarks = results.multi_hand_landmarks[0]
            vector = preprocess_landmarks(landmarks)

            X.append(vector)
            y.append(label)

    return np.array(X), np.array(y), class_names

print("Processing TRAIN set...")
X_train, y_train, labels = build_dataset(TRAIN_PATH)

print("Processing TEST set...")
X_test, y_test, _ = build_dataset(TEST_PATH)


np.save(os.path.join(OUT_PATH, "X_train.npy"), X_train)
np.save(os.path.join(OUT_PATH, "y_train.npy"), y_train)
np.save(os.path.join(OUT_PATH, "X_test.npy"), X_test)
np.save(os.path.join(OUT_PATH, "y_test.npy"), y_test)

with open(os.path.join(OUT_PATH, "labels.txt"), "w") as f:
    for l in labels:
        f.write(l + "\n")

print("\n Landmark dataset created")
print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)
