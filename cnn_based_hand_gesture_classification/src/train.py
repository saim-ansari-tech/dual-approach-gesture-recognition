from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.utils import to_categorical  # Used to convert class labels into one-hot encoded vectors for training the model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2


train_path = "dataset/train"
test_path = "dataset/test"

def load_dataset(path, img_size=128):
    X = []
    y = []


    allowed_classes = ['Go','Good', 'Stop']

    class_names = []
    for cname in os.listdir(path):
        if cname in allowed_classes:
            class_names.append(cname)

    class_names = sorted(class_names)

    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(path, class_name)

        for file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, file)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype('float32') / 255.0


            X.append(img)
            y.append(label)

    return np.array(X), np.array(y), class_names

X_train, y_train, class_names = load_dataset(train_path)


X_test, y_test, class_names = load_dataset(test_path)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

y_train = to_categorical(y_train,3)
y_test = to_categorical(y_test,3)

from keras.src.layers.pooling.max_pooling2d import MaxPooling2D
model = Sequential()

model.add(Conv2D(32,(3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3), activation='relu'))

model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dense(64,activation='relu'))

model.add(Dense(3, activation='softmax'))

model.summary()

from keras.src.ops import categorical_crossentropy
model.compile(optimizer=Adam(learning_rate=0.0005),loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(X_train,y_train,batch_size=32,epochs=10,validation_data=(X_test,y_test))

loss, accuracy = model.evaluate(X_test,y_test)
print(accuracy)

model.predict(X_test)

pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)
print(pred_classes)

true_classes = np.argmax(y_test, axis=1)
print(true_classes)

sum(pred_classes == true_classes) / len(true_classes)

model.save("model/trained_model.h5")

