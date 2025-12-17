import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import joblib


x_test = np.load("dataset/X_test.npy")
x_train = np.load("dataset/X_train.npy")
y_test = np.load("dataset/y_test.npy")
y_train = np.load("dataset/y_train.npy")

print("Train shape:", x_train.shape)
print("Test shape :", x_test.shape)

model = RandomForestClassifier(n_estimators=100,max_depth=None,random_state=42)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(y_pred)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test,y_pred)

print("Accuracy Score", acc)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

joblib.dump(model, "trained_model.pkl")
