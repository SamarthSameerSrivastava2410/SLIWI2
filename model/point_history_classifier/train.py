import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import os

# ---------------- Paths ----------------
POINT_HISTORY_CSV = "model/point_history_classifier/point_history.csv"
POINT_HISTORY_TFLITE = "model/point_history_classifier/point_history_classifier.tflite"
POINT_HISTORY_LABEL_CSV = "model/point_history_classifier/point_history_classifier_label.csv"

# ---------------- Load CSV ----------------
def load_csv(file_path):
    X, y = [], []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")
    with open(file_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            *features, label = row
            X.append([float(v) for v in features])
            y.append(label)
    return np.array(X, np.float32), np.array(y)

# ---------------- Train ----------------
def train():
    X, y = load_csv(POINT_HISTORY_CSV)

    # Preserve label order as first seen in CSV
    labels = []
    for v in y:
        if v not in labels:
            labels.append(v)

    label_map = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_map[v] for v in y])
    y_oh = tf.keras.utils.to_categorical(y_idx, len(labels))

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_oh, test_size=0.2, shuffle=True
    )

    # Build model
    model = Sequential([
        Dense(256, activation="relu", input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(len(labels), activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32
    )

    # Save label CSV
    os.makedirs(os.path.dirname(POINT_HISTORY_LABEL_CSV), exist_ok=True)
    with open(POINT_HISTORY_LABEL_CSV, "w", newline="") as f:
        csv.writer(f).writerows([[l] for l in labels])

    # Convert to TFLite
    os.makedirs(os.path.dirname(POINT_HISTORY_TFLITE), exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(POINT_HISTORY_TFLITE, "wb") as f:
        f.write(tflite_model)

    print("✅ Dynamic model trained and saved as TFLite!")
    print(f"Labels: {labels}")

# ---------------- Run ----------------
if __name__ == "__main__":
    train()
