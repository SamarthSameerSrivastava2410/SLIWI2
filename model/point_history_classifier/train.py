import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ---------------- Paths ----------------
POINT_HISTORY_CSV = "model/point_history_classifier/point_history.csv"
POINT_HISTORY_TFLITE = "model/point_history_classifier/point_history_classifier.tflite"
POINT_HISTORY_LABEL_CSV = "model/point_history_classifier/point_history_classifier_label.csv"

# ---------------- Load CSV ----------------
def load_csv_data(file_path):
    X, y = [], []
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            *features, label = row
            X.append([float(x) for x in features])
            y.append(int(label))
    return np.array(X), np.array(y)

# ---------------- Train and convert ----------------
def train_point_history_classifier():
    X, y = load_csv_data(POINT_HISTORY_CSV)

    # Dynamic labels
    unique_labels = sorted(list(set(y)))
    label_map = {l:i for i,l in enumerate(unique_labels)}
    y_mapped = np.array([label_map[v] for v in y])
    num_classes = len(unique_labels)
    input_dim = X.shape[1]

    y_onehot = tf.keras.utils.to_categorical(y_mapped, num_classes)

    # Model
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y_onehot, epochs=20, batch_size=32, verbose=1)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(POINT_HISTORY_TFLITE, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {POINT_HISTORY_TFLITE}")

    # Save labels CSV
    with open(POINT_HISTORY_LABEL_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        for label in unique_labels:
            writer.writerow([label])
    print(f"Label CSV saved to {POINT_HISTORY_LABEL_CSV}")

if __name__ == "__main__":
    train_point_history_classifier()
