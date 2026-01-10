import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ---------------- Paths ----------------
KEYPOINT_CSV = "model/keypoint_classifier/keypoint.csv"
KEYPOINT_TFLITE = "model/keypoint_classifier/keypoint_classifier.tflite"
KEYPOINT_LABEL_CSV = "model/keypoint_classifier/keypoint_classifier_label.csv"

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
def train_keypoint_classifier():
    X, y = load_csv_data(KEYPOINT_CSV)

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
    with open(KEYPOINT_TFLITE, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {KEYPOINT_TFLITE}")

    # Save labels CSV
    with open(KEYPOINT_LABEL_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        for label in unique_labels:
            writer.writerow([label])
    print(f"Label CSV saved to {KEYPOINT_LABEL_CSV}")

if __name__ == "__main__":
    train_keypoint_classifier()
