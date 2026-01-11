import csv
import numpy as np
import tensorflow as tf
import os

class PointHistoryClassifier:
    def __init__(self,
                 model_path="model/point_history_classifier/point_history_classifier.tflite",
                 label_path="model/point_history_classifier/point_history_classifier_label.csv"):
        # Load labels
        self.labels = self._load_labels(label_path)
        self.num_classes = len(self.labels)

        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Input/output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _load_labels(self, label_path):
        labels = []
        if os.path.exists(label_path):
            with open(label_path, newline="", encoding="utf-8-sig") as f:
                for row in csv.reader(f):
                    if row:
                        labels.append(row[0])
        return labels

    def __call__(self, data):
        """
        Predict gesture from normalized point history.

        Args:
            data (list or np.array): Flattened normalized point history.

        Returns:
            label_str (str): Gesture name (e.g., 'wave')
            confidence (float): Softmax probability
        """
        input_data = np.array([data], dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        pred_idx = int(np.argmax(output_data))
        confidence = float(output_data[pred_idx])
        label_str = self.labels[pred_idx] if pred_idx < len(self.labels) else "Unknown"

        return label_str, confidence
